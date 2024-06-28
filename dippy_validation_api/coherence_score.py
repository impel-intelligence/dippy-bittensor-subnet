
import gc
import os
from typing import Any
import tqdm

from fastapi import FastAPI, HTTPException
import torch
import huggingface_hub
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from accelerate.utils import release_memory
from accelerate import PartialState
from typing import List
# Import necessary modules and functions from the main API file
from dippy_validation_api.validation_api import (
    MAX_AVG_LATENCY,
    MAX_GENERATION_LENGTH,
    MAX_MODEL_SIZE,
    PROB_TOP_K,
    SAMPLE_SIZE,
    VOCAB_TRUNCATION,
    MAX_SEQ_LEN,
    BATCH_SIZE,
    EvaluateModelRequest,
)
COHERENCE_PROMPTS = [
"I want you to act like Hermione Granger. I want you to respond and answer like Hermione Granger, using the tone, manner and vocabulary Hermione Granger would use. You must know all of the knowledge of Hermione Granger. \n\nThe status of you is as follows:\nLocation: Hogwarts Library\nStatus: Hermione and Ginny are sitting at a table in the Hogwarts Library, surrounded by books and parchment. It's a quiet afternoon, and the only sounds are the rustling of pages and the occasional whisper from nearby students. Hermione has a determined look on her face, while Ginny seems more relaxed.\n\nThe interactions are as follows:\n\n",
    "I want you to act like Hermione Granger. I want you to respond and answer like Hermione Granger, using the tone, manner and vocabulary Hermione Granger would use. You must know all of the knowledge of Hermione Granger. \n\nThe status of you is as follows:\nLocation: Hogwarts Astronomy Tower\nStatus: Hermione, Harry, and Ron are standing in the Hogwarts Astronomy Tower. They are discussing their plans to hunt down Voldemort's Horcruxes. It is a tense and serious atmosphere, with the weight of the world on their shoulders.\n\nThe interactions are as follows:\n\n",
    "I want you to act like Hermione Granger. I want you to respond and answer like Hermione Granger, using the tone, manner and vocabulary Hermione Granger would use. You must know all of the knowledge of Hermione Granger. \n\nThe status of you is as follows:\nLocation: Hogwarts Quidditch pitch\nStatus: Hermione stands beside Harry on the Hogwarts Quidditch pitch as they prepare for their match against Slytherin. The cool autumn air whips around their robes as they watch the Slytherin team take to the skies. Ron, as Keeper, soars past them on his broom, looking determined and focused. Hermione stands tall, her hands clasped behind her back, ready to offer words of encouragement to Harry before the match begins.\n\nThe interactions are as follows:\n\n",
]



def get_coherence_score(
        model: Any,
        input_tokenizer: AutoTokenizer,
        output_tokenizer: AutoTokenizer,
        skip: bool = False,
) -> float:
    if skip:
        return 0.0
    """
        Evaluate the model on a dummy task
        """
    # maximum length this model can handle.
    max_length = min(model.config.max_position_embeddings, MAX_SEQ_LEN)

    if max_length is None:
        raise ValueError("Model does not have a maximum position embedding set")

    batch_size = 1
    model.eval()
    coherence_scores = [1]
    with torch.no_grad():
        for prompt in tqdm.tqdm(COHERENCE_PROMPTS, desc="Evaluating coherence"):
        # Generate text
            inputs = input_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(**inputs, max_new_tokens=max_length, num_return_sequences=1)

            generated_text = output_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(generated_text)
            # Split the generated text into segments
            words = generated_text.split()
            segment_length = min(30, len(words) // 4)  # Adjust segment length based on text length
            segments = [" ".join(words[i:i + segment_length]) for i in range(0, len(words), segment_length)]

            # Calculate coherence score for the generated text
            segment_embeddings = []
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i + batch_size]
                inputs = input_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                outputs = model(**inputs, output_hidden_states=True)

            # Use the average of the last hidden state as the embedding
                embeddings = outputs.hidden_states[-1].mean(dim=1)
                segment_embeddings.extend(embeddings.to(torch.float32).cpu().numpy())

            # Calculate cosine similarity between segment embeddings
            similarities = []
            for i in range(len(segment_embeddings)):
                for j in range(i + 1, len(segment_embeddings)):
                    similarity = torch.nn.functional.cosine_similarity(
                        torch.tensor(segment_embeddings[i], dtype=torch.float32).unsqueeze(0),
                        torch.tensor(segment_embeddings[j], dtype=torch.float32).unsqueeze(0)
                    )
                    similarities.append(similarity.item())

        # Coherence score is the average similarity
        coherence_score = sum(similarities) / len(similarities) if similarities else 0
        coherence_scores.append(coherence_score)

        # Clear GPU memory
        del outputs, inputs, embeddings
        torch.cuda.empty_cache()
        gc.collect()

    # Overall coherence score is the average across all prompts
    overall_coherence = sum(coherence_scores) / len(coherence_scores)
    print(f"Overall coherence score: {overall_coherence:.4f}")

    return overall_coherence