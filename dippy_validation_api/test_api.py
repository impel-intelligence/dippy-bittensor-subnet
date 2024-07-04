import os
import requests
import dotenv
from supabase import create_client
from utilities.validation_utils import regenerate_hash


llm = "Manavshah/llama-test"

dotenv.load_dotenv("../.env")


supabase_url = os.environ["SUPABASE_URL"]
supabase_key = os.environ["SUPABASE_KEY"]
try:
    supabase_client = create_client(supabase_url, supabase_key)
except Exception as e:
    print(f"Failed to create Supabase client: {e}. Leaderboard will only be saved locally.")
    supabase_client = None


def test_evaluate_model():
    # Define the request payload
    request_payload = {
        "admin_key": os.environ["ADMIN_KEY"],
        "repo_namespace": llm.split("/")[0],
        "repo_name": llm.split("/")[1],
        "chat_template_type": "zephyr",
        "hash": None,
        "revision": "main",
        "competition_id": "test",
    }

    # generate the hash based on regenerate_hash
    request_payload["hash"] = str(
        regenerate_hash(
            request_payload["repo_namespace"],
            request_payload["repo_name"],
            request_payload["chat_template_type"],
            request_payload["competition_id"],
        )
    )

    print("Request payload:", request_payload)

    # Send a POST request to the evaluate_model endpoint
    response = requests.post("http://localhost:8000/evaluate_model", json=request_payload)
    # Check that the response status code is 200 (OK)
    if response.status_code != 200:
        print(response.text)

    # Optionally, check the response content
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] in ["QUEUED", "RUNNING", "COMPLETED", "FAILED"]

    print(response_data)

    # If you want to check for a specific response structure, you can do so here
    # For example, if you expect a certain JSON structure, you can compare it like this:
    expected_response_structure = {
        "score": {
            "model_size_score": float,
            "qualitative_score": float,
            "latency_score": float,
            "vibe_score": float,
            "total_score": float,
        },
        "status": str,
    }

    for key, value_type in expected_response_structure.items():
        assert key in response_data
        if key == "score":
            for score_key, score_type in expected_response_structure[key].items():
                assert score_key in response_data[key]
                assert isinstance(response_data[key][score_key], score_type)
        else:
            assert isinstance(response_data[key], value_type)

    print(response_data)


# Run the test function
test_evaluate_model()
