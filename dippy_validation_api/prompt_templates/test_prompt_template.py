from jinja2 import Environment, FileSystemLoader

# Set up the Jinja environment and loader
env = Environment(loader=FileSystemLoader('.'))

# Iterate over all templates in the environment and render them with dummy data
for template_name in env.list_templates(extensions=["jinja"]):
    template = env.get_template(template_name)

    # Define dummy data for the 'messages' variable and 'add_generation_prompt' flag
    dummy_data = {
        'messages': [
            {'role': 'user', 'content': 'Hello, how are you?'},
            {'role': 'assistant', 'content': 'I am fine, thank you!'},
            {'role': 'user', 'content': 'What are you doing?'},
        ],
        'add_generation_prompt': True,
        'bos_token': '<BOS>',
        'eos_token': '<EOS>',
        'include_beginning_of_conversation': True
    }

    # Render the template with the dummy data
    rendered_template = template.render(dummy_data)

    # Print the template name and the rendered template
    print(f"Template: {template_name}")
    print(rendered_template)
    print("-" * 40)  # Separator between templates for readability
