# Advanced Validator setup

## Prerequisites
- Docker
- Existing validator prerequisites (python)
- Access tokens for following services below

## Environment Setup
```shell
SUPABASE_KEY=<your_supabase_project_api_key>
SUPABASE_URL=<your_supabase_project_url>
ADMIN_KEY=<your_admin_access_key>
HF_ACCESS_TOKEN=<your_huggingface_access_token>
HF_USER=<your_huggingface_username>
DIPPY_KEY=<your_dippy_bot_access_key>
OPENAI_API_KEY=<your_openai_api_key>
```

## Steps
1. Navigate to `dippy_validation_api` from the root of the repository.
2. Start the validation api via the script `./start_validation_service.sh`
3. Start the model queue via the command `python neurons/model_queue.py` (modify endpoint data as needed) from repository root
4. Run validator with commands specified to point to local validation api instance