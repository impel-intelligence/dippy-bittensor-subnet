# Advanced Validator setup

## Prerequisites
- Docker
- Existing validator prerequisites (python, `wandb`)
- 

## Environment Setup
```shell
SUPABASE_KEY=<your_supabase_project_api_key>
SUPABASE_URL=<your_supabase_project_url>
ADMIN_KEY=<your_admin_access_key>
HF_ACCESS_TOKEN=<your_huggingface_access_token>
HF_USER=<your_huggingface_username>
DIPPY_KEY=<your_dippy_bot_access_key>
```

## Steps
1. Navigate to `dippy_validation_api` from the root of the repository.
2. Start the validation api via the script `./start_validation_service_solo.sh`
3. Start the model queue via the script ``