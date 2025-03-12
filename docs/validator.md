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
OPENAI_API_KEY=<your_openai_api_key>
```

## Steps
1. Navigate to `dippy_validation_api` from the root of the repository.
2. Start the validation api via the script `./start_validation_service.sh`
3. Start the model queue via the command `python neurons/model_queue.py` (modify endpoint data as needed) from repository root
4. Run validator with commands specified to point to local validation api instance


### Running the model evaluation API (Optional, not recommended)

**Note**: Currently (Jan 17 2025) there are some issues with the local evaluation api. We recommend using the remote validation api temporarily.

Starting a validator using your local validator API requires starting validator with `--use-local-validation-api` flag. 
Additionally, a model queue is required to push models to the validation api.

**Note**: Validator API needs to be installed in a different venv than validator due to `pydantic` version conflict. 


### Requirements
- Python 3.9+
- Linux

#### Setup

Install Git Lfs if not installed.
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

To start, clone the repository and `cd` into it:
```bash
git clone https://github.com/impel-intelligence/dippy-bittensor-subnet.git
cd dippy-bittensor-subnet
python3 -m venv model_validation_venv
source model_validation_venv/bin/activate
model_validation_venv/bin/pip install -e . --no-deps
model_validation_venv/bin/pip install -r requirements.api.txt
```