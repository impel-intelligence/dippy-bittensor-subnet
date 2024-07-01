from dippy_validation_api.persistence import SupabaseState
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.hf_api import HfApi, RepositoryNotFoundError

hf_api = HfApi()


def clean_up():
    def check_repository_exists(repo_id):
        try:
            hf_api.repo_info(repo_id)
            return True
        except RepositoryNotFoundError:
            return False
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False

    supa = SupabaseState()
    records = supa.get_top_completed()
    for record in records:
        if "repo_namespace" not in record:
            continue
        id = f'{record["repo_namespace"]}/{record["repo_name"]}'
        exists = check_repository_exists(id)
        if not exists:
            print(f"remove because not exists : {id} {record}")
            supa.remove_record(record["hash"])


# Manually run to clean up entries in leaderboard that no longer exist in huggingface
if __name__ == "__main__":
    clean_up()
