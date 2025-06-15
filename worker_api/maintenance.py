from worker_api.persistence import SupabaseState
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

import os
from datetime import datetime
import shutil
from operator import itemgetter

def clean_old_folders(path: str = "/tmp/modelcache", keep_n: int = 4):
    # Get all directories and their modification times
    folders = []
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path) and item not in ['.', '..']:
            mod_time = os.path.getmtime(full_path)
            folders.append((full_path, mod_time))

    # Sort by modification time (newest first)
    folders.sort(key=itemgetter(1), reverse=True)

    # Keep the N most recent folders, delete the rest
    folders_to_delete = folders[keep_n:]

    # Show what will be deleted
    if folders_to_delete:
        print(f"\nThe following {len(folders_to_delete)} folders will be deleted:")
        for folder, mod_time in folders_to_delete:
            print(f"{folder} (modified: {datetime.fromtimestamp(mod_time)})")

        
        for folder, _ in folders_to_delete:
            try:
                shutil.rmtree(folder)
                print(f"Deleted: {folder}")
            except Exception as e:
                print(f"Error deleting {folder}: {e}")
    else:
        print(f"No folders to delete. Only found {len(folders)} folders total.")

# Manually run to clean up entries in leaderboard that no longer exist in huggingface
if __name__ == "__main__":
    clean_up()
