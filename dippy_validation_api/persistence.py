import os
import logging
from typing import Optional

import supabase
from supabase import create_client
import pandas as pd
from datetime import datetime, timedelta


class SupabaseState:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supabase_url = os.environ["SUPABASE_URL"]
        self.supabase_key = os.environ["SUPABASE_KEY"]
        self.client = create_client(self.supabase_url, self.supabase_key)

    def supa_client(self):
        return self.client

    def update_leaderboard_status(self, hash, status, notes=""):
        try:
            response = (
                self.client.table("leaderboard")
                .update(
                    {"status": status, "notes": notes},
                )
                .eq("hash", hash)
                .execute()
            )
            return response
        except Exception as e:
            self.logger.error(f"Error updating leaderboard status for {hash}: {e}")
            return None

    def upsert_and_return(self, row, hash: str):
        self.upsert_row(row)
        self.logger.info(f"upserted row {row}")
        return self.get_json_result(hash)

    def upsert_row(self, row):
        self.client.table("leaderboard").upsert(row).execute()

    def search_record_with_model_hash(self, model_hash_value: str, block: int):
        if not model_hash_value:  # Check if the provided model_hash_value is empty
            return None
        if not block:
            return None

        # Query the table to check for the existence of the model_hash
        response = (
            self.client.table("leaderboard")
            .select("*, minerboard(*)")
            .neq("model_hash", "")
            .neq("model_hash", None)
            .eq("model_hash", model_hash_value)
            .execute()
        )
        if len(response.data) < 1:
            return None
        for record in response.data:
            model_hash_matches = model_hash_value == record.get("model_hash")
            minerboard_entry = record.get("minerboard", [{}])
            exists_in_older_block = block > minerboard_entry[0].get("block", block)
            if model_hash_matches and exists_in_older_block:
                return record

        return None

    def last_uploaded_model(self, miner_hotkey: str):
        data = self.client.table("minerboard").select("*, leaderboard(status)").eq("hotkey", miner_hotkey).execute()
        if len(data.data) > 0:
            return data.data[0]
        return None

    def update_minerboard_status(
        self,
        minerhash: str,
        uid: int,
        hotkey: str,
        block: int,
    ):
        try:
            response = (
                self.client.table("minerboard")
                .upsert(
                    {"hash": minerhash, "uid": uid, "hotkey": hotkey, "block": block},
                    returning="minimal",
                )
                .execute()
            )
            return response
        except Exception as e:
            self.logger.error(f"Error updating minerboard status for {hash}: {e}")
            return None

    def minerboard_fetch(self):
        try:
            response = self.client.table("minerboard").select("*, leaderboard(*)").execute()
            return response.data
        except Exception as e:
            self.logger.error(f"Error updating minerboard : {e}")
            return None

    def get_json_result(self, hash):
        try:
            response = self.client.table("leaderboard").select("*").eq("hash", hash).execute()
            if len(response.data) > 0:
                result = {
                    "score": {
                        "model_size_score": response.data[0]["model_size_score"],
                        "qualitative_score": response.data[0]["qualitative_score"],
                        "latency_score": response.data[0]["latency_score"],
                        "vibe_score": response.data[0]["vibe_score"],
                        "total_score": response.data[0]["total_score"],
                        "coherence_score": response.data[0]["coherence_score"],
                        "creativity_score": response.data[0]["creativity_score"],
                        "post_eval_score": response.data[0]["post_eval_score"],
                    },
                    "details": {
                        "model_hash": response.data[0]["model_hash"],
                    },
                    "status": response.data[0]["status"],
                }
                return result
            self.logger.error(f"Did not find leaderboard entry for hash {hash}: {response}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching leaderboard entry from database: {e}")
            raise e
        r

    def get_internal_result(self, hash):
        try:
            response = self.client.table("leaderboard").select("*").eq("hash", hash).execute()
            if len(response.data) > 0:
                return response.data[0]
            self.logger.error(f"Did not find leaderboard entry : {response}")
        except Exception as e:
            self.logger.error(f"Error fetching leaderboard entry from database: {e}")
        return None

    def remove_record(self, hash: str):
        try:
            status = self.client.table("leaderboard").delete(returning="minimal").eq("hash", hash).execute()
            return status
        except Exception as e:
            self.logger.error(f"could not delete record {str(e)}")
            return None

    def upsert_row(self, row):
        if "timestamp" in row:
            row["timestamp"] = row["timestamp"].isoformat()
        try:
            response = self.client.table("leaderboard").upsert(row).execute()
            return response
        except Exception as e:
            self.logger.error(f"Error updating row in Supabase: {e}")
            return None

    def get_top_completed(self):
        response = (
            self.client.table("leaderboard")
            .select("*")
            .eq("status", "COMPLETED")
            .order("total_score", desc=True)
            .limit(10)
            .execute()
        )
        return response.data

    def get_leaderboard(self):
        try:
            response = (
                self.client.table("leaderboard")
                .select("*")
                .eq("status", "COMPLETED")
                .order("total_score", desc=True)
                .limit(10)
                .execute()
            )
            leaderboard = pd.DataFrame(response.data)
            leaderboard = leaderboard.fillna(value=0)
            leaderboard = leaderboard.sort_values(by="total_score", ascending=False)
            return leaderboard.to_dict(orient="records")
        except Exception as e:
            self.logger.error(f"Error fetching leaderboard from Supabase: {e}")
            return None

    def get_next_model_to_eval(self):
        try:
            response = (
                self.client.table("minerboard")
                .select("*, leaderboard(*)")
                .order("block", desc=False)
                .execute()
            )
            if len(response.data) == 0:
                return None
            # Filter for only QUEUED status entries
            queued_entries = [entry for entry in response.data if entry.get('leaderboard', {}).get('status') == 'QUEUED']
            if len(queued_entries) == 0:
                return None
            return queued_entries[0]['leaderboard']
        except Exception as e:
            self.logger.error(f"Error fetching next model to evaluate: {e}")
            return None

    def get_failed_model_to_eval(self):
        try:
            response = (
                self.client.table("leaderboard")
                .select("*")
                .eq("status", "FAILED")
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )
            if len(response.data) == 0:
                return None
            return response.data[0]
        except Exception as e:
            self.logger.error(f"Error fetching next model to evaluate: {e}")
            return None


def debug():
    supabaser = SupabaseState()
    x = supabaser.last_uploaded_model("x")
    print(x)
    return


if __name__ == "__main__":
    debug()
