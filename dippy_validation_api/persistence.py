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
                .upsert(
                    {"hash": hash, "status": status, "notes": notes},
                    returning="minimal",
                )
                .execute()
            )
            return response
        except Exception as e:
            self.logger.error(f"Error updating leaderboard status for {hash}: {e}")
            return None

    def update_minerboard_status(self,
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
            self.logger.error(f"Error updating leaderboard status for {hash}: {e}")
            return None

    def minerboard_fetch(self):
        try:
            response = (
                self.client.table("minerboard").select("*, leaderboard(*)").execute()
            )
            return response.data
        except Exception as e:
            self.logger.error(f"Error updating minerboard : {e}")
            return None

    def update_model_hash(self, hash: str, model_hash: str):
        try:
            response = (
                self.client.table("leaderboard")
                .update(
                    {"hash": hash, "model_hash": model_hash},
                    returning="minimal",
                )
                .execute()
            )
            return response
        except Exception as e:
            self.logger.error(f"Error updating leaderboard status for {hash}: {e}")
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

                    },
                    "details": {
                        "block": response.data[0]["block"],
                        "model_hash": response.data[0]["model_hash"],
                    },
                    "status": response.data[0]["status"],
                }
                return result
            raise RuntimeError("No record QUEUED")
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

    def update_row(self, row):
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
            leaderboard = leaderboard.fillna(value=-1)
            leaderboard = leaderboard.sort_values(by="total_score", ascending=False)
            # # filter out entries older than two weeks
            # two_weeks_ago = datetime.now() - timedelta(weeks=2)
            # # Convert the 'timestamp' column to datetime format. If parsing errors occur, 'coerce' will replace problematic inputs with NaT (Not a Time)
            # leaderboard['timestamp'] = pd.to_datetime(leaderboard['timestamp'], errors='coerce', utc=True)
            # leaderboard = leaderboard[(leaderboard['timestamp'].dt.tz_convert(None) > two_weeks_ago) | (leaderboard.index < 1000)]
            # leaderboard = leaderboard.sort_values(by='total_score', ascending=False)
            return leaderboard.to_dict(orient="records")
        except Exception as e:
            self.logger.error(f"Error fetching leaderboard from Supabase: {e}")
            return None

    def get_next_model_to_eval(self):
        try:
            response = (
                self.client.table("leaderboard")
                .select("*")
                .eq("status", "QUEUED")
                .order("timestamp", desc=False)
                .limit(1)
                .execute()
            )
            if len(response.data) == 0:
                return None
            return response.data[0]
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
