#!/bin/bash

# Function to kill a process and its subprocesses
kill_process_group() {
    local pid=$1
    local pgrp=$(ps -o pgid= $pid | grep -o '[0-9]*')
    echo "Stopping process group: $pgrp"
    kill -- -$pgrp
    rm -f "log/$2.pid"
}

# Kill the validation_api
echo "Stopping validation_api..."
kill $(cat log/validation_api.pid)
rm -f log/validation_api.pid

# Kill the eval_score_api and its subprocesses
echo "Stopping eval_score_api..."
kill_process_group $(cat log/eval_score_api.pid) "eval_score_api"

# Kill the vibe_score_api and its subprocesses
echo "Stopping vibe_score_api..."
kill_process_group $(cat log/vibe_score_api.pid) "vibe_score_api"

echo "All APIs and their subprocesses have been stopped and PID files removed."