#!/bin/bash

# Check if a batch number is provided and if it is within the valid range (0-5)
if [ -z "$1" ] || [ "$1" -lt 0 ] || [ "$1" -gt 5 ]; then
    echo "Please provide a batch number between 0 and 5."
    exit 1
fi

# Batch number
BATCH_NUM=$1

# Function to kill a process and its subprocesses
kill_process_group() {
    local pid=$1
    local pgrp=$(ps -o pgid= $pid | grep -o '[0-9]*')
    echo "Stopping process group: $pgrp"
    kill -- -$pgrp
    rm -f "log/$2_${BATCH_NUM}.pid"
}

# Kill the validation_api
echo "Stopping validation_api..."
kill $(cat log/validation_api_${BATCH_NUM}.pid)
rm -f log/validation_api_${BATCH_NUM}.pid

# Kill the eval_score_api and its subprocesses
echo "Stopping eval_score_api..."
kill_process_group $(cat log/eval_score_api_${BATCH_NUM}.pid) "eval_score_api"

# Kill the vibe_score_api and its subprocesses
echo "Stopping vibe_score_api..."
kill_process_group $(cat log/vibe_score_api_${BATCH_NUM}.pid) "vibe_score_api"

echo "All APIs and their subprocesses for batch $BATCH_NUM have been stopped and PID files removed."
