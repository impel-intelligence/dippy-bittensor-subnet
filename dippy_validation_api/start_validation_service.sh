#!/bin/bash

# Create a log directory if it doesn't exist
mkdir -p log

# Virtual environment name
VENV_NAME="model_validation_venv"

# Function to restart a service
restart_service() {
    local service_name=$1
    local service_script=$2
    local log_file=$3
    local pid_file=$4
    local loop_pid_file=$5

    echo $$ > $loop_pid_file  # Store the PID of the loop process

    while true; do
        echo "Starting $service_name..."
        ./../$VENV_NAME/bin/python3 $service_script >> $log_file 2>&1 &
        local pid=$!
        echo $pid > $pid_file
        wait $pid
        echo "$service_name has stopped. Restarting..."
    done
}

# Start the validation_api
echo "Starting validation_api..."
./../$VENV_NAME/bin/python3 validation_api.py >> "log/validation_api.log" 2>&1 &
echo $! > log/validation_api.pid

# Start the eval_score_api in a loop to restart after each request
restart_service "eval_score_api" "eval_score_api.py" "log/eval_score_api.log" "log/eval_score_api.pid" "log/eval_score_api_loop.pid" &

# Start the vibe_score_api in a loop to restart after each request
restart_service "vibe_score_api" "vibe_score_api.py" "log/vibe_score_api.log" "log/vibe_score_api.pid" "log/vibe_score_api_loop.pid" &

echo "All APIs are running in the background."