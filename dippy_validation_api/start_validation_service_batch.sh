#!/bin/bash

# Check if a batch number is provided and if it is within the valid range (0-5)
if [ -z "$1" ] || [ "$1" -lt 0 ] || [ "$1" -gt 5 ]; then
    echo "Please provide a batch number between 0 and 5."
    exit 1
fi

# Create a log directory if it doesn't exist
mkdir -p log

# Virtual environment name
VENV_NAME="model_validation_venv"

# Function to check if a port is in use
is_port_in_use() {
    local port=$1
    netstat -tuln | grep ":$port" > /dev/null
}

# Function to exit if any port is in use
exit_if_port_in_use() {
    local service_name=$1
    local port=$2
    if is_port_in_use $port; then
        echo "Port $port is already in use. Cannot start $service_name."
        exit 1
    fi
}

# Calculate the port offsets based on the batch number
BATCH_NUM=$1
PORT_OFFSET=$(( BATCH_NUM * 5 ))

# Define ports for services with offsets
VALIDATION_API_PORT=$(( 8000 + PORT_OFFSET ))
EVAL_SCORE_API_PORT=$(( 8001 + PORT_OFFSET ))
VIBE_SCORE_API_PORT=$(( 8002 + PORT_OFFSET ))

# Check if ports are already in use before starting services
exit_if_port_in_use "validation_api" $VALIDATION_API_PORT
exit_if_port_in_use "eval_score_api" $EVAL_SCORE_API_PORT
exit_if_port_in_use "vibe_score_api" $VIBE_SCORE_API_PORT

# Function to restart a service
restart_service() {
    local service_name=$1
    local service_script=$2
    local log_file=$3
    local pid_file=$4
    local loop_pid_file=$5
    local port=$6

    echo $$ > $loop_pid_file  # Store the PID of the loop process

    while true; do
        if is_port_in_use $port; then
            echo "Port $port is already in use. Cannot restart $service_name."
            exit 1
        else
            echo "Starting $service_name..."
            ./../$VENV_NAME/bin/python3 $service_script $port >> $log_file 2>&1 &
            local pid=$!
            echo $pid > $pid_file
            wait $pid
            if [ $? -ne 0 ]; then
                echo "Service $service_name exited with a non-zero status. Not restarting."
                break
            fi
            echo "$service_name has stopped. Restarting..."
        fi
    done
}

# Start the validation_api
echo "Starting validation_api..."
./../$VENV_NAME/bin/python3 validation_api_batch.py --main-api-port $VALIDATION_API_PORT --eval-score-port $EVAL_SCORE_API_PORT --vibe-score-port $VIBE_SCORE_API_PORT >> "log/validation_api_${BATCH_NUM}.log" 2>&1 &
echo $! > log/validation_api_${BATCH_NUM}.pid

# Start the eval_score_api in a loop to restart after each request
restart_service "eval_score_api" "eval_score_api.py" "log/eval_score_api_${BATCH_NUM}.log" "log/eval_score_api_${BATCH_NUM}.pid" "log/eval_score_api_loop_${BATCH_NUM}.pid" $EVAL_SCORE_API_PORT &

# Start the vibe_score_api in a loop to restart after each request
restart_service "vibe_score_api" "vibe_score_api.py" "log/vibe_score_api_${BATCH_NUM}.log" "log/vibe_score_api_${BATCH_NUM}.pid" "log/vibe_score_api_loop_${BATCH_NUM}.pid" $VIBE_SCORE_API_PORT &

echo "All APIs are running in the background for batch $BATCH_NUM."
