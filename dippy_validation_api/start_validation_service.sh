#!/bin/bash

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


# Define ports for services with offsets
VALIDATION_API_PORT=8000

# Check if ports are already in use before starting services
exit_if_port_in_use "validation_api" $VALIDATION_API_PORT


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

# check if .env file exists
if [ ! -f ../.env ]; then
    echo "Error: .env file does not exist."
    exit 1
fi

# Export environment variables from .env file
export $(grep -v '^#' ../.env | xargs)

# check if ADMIN_KEY is loaded
echo "ADMIN_KEY: $ADMIN_KEY"
echo "DIPPY_KEY: $DIPPY_KEY"
EVAL_SCORE_API_PORT=1111
VIBE_SCORE_API_PORT=1111
# Start the validation_api
echo "Starting validation_api..."
./../$VENV_NAME/bin/python3 validation_api.py --main-api-port $VALIDATION_API_PORT >> "logs/validation_api.log" 2>&1 &
echo $! > logs/validation_api.pid

echo "All APIs are running in the background"
