#!/bin/bash



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
kill $(cat log/validation_api.pid)
rm -f log/validation_api.pid

echo "All APIs and their subprocesses for batch $BATCH_NUM have been stopped and PID files removed."
