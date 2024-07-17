#!/bin/bash

kill_process_group() {
    local pid=$1
    local pgrp=$(ps -o pgid= $pid | grep -o '[0-9]*')
    echo "Stopping process group: $pgrp"
    kill -- -$pgrp
    rm -f "log/$2.pid"
}
# Function to kill child processes of a given parent PID
kill_child_processes() {
    local PARENT_PID=$1

    # Check if PARENT_PID is provided
    if [ -z "$PARENT_PID" ]; then
        echo "Usage: kill_child_processes <parent_pid>"
        return 1
    fi

    # Find child processes of the given parent PID
    local CHILD_PIDS=$(ps --ppid $PARENT_PID -o pid=)

    if [ -z "$CHILD_PIDS" ]; then
        echo "No child processes found for PID $PARENT_PID"
        return 0
    fi

    echo "Killing child processes of PID $PARENT_PID: $CHILD_PIDS"
    for PID in $CHILD_PIDS; do
        kill -TERM $PID
        echo "Killed process $PID"
    done
}

echo "Stopping validation_api..."

kill_process_group $(cat api_logs/validation_api.pid)

kill $(cat api_logs/validation_api.pid)
rm -f api_logs/validation_api.pid


echo "All APIs and their subprocesses have been stopped and PID files removed."
