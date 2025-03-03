#!/bin/bash

remove_files() {
    local dir=$1
    if [ -d "$dir" ]; then
        echo "Removing files in $dir"
        rm -rf "$dir"/*
        if [ $? -eq 0 ]; then
            echo "Successfully removed files in $dir"
        else
            echo "Error removing files in $dir"
        fi
    fi
}

# Remove files in ./logs
remove_files "./api_logs"

# Remove files in /tmp/valapi_event_logs
remove_files "/tmp/valapi_event_logs"
remove_files "/tmp/validation_api"
remove_files "/tmp/validation_api_models/"

echo "Cleanup completed"