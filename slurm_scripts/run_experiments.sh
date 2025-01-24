#!/bin/bash

CMD_TO_RUN=sbatch                                       # Command to be run
CONFIG_FILE="slurm_scripts/experiments_config.txt"  # Path to your configuration file
WAIT_TIME=2                                             # Time to wait in seconds

# Read the config file line by line
while IFS= read -r line; do
    # Skip empty lines or lines starting with a #
    [[ -z "$line" || "$line" =~ ^# ]] && continue

    # Extract parameters
    IFS=',' read -ra PARAMS <<< "$line" # Adjust IFS for comma-separated configs

    # Run the script with the parameters
    echo "Running: $CMD_TO_RUN -J ${PARAMS[*]}"
    $CMD_TO_RUN -J "${PARAMS[@]}"
    
    # Wait before the next execution
    sleep "$WAIT_TIME"
done < "$CONFIG_FILE"

echo "All scripts started!"
