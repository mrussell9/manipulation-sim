#!/bin/bash

set -e

error_exit()
{
    echo "There was an error running python"
    exit 1
}

# Run from within docker container, assume ISAAC_PATH has been set
# Source Isaac Sim's Python environment (suppress Humble error to stderr)
source ${ISAAC_PATH}/setup_python_env.sh 2>/dev/null || source ${ISAAC_PATH}/setup_python_env.sh

python_exe="${ISAAC_PATH}/kit/python/bin/python3"
$python_exe "$@" $args || error_exit
