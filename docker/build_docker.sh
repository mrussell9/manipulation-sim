#!/bin/bash

# Run this from any directory
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$script_dir/.."

docker build -t isaac-sim-ros2:jazzy -f docker/Dockerfile .
