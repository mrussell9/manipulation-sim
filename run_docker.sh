#!/bin/bash
xhost +si:localuser:root

# Run this from any directory
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$script_dir"

set -e
MAN_ISAAC_ROOT=${script_dir}

usage="$(basename "$0") [-r manipulation-isaacsim root dir (default:$MAN_ISAAC_ROOT)] [-h]"

while getopts ":r:h" opt; do
    case $opt in
        r) MAN_ISAAC_ROOT=$OPTARG ;;
        h) echo "$usage"; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

shift $((OPTIND-1))

echo "Starting Isaac Sim container with root: $MAN_ISAAC_ROOT"

# CUDA_VISIBLE_DEVICES below is necessary for multi-GPU support on some systems
docker run --name isaac-sim --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --rm --network=host --ipc=host \
    -e DISPLAY=$DISPLAY \
    -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    -e CUDA_VISIBLE_DEVICES \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e "PRIVACY_CONSENT=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v "$MAN_ISAAC_ROOT":/manipulation-isaacsim:rw \
    -v ros-ws-build:/manipulation-isaacsim/src/manipulation_isaacsim/ros_ws/build:rw \
    -v ros-ws-install:/manipulation-isaacsim/src/manipulation_isaacsim/ros_ws/install:rw \
    -v ros-ws-log:/manipulation-isaacsim/src/manipulation_isaacsim/ros_ws/log:rw \
    isaac-sim-ros2:jazzy

xhost -si:localuser:root