#!/bin/bash

# Sym-link _after_ mounting the large scenes directory, avoiding directly adding
#   the large scenes directory to the host location / docker context.
# if [ -d "/scenes" ]; then
#   ln -s /scenes /manipulation-isaacsim/src/manipulation_isaacsim/scenes/usd/scenes/
# fi

# Source ROS 2 Humble setup (Isaac Sim may try to source Humble, ignore errors)
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

# Source the ROS workspace if it exists
if [ -f /manipulation-isaacsim/src/manipulation_isaacsim/ros_ws/install/setup.bash ]; then
    source /manipulation-isaacsim/src/manipulation_isaacsim/ros_ws/install/setup.bash
fi

# Execute the command
exec "$@"
