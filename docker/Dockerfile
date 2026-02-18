
FROM nvcr.io/nvidia/isaac-sim:5.1.0

# Switch to root to install packages
USER root

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    curl \
    git \
    gnupg2 \
    lsb-release \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update package lists after adding ROS 2 repository
RUN apt-get update && apt upgrade -y

# # Install ROS 2 Jazzy Desktop and development tools
RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-base \
    ros-jazzy-ament-cmake \
    ros-jazzy-rosidl-default-generators \
    ros-jazzy-rosidl-default-runtime \
    ros-jazzy-geometry-msgs \
    ros-jazzy-std-msgs \
    ros-jazzy-builtin-interfaces \
    ros-jazzy-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    locate \
    python3-colcon-ros \
    python3-rosdep \
    python3-pip \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Use Isaac Python installation, ensure pip uses that installation
ENV ISAAC_PATH=/isaac-sim
ENV MAN_ISAAC_SIM_PATH=/manipulation-isaacsim
ENV CARB_APP_PATH=${ISAAC_PATH}/kit
ENV EXP_PATH=${ISAAC_PATH}/apps
ENV MAN_ISAAC_SIM_PYTHON=${MAN_ISAAC_SIM_PATH}/docker/isaac_python_docker_setup.sh

# RUN echo 'export PYTHONPATH=${ISAAC_SIM_ROOT}/kit/python/bin:${PYTHONPATH}' >> /root/.bashrc
RUN echo 'source /opt/ros/jazzy/setup.bash' >> /root/.bashrc

# Copy over manipulation-isaacsim to pull in python script
COPY . /manipulation-isaacsim
RUN ln -sf ${MAN_ISAAC_SIM_PYTHON} /usr/bin/python3
RUN ln -sf ${MAN_ISAAC_SIM_PYTHON} /usr/bin/python
RUN pip install -U \
    pip \
    setuptools \
    netifaces \
    wheel \
    opencv-python --no-deps
RUN cd /manipulation-isaacsim && pip install -e .

# Patch shebangs (TODO: dehack)
RUN find /isaac-sim/kit/python/bin -type f -exec grep -l "isaac-sim/kit/python/bin/python3" {} \; | xargs -r sed -i '1s|#!/isaac-sim/kit/python/bin/python3|#!/usr/bin/env python3|'
ENV PATH=$PATH:${CARB_APP_PATH}/python/bin

RUN pip install -U colcon-core colcon-common-extensions && \
    pip install empy==3.3.4 && \
    pip install lark

RUN pip3 install --upgrade pip && \
    pip3 install uv && \
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install numpy opencv-python pyzmq pytest chex

# Clone and install OpenPI repository
RUN git clone https://github.com/Physical-Intelligence/openpi.git /opt/openpi && \
    cd /opt/openpi && \
    git submodule update --init --recursive && \
    uv pip install -e . --system

# Build the ROS workspace
WORKDIR /manipulation-isaacsim/src/manipulation_isaacsim/ros_ws

RUN /bin/bash -c ". /opt/ros/jazzy/setup.bash && colcon build"

# Add the workspace setup to bashrc so it's sourced in new shells
RUN echo 'source /manipulation-isaacsim/src/manipulation_isaacsim/ros_ws/install/setup.bash' >> /root/.bashrc

WORKDIR /
# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
