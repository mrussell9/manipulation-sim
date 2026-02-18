# manipulation-sim

Robot manipulation simulation environment using NVIDIA Isaac Sim with Pi0 policy integration.

## Overview

This project provides a simulation environment for robot manipulation tasks using Isaac Sim. It supports multiple robot configurations (Franka Panda, Franka with Robotiq gripper) and integrates with the Pi0 vision-language-action model for policy-driven control.

## Features

- **Isaac Sim Integration**: Leverages NVIDIA Isaac Sim for physics simulation
- **ROS2 Support**: Includes ROS2 workspace for robotic middleware integration
- **Pi0 Policy**: Pre-trained vision-language-action models with support for multiple robot types
- **Camera System**: Wrist-mounted camera with real-time viewer
- **Control Options**: UI control panel for manual manipulation or autonomous policy execution
- **Docker Support**: Containerized environment for reproducible deployments

## Project Structure

```
manipulation-sim/
├── src/manipulation_isaacsim/
│   ├── agent/           # Pi0 policy agents and model checkpoints
│   ├── robots/          # Robot definitions (Franka, etc.) and USD files
│   ├── payload/         # Camera and payload management
│   ├── utils/           # Camera viewer and control panel utilities
│   ├── ros_ws/          # ROS2 workspace
│   └── config/          # Configuration files
├── docker/              # Docker build files and entrypoint scripts
└── pyproject.toml       # Python project configuration
```

## Requirements

- Python >= 3.10
- NVIDIA Isaac Sim 4.5.0
- PyTorch 2.5.1
- CUDA-capable GPU

## Installation

### 1. Install dependencies
```bash
pip install -e .
```

### 2. Install OpenPI package
```bash
pip install openpi
```

### 3. Download Pi0 model checkpoints

The Pi0 models are hosted by Physical Intelligence and automatically downloaded when first used. They will be cached locally at `~/.cache/openpi/` by default.

**Option A: Automatic Download (Recommended)**

The checkpoints will download automatically when you first run the simulation. The `pi0_base` checkpoint (~3GB) will be fetched from `gs://openpi-assets/checkpoints/pi0_base` and cached.

**Option B: Manual Download and Setup**

If you prefer to manage the checkpoint location manually:

```bash
# Install gsutil if you don't have it
pip install gsutil

# Download the checkpoint to a custom location
mkdir -p src/manipulation_isaacsim/agent/checkpoints
gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_base \
    src/manipulation_isaacsim/agent/checkpoints/

# Download the PaliGemma tokenizer 
mkdir -p src/manipulation_isaacsim/agent/tokenizers/
curl -L -o src/manipulation_isaacsim/agent/tokenizers/paligemma_tokenizer.model \
https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
```

**Note**: 
- The `agent/` folder is in `.gitignore` to prevent committing large model files
- Available checkpoints: `pi0_base`, `pi0_droid`, `pi0_fast_base`, `pi05_base`, `pi05_droid`
- See [OpenPI documentation](https://github.com/Physical-Intelligence/openpi) for more checkpoint options

### 4. Ensure Isaac Sim is installed

Make sure NVIDIA Isaac Sim 5.1.0 is properly installed and configured on your system.

## Usage

Run the simulation with default settings:
```bash
run-sim
```

### Command-Line Options

- `--headless`: Run without GUI rendering (faster execution)
- `--control_panel`: Enable manual control UI with sliders
- `--camera_viewer`: Display camera feed viewer
- `--robot <name>`: Specify robot configuration (default: franka_robotiq)

### Examples

Run with camera viewer and control panel:
```bash
run-sim --camera_viewer --control_panel
```

Run headless for faster execution:
```bash
run-sim --headless
```

## Docker

Build and run using Docker:
```bash
cd docker
./build_docker.sh
cd ..
./run_docker.sh
```

## Supported Robots

- Franka Panda
- Franka with Robotiq Gripper
- Custom robot configurations via USD files

## License

[Add your license information here]
