"""Command-line argument parsing utilities"""

import argparse


def parse_args():
    """Parse command-line arguments for the manipulation simulation"""
    parser = argparse.ArgumentParser(
        description="Run robot manipulation simulation with IsaacSim"
    )
    
    parser.add_argument(
        "--control_panel",
        action="store_true",
        help="Enable UI control panel for robot control (sliders for end effector and gripper)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run IsaacSim in headless mode (no rendering, for faster execution)"
    )
    parser.add_argument(
        "--camera_viewer",
        action="store_true",
        help="Enable camera viewer UI to display robot camera feeds"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="franka_robotiq",
        help="Path to the robot config"
    )
    
    return parser.parse_args()
