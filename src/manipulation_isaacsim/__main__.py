import os
# Prevent JAX from taking 90% of VRAM immediately
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["JAX_PLATFORMS"] = "cpu,cuda"

import numpy as np
import time
from isaacsim import SimulationApp

# 1. Start SimulationApp
from manipulation_isaacsim.cli_args import parse_args
# from manipulation_isaacsim.payload.camera import Camera

args_cli = parse_args()
simulation_app = SimulationApp({"headless": args_cli.headless})

# 2. Imports after SimulationApp
import omni
from isaacsim.core.api import World
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.experimental.prims import XformPrim
from manipulation_isaacsim.robots import FrankaRobotiq, FrankaPanda 

# from manipulation_isaacsim.payload.payload_manager import PayloadManager
from manipulation_isaacsim.agent.pi0_agent import Pi0Agent
from manipulation_isaacsim.utils.camera_viewer import CameraViewer
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import matplotlib.pyplot as plt


class SimulationRunner:
    def __init__(
            self, 
            physics_dt, 
            render_dt, 
            robot
        ):
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
        self.robot = robot
        self.agent = None
        self.action_buffer = []
        self.last_camera_warning_time = 0
        self.camera_viewer = None  # Will be set after creation

        # Use existing camera on the robot (adjust the prim_path to match your robot's camera)
        # Example paths: "/World/robot/camera" or "/World/robot/robot/Camera"
        self.wrist_camera = Camera(
            prim_path="/World/robot/robot/panda_link7/wrist_camera/RSD455/Camera_OmniVision_OV9782_Color",  # Replace with actual camera path
            frequency=20,
            resolution=(224, 224),
            )
        self.base_camera = Camera(
            prim_path="/World/robot/robot/panda_link0/base_camera/RSD455/Camera_OmniVision_OV9782_Color",  # Replace with actual camera path
            frequency=20,
            resolution=(224, 224),
        )
        self.wrist_camera.initialize()
        self.base_camera.initialize()

    def run(self) -> None:
        # pi0 inference frequency (10Hz)
        inference_skip = int(1.0 / (self._world.get_physics_dt() * 10))
        # Robot control frequency (matching your physics_dt or a multiple)
        control_skip = 1 
        step_counter = 0

        print("Entering Main Loop...")
        while simulation_app.is_running():
            # Only run if agent is loaded
            wrist_cam_data = self.wrist_camera.get_rgba()
            base_cam_data = self.base_camera.get_rgba()

            if wrist_cam_data.shape[0] == 0 or base_cam_data.shape[0] == 0:
                current_time = time.time()
                if current_time - self.last_camera_warning_time >= 2.0:
                    print("Camera data not ready, skipping step...")
                    self.last_camera_warning_time = current_time
                self._world.step(render=True)
                continue
            
            # Update camera viewer if enabled
            if self.camera_viewer:
                self.camera_viewer.update(base_image=base_cam_data, wrist_image=wrist_cam_data)
            
            if self.agent:
                # 1. Inference Step
                if step_counter % inference_skip == 0:
                        obs = self.agent.get_observation(base_cam_data[:,:,:3], wrist_cam_data[:,:,:3], self.robot, "pick up the block")
                        
                        new_actions = self.agent.compute_step(obs)
                        if new_actions is not None:
                            self.action_buffer = list(new_actions)

                # 2. Control Step
                if self.action_buffer and step_counter % control_skip == 0:
                    target_pose = self.action_buffer.pop(0)
                    # Assuming target_pose is [x, y, z, qx, qy, qz, qw]
                    # orientation = target_pose[3:7]
                    # IsaacSim expects quaternions to be scalar first [qw, qx, qy, qz]
                    # orientation = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])
                    self.robot.apply_joint_action(joint_positions=target_pose[:7], gripper_state=target_pose[7])  # Adjust slicing based on your action format

            self._world.step(render=True)
            step_counter += 1

def main():
    # Setup Scene
    assets_root_path = get_assets_root_path()
    usd_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    add_reference_to_stage(usd_path=usd_path, prim_path="/World")

    # Add a cube to the scene
    cube_usd = assets_root_path + "/Isaac/Props/Blocks/red_block.usd"
    add_reference_to_stage(usd_path=cube_usd, prim_path="/World/cube")

    table_usd = assets_root_path + "/Isaac/Props/PackingTable/packing_table.usd"
    add_reference_to_stage(usd_path=table_usd, prim_path="/World/Table")

    # Set the position of the cube
    cube = XformPrim("/World/cube")
    cube.reset_xform_op_properties()
    cube.set_world_poses(positions=np.array([0.35, 0.0, 1.2]))

    table = XformPrim("/World/Table")
    table.reset_xform_op_properties()
    table.set_world_poses(positions=np.array([0.3, 0.0, 0.0]))

    # Setup Robot
    if args_cli.robot == "franka_robotiq":
        robot = FrankaRobotiq(robot_path="/World/robot", create_robot=True)
    elif args_cli.robot == "franka_panda":
        robot = FrankaPanda(robot_path="/World/robot", create_robot=True)

    runner = SimulationRunner(0.005, 0.02, robot)
    
    # Create camera viewer if enabled
    if args_cli.camera_viewer:
        camera_viewer = CameraViewer()
        runner.camera_viewer = camera_viewer

    print("Stage 4: Loading π₀ Agent (JAX/GPU Staging)...")
    checkpoint_path = os.path.join(os.path.dirname(__file__), "agent", "checkpoints", "pi0_base")
    agent = Pi0Agent(checkpoint_path=checkpoint_path)
    runner.agent = agent

    print("Startup complete. Running simulation...")
    runner.run()
    simulation_app.close()

if __name__ == "__main__":
    main()