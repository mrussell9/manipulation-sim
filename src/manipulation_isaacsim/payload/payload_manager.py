import numpy as np
import omni.usd
import omni.kit.commands
from pxr import UsdPhysics, Usd, Sdf
from isaacsim.sensors.camera import Camera

class PayloadManager:
    def __init__(self, robot_root_path: str):
        self.robot_root = robot_root_path
        self.attached_sensors = {}

    def create_camera(self, name, parent_link, position, orientation, resolution=(256, 256), frequency=20):
            # Construct the camera path
            camera_path = f"{self.robot_root}/{parent_link}/{name}"
            
            # 1. Check if parent exists
            stage = omni.usd.get_context().get_stage()
            parent_prim = stage.GetPrimAtPath(f"{self.robot_root}/{parent_link}")
            
            if not parent_prim.IsValid():
                print(f"Warning: Parent link {parent_link} not found yet. Camera might fail.")

            # 2. Create the Camera
            camera = Camera(
                prim_path=camera_path,
                resolution=resolution,
                frequency=frequency
            )
            
            # 3. Apply local transforms manually to ensure they stick
            camera.set_local_pose(translation=position, orientation=orientation)
            
            self.attached_sensors[name] = camera
            return camera
    
    def initialize_cameras(self):
        """Initialize all attached cameras. Call this after world reset."""
        for name, camera in self.attached_sensors.items():
            camera.initialize()
            print(f"Initialized camera: {name}")