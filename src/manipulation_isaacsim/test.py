import numpy as np
import cv2
from isaacsim import SimulationApp

# 1. Start App
simulation_app = SimulationApp({"headless": False})

# 2. Post-App Imports
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from manipulation_isaacsim.robots import FrankaRobotiq
from manipulation_isaacsim.payload.payload_manager import PayloadManager

def main():
    assets_root_path = get_assets_root_path()
    world = World(stage_units_in_meters=1.0)
    
    # Setup Scene
    usd_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
    add_reference_to_stage(usd_path=usd_path, prim_path="/World")

    # 1. Create Robot
    robot = FrankaRobotiq(robot_path="/World/robot", create_robot=True)
    
    # 2. CRITICAL: Step twice to ensure the Robot's links are parsed into the USD Stage
    simulation_app.update()
    simulation_app.update()

    # 3. Create Camera (but DO NOT initialize yet)
    payload_manager = PayloadManager(robot_root_path="/World/robot/robot")
    wrist_cam = payload_manager.create_camera(
        name="wrist_camera",
        parent_link="panda_link7",
        position=np.array([0.0, 0.0, 0.05]),
        orientation=np.array([1, 0, 0, 0])
    )

    # 4. THE HANDSHAKE SEQUENCE
    # We must step the renderer so the 'RenderProduct' is allocated a GPU ID
    print("Pre-warming GPU buffers...")
    for _ in range(60):
        simulation_app.update()
    
    print("Resetting world to stabilize physics...")
    world.reset()
    
    # Update again after reset
    for _ in range(30):
        simulation_app.update()

    # 5. NOW attempt initialization
    print("Finalizing Camera Bridge...")
    try:
        # If PayloadManager's initialize_cameras calls camera.initialize(), call it here
        wrist_cam.initialize() 
        print("SUCCESS: Camera is live.")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    # 6. Simple Test Loop
    while simulation_app.is_running():
        world.step(render=True)
        
        rgba = wrist_cam.get_rgba()
        if rgba is not None and rgba.size > 0:
            # Show the frame to prove it's working
            frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            cv2.imshow("Wrist Cam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Warning: Camera buffer still empty...")

    cv2.destroyAllWindows()
    simulation_app.close()

if __name__ == "__main__":
    main()