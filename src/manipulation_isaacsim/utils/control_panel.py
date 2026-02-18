"""UI controls for robot manipulation"""

import time
import numpy as np
import omni.ui as ui
from PIL import Image
import io


class ControlPanel:
    """Control panel UI for robot manipulation with sliders and preset controls."""
    
    def __init__(self, runner):
        """Initialize the control panel.
        
        Args:
            runner: simulation_runner instance with robot and state
        """
        self.runner = runner
        self.window = None
        
        # Slider references
        self._x_slider = None
        self._y_slider = None
        self._z_slider = None
        self._x_value_label = None
        self._y_value_label = None
        self._z_value_label = None
        self._gripper_slider = None
        self._gripper_value_label = None
        
        # Camera image references
        self._camera_image_widget = None
        self._selected_camera = "base"  # "base" or "wrist"
        self._camera_dropdown = None
    
    def get_robot_base_position(self) -> np.ndarray:
        """Get robot base position in world frame"""
        base_position, _ = self.runner.robot.get_world_poses()
        return base_position.numpy()[0]
    
    def update_ee_position(self) -> None:
        """Update end effector position with current target (robot-relative)"""
        # Convert robot-relative position to world frame
        robot_base_pos = self.get_robot_base_position()
        world_position = self.runner.target_ee_position + robot_base_pos
        
        _, _, orientation = self.runner.robot.get_current_state()
        self.runner.robot.set_end_effector_pose(
            position=self.runner.target_ee_position,
            orientation=orientation
        )
    
    def set_ee_position(self, x: float, y: float, z: float) -> None:
        """Set end effector to specific position and update UI"""
        self.runner.target_ee_position = np.array([x, y, z])
        if self._x_slider is not None:
            self._x_slider.model.set_value(x)
            self._y_slider.model.set_value(y)
            self._z_slider.model.set_value(z)
            self._x_value_label.text = f"{x:.3f}"
            self._y_value_label.text = f"{y:.3f}"
            self._z_value_label.text = f"{z:.3f}"
        self.update_ee_position()
        self.runner.last_ee_command_time = time.time()
    
    def set_gripper(self, value: float) -> None:
        """Set gripper to specific value and update UI"""
        self.runner.gripper_value = value
        if self._gripper_slider is not None:
            self._gripper_slider.model.set_value(value)
            self._gripper_value_label.text = f"{value:.4f}"
        # Convert gripper value to proper shape: [finger1, finger2]
        gripper_positions = np.array([value * 0.04])
        self.runner.robot.set_gripper_position(gripper_positions)
        self.runner.last_gripper_command_time = time.time()
    
    def update_camera_display(self, base_image: np.ndarray = None, wrist_image: np.ndarray = None) -> None:
        """Update the camera image display in the UI
        
        Args:
            base_image: RGBA image from base camera (H, W, 4)
            wrist_image: RGBA image from wrist camera (H, W, 4)
        """
        if self._camera_image_widget is None:
            return
        
        try:
            # Select which image to display based on selected camera
            if self._selected_camera == "base" and base_image is not None:
                camera_data = base_image
            elif self._selected_camera == "wrist" and wrist_image is not None:
                camera_data = wrist_image
            else:
                return
            
            # Convert RGBA to RGB and scale to uint8
            rgb_data = (camera_data[:, :, :3] * 255).astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(rgb_data)
            
            # Convert PIL Image to bytes
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                image_bytes = output.getvalue()
            
            # Update the image widget
            self._camera_image_widget.source_url = f"data:image/png;base64,{__import__('base64').b64encode(image_bytes).decode()}"
        except Exception as e:
            print(f"Error updating camera display: {e}")
    
    def build(self) -> ui.Window:
        """Build UI window with robot controls
        
        Returns:
            ui.Window: The control window
        """
        self.window = ui.Window(
            "Robot Control", 
            width=400, 
            height=500
        )
        with self.window.frame:
            with ui.VStack(spacing=10):
                # End Effector Position Control
                ui.Label("End Effector Position (Robot Frame)", height=20, style={"font_size": 16})
                
                # X Position
                with ui.HStack(spacing=5):
                    ui.Label("X:", width=30)
                    self._x_slider = ui.FloatSlider(min=-0.8, max=0.8, height=20)
                    self._x_slider.model.set_value(self.runner.target_ee_position[0])
                    self._x_value_label = ui.Label(f"{self.runner.target_ee_position[0]:.3f}", width=60)
                
                # Y Position
                with ui.HStack(spacing=5):
                    ui.Label("Y:", width=30)
                    self._y_slider = ui.FloatSlider(min=-0.8, max=0.8, height=20)
                    self._y_slider.model.set_value(self.runner.target_ee_position[1])
                    self._y_value_label = ui.Label(f"{self.runner.target_ee_position[1]:.3f}", width=60)
                
                # Z Position
                with ui.HStack(spacing=5):
                    ui.Label("Z:", width=30)
                    self._z_slider = ui.FloatSlider(min=0.0, max=1.0, height=20)
                    self._z_slider.model.set_value(self.runner.target_ee_position[2])
                    self._z_value_label = ui.Label(f"{self.runner.target_ee_position[2]:.3f}", width=60)
                
                # Set up position slider callbacks
                def on_x_changed(model):
                    current_time = time.time()
                    self.runner.target_ee_position[0] = model.get_value_as_float()
                    self._x_value_label.text = f"{self.runner.target_ee_position[0]:.3f}"
                    if current_time - self.runner.last_ee_command_time >= self.runner.ee_command_interval:
                        self.update_ee_position()
                        self.runner.last_ee_command_time = current_time
                
                def on_y_changed(model):
                    current_time = time.time()
                    self.runner.target_ee_position[1] = model.get_value_as_float()
                    self._y_value_label.text = f"{self.runner.target_ee_position[1]:.3f}"
                    if current_time - self.runner.last_ee_command_time >= self.runner.ee_command_interval:
                        self.update_ee_position()
                        self.runner.last_ee_command_time = current_time
                
                def on_z_changed(model):
                    current_time = time.time()
                    self.runner.target_ee_position[2] = model.get_value_as_float()
                    self._z_value_label.text = f"{self.runner.target_ee_position[2]:.3f}"
                    if current_time - self.runner.last_ee_command_time >= self.runner.ee_command_interval:
                        self.update_ee_position()
                        self.runner.last_ee_command_time = current_time
                
                self._x_slider.model.add_value_changed_fn(on_x_changed)
                self._y_slider.model.add_value_changed_fn(on_y_changed)
                self._z_slider.model.add_value_changed_fn(on_z_changed)
                
                # Preset poses
                ui.Label("Preset Poses", height=20)
                with ui.HStack(spacing=5):
                    if ui.Button("Home", height=30, clicked_fn=self.runner.robot.reset_to_default_pose):
                        pass
                    if ui.Button("Front", height=30, clicked_fn=lambda: self.set_ee_position(0.5, 0.0, 0.3)):
                        pass
                with ui.HStack(spacing=5):
                    if ui.Button("Left", height=30, clicked_fn=lambda: self.set_ee_position(0.3, 0.3, 0.3)):
                        pass
                    if ui.Button("Right", height=30, clicked_fn=lambda: self.set_ee_position(0.3, -0.3, 0.3)):
                        pass
                
                ui.Spacer(height=10)
                ui.Separator()
                ui.Spacer(height=10)
                
                # Gripper Position Control
                ui.Label("Gripper Position", height=20, style={"font_size": 16})
                
                with ui.HStack(spacing=5):
                    ui.Label("Position:", width=80)
                    self._gripper_slider = ui.FloatSlider(
                        min=0.0, 
                        max=1.0, 
                        height=20
                    )
                    self._gripper_slider.model.set_value(self.runner.gripper_value)
                    self._gripper_value_label = ui.Label(f"{self.runner.gripper_value:.4f}", width=60)
                
                def on_gripper_changed(model):
                    current_time = time.time()
                    self.runner.gripper_value = model.get_value_as_float()
                    self._gripper_value_label.text = f"{self.runner.gripper_value:.4f}"
                    # Debounce: only send command if enough time has passed
                    if current_time - self.runner.last_gripper_command_time >= self.runner.gripper_command_interval:
                        gripper_positions = np.array([self.runner.gripper_value])
                        self.runner.robot.set_gripper_position(gripper_positions)
                        self.runner.last_gripper_command_time = current_time
                
                self._gripper_slider.model.add_value_changed_fn(on_gripper_changed)
                
                with ui.HStack(spacing=5):
                    if ui.Button("Open", height=30, clicked_fn=lambda: self.runner.robot.open_gripper()):
                        pass
                    if ui.Button("Close", height=30, clicked_fn=lambda: self.runner.robot.close_gripper()):
                        pass
                
                ui.Spacer(height=10)
                ui.Separator()
                ui.Spacer(height=10)
                
                # Camera Display
                ui.Label("Camera Feed", height=20, style={"font_size": 16})
                
                with ui.HStack(spacing=5):
                    ui.Label("Select Camera:", width=100)
                    self._camera_dropdown = ui.ComboBox(
                        0,
                        "Base Camera",
                        "Wrist Camera",
                        height=25,
                        width=150
                    )
                    
                    def on_camera_changed(model, value):
                        camera_names = ["base", "wrist"]
                        self._selected_camera = camera_names[value]
                    
                    self._camera_dropdown.model.add_item_changed_fn(on_camera_changed)
                
                # Camera image display (512x512 pixels)
                self._camera_image_widget = ui.Image(width=300, height=300)
        
        return self.window
