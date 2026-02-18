"""Camera viewer UI for displaying robot camera feeds"""

import numpy as np
import omni.ui as ui


class CameraImageProvider(ui.ByteImageProvider):
    """Custom image provider for camera feed"""
    
    def __init__(self):
        super().__init__()
        self._image_data = None
        
    def update_image(self, rgba_array: np.ndarray):
        """Update the image data
        
        Args:
            rgba_array: RGBA image array (H, W, 4) in 0-255 range
        """
        height, width = rgba_array.shape[:2]
        
        # Image is already in BGRA format, just flatten and convert to list
        self._image_data = np.ascontiguousarray(rgba_array).flatten().tolist()
        self.set_bytes_data(self._image_data, [width, height])


class CameraViewer:
    """Simple camera viewer UI for displaying robot camera feeds."""
    
    def __init__(self):
        """Initialize the camera viewer."""
        self.window = None
        self._camera_image = None
        self._selected_camera = "base"  # "base" or "wrist"
        self._camera_dropdown = None
        self._image_provider = CameraImageProvider()
        self.build()
        
    def build(self):
        """Build the camera viewer window"""
        self.window = ui.Window("Camera Viewer", width=400, height=500)
        with self.window.frame:
            with ui.VStack(spacing=10):
                # Camera selection dropdown
                with ui.HStack(spacing=5, height=30):
                    ui.Label("Camera:", width=80)
                    self._camera_dropdown = ui.ComboBox(
                        0,
                        "Base Camera",
                        "Wrist Camera",
                        height=25,
                        width=200
                    )
                    
                    def on_camera_changed(model, item):
                        camera_names = ["base", "wrist"]
                        index = model.get_item_value_model().as_int
                        self._selected_camera = camera_names[index]
                    
                    self._camera_dropdown.model.add_item_changed_fn(on_camera_changed)
                
                # Camera image display
                self._camera_image = ui.ImageWithProvider(
                    self._image_provider,
                    width=360,
                    height=360
                )
    
    def update(self, base_image: np.ndarray = None, wrist_image: np.ndarray = None) -> None:
        """Update the camera display
        
        Args:
            base_image: RGBA image from base camera (H, W, 4) in 0-1 range
            wrist_image: RGBA image from wrist camera (H, W, 4) in 0-1 range
        """
        try:
            # Select which image to display based on selected camera
            if self._selected_camera == "base" and base_image is not None:
                camera_data = base_image
            elif self._selected_camera == "wrist" and wrist_image is not None:
                camera_data = wrist_image
            else:
                return
                        
            # Update the image provider
            self._image_provider.update_image(camera_data)
            
        except Exception as e:
            print(f"Error updating camera display: {e}")
    
    def set_camera(self, camera_name: str) -> None:
        """Switch between cameras
        
        Args:
            camera_name: Either "base" or "wrist"
        """
        if camera_name in ["base", "wrist"]:
            self._selected_camera = camera_name
    
    def close(self) -> None:
        """Close the camera viewer window"""
        if self.window:
            self.window.visible = False
