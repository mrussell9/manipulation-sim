import numpy as np
from typing import List, Optional, Dict
from isaacsim.core.experimental.utils.impl.transform import euler_angles_to_quaternion
from isaacsim.storage.native import get_assets_root_path


# Camera catalog: maps camera type names to their USD paths
def _get_camera_catalog() -> Dict[str, str]:
    """Returns a catalog of available camera types and their USD paths."""
    assets_root = get_assets_root_path()
    return {
        "realsense_d455": f"{assets_root}/Isaac/Sensors/Intel/RealSense/rsd455.usd",
        "Zed_X": f"{assets_root}/Isaac/Sensors/Stereolabs/Zed_X.usdc",
        "Zed_X_Mini": f"{assets_root}/Isaac/Sensors/Stereolabs/Zed_X_Mini.usdc",
    }

class Camera:
    def __init__(self, name, usd_path, parent_link, xyz, quat):
        self.name = name
        self.usd_path = usd_path
        self.parent_link = parent_link
        self.xyz = np.array(xyz)
        
        # Convert and immediately cast to numpy to avoid Warp errors later
        self.quat = np.array(quat)
    
    @classmethod
    def get_catalog(cls) -> Dict[str, str]:
        """Get the catalog of available camera types."""
        cls._catalog = _get_camera_catalog()
        return cls._catalog
    
    @classmethod
    def from_type(
        cls, 
        camera_type: str, 
        name: str, 
        parent_link: str, 
        xyz: List[float] = [0, 0, 0], 
        quat: List[float] = [1, 0, 0, 0] # w, x, y, z
    ) -> 'Camera':
        """Create a camera from a registered camera type.
        
        Args:
            camera_type: Type of camera from the catalog (e.g., "realsense_d455")
            name: Unique name for this camera instance
            parent_link: Name of the robot link to attach to
            xyz: Position offset [x, y, z] in meters
            quat: Orientation quaternion [w, x, y, z]
            
        Returns:
            Camera instance
            
        Raises:
            ValueError: If camera_type is not in the catalog
            
        Example:
            >>> cam = Camera.from_type("realsense_d455", "wrist_cam", "end_effector", [0.05, 0, 0.03], [0.2706, 0.65328, -0.2706, 0.65328])
        """
        catalog = cls.get_catalog()
        if camera_type not in catalog:
            available = ", ".join(catalog.keys())
            raise ValueError(f"Unknown camera type '{camera_type}'. Available types: {available}")
        
        usd_path = catalog[camera_type]
        return cls(name, usd_path, parent_link, xyz, quat)
    