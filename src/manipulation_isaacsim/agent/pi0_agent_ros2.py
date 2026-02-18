#!/usr/bin/env python3
import jax
import torch
import dataclasses
import numpy as np
from openpi.policies import policy_config
from openpi.training import config as train_config
from openpi.models.pi0 import pi0_config

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge


class Pi0AgentNode(Node):
    def __init__(self, checkpoint_path, prompt="pick up the object"):
        super().__init__('pi0_agent')
        
        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Store latest data
        self.latest_image = None
        self.latest_joint_state = None
        self.prompt = prompt
        
        # Create subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Create action publisher
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            '/pi0/action',
            10
        )
        
        # Initialize the policy model
        self.get_logger().info("Restoring π₀ parameters (CPU staging)...")
        with jax.default_device(jax.devices("cpu")[0]):
            base_train_config = train_config.get_config("pi0_droid")
            optimized_model = pi0_config.Pi0Config(
                action_dim=32, action_horizon=4, max_token_len=32,
                dtype='bfloat16', paligemma_variant='gemma_2b',
                action_expert_variant='gemma_300m', pi05=False
            )
            model_config = dataclasses.replace(base_train_config, model=optimized_model)
            self.policy = policy_config.create_trained_policy(model_config, checkpoint_path)
        
        gpu = jax.devices("gpu")[0]
        self.policy._model = jax.tree.map(
            lambda x: jax.device_put(x, gpu) if hasattr(x, "dtype") else x, 
            self.policy._model
        )
        self.get_logger().info(f"π₀ Model initialized on {gpu}")
        
        # Create timer to run inference at regular intervals
        self.timer = self.create_timer(0.1, self.inference_callback)  # 10 Hz
        
    def image_callback(self, msg):
        """Callback for image messages"""
        try:
            # Convert ROS Image to OpenCV format (RGB)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')
    
    def joint_callback(self, msg):
        """Callback for joint state messages"""
        # Store the first 7 joint positions (arm joints)
        if len(msg.position) >= 7:
            self.latest_joint_state = np.array(msg.position[:7])
        else:
            self.get_logger().warn(f'Received joint state with only {len(msg.position)} positions')
    
    def inference_callback(self):
        """Run inference and publish actions"""
        if self.latest_image is None or self.latest_joint_state is None:
            self.get_logger().warn('Waiting for image and joint data...', throttle_duration_sec=5.0)
            return
        
        # Prepare observation packet
        obs_packet = {
            "image": self.latest_image,
            "state": self.latest_joint_state,
            "prompt": self.prompt
        }
        
        # Compute action
        try:
            with torch.no_grad():
                result = self.policy.infer(obs_packet)
                actions = result["actions"]
            
            # Publish action
            action_msg = Float32MultiArray()
            action_msg.data = actions.flatten().tolist() if hasattr(actions, 'flatten') else list(actions)
            self.action_pub.publish(action_msg)
            
            self.get_logger().info(f'Published action with {len(action_msg.data)} elements', throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f'Error computing action: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    # Path to checkpoint - adjust as needed
    checkpoint_path = "/app/manipulation-isaacsim/src/manipulation_isaacsim/agent/checkpoints/pi0_base"
    prompt = "pick up the object"
    
    node = Pi0AgentNode(checkpoint_path, prompt)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
