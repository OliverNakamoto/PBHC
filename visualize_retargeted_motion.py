#!/usr/bin/env python3
"""
Visualize and save retargeted motion data from SMPL to Booster T1 robot.
This script loads the motion data, performs retargeting, and allows visualization/saving.
"""

import os
import sys
import torch
import numpy as np
import joblib
import mujoco
import mujoco.viewer
import time
import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as sRot

sys.path.append(os.getcwd())

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from humanoidverse.utils.motion_lib.motion_lib_robot_WJX import MotionLibRobotWJX


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene for visualization."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])


class RetargetedMotionVisualizer:
    def __init__(self, motion_file, robot_config_path, save_retargeted=False):
        self.motion_file = motion_file
        self.robot_config_path = robot_config_path
        self.save_retargeted = save_retargeted
        
        # Load robot configuration
        self.robot_cfg = OmegaConf.load(robot_config_path)
        
        # Initialize motion library
        motion_lib_cfg = OmegaConf.create({
            "motion_file": motion_file,
            "asset": self.robot_cfg.asset,
            "step_dt": 1/30.0
        })
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.motion_lib = MotionLibRobotWJX(motion_lib_cfg, num_envs=1, device=self.device)
        
        # Load motion data
        self.motion_data = joblib.load(motion_file)
        self.motion_keys = list(self.motion_data.keys())
        
        # Initialize humanoid FK model for retargeting
        self.humanoid_fk = Humanoid_Batch(self.robot_cfg)
        
        # Control variables
        self.paused = False
        self.time_step = 0
        self.speed = 1.0
        self.current_motion_idx = 0
        self.dt = 1/30.0
        
    def key_callback(self, keycode):
        """Handle keyboard input during visualization."""
        if chr(keycode) == " ":
            self.paused = not self.paused
            print(f"Paused: {self.paused}")
        elif chr(keycode) == "R":
            self.time_step = 0
            print("Reset to beginning")
        elif chr(keycode) == "S" and self.save_retargeted:
            self.save_current_retargeted_motion()
        elif chr(keycode) == "N":
            self.next_motion()
        elif chr(keycode) == "P":
            self.prev_motion()
        elif chr(keycode) == "L":
            self.speed *= 1.5
            print(f"Speed: {self.speed:.2f}x")
        elif chr(keycode) == "K":
            self.speed /= 1.5
            print(f"Speed: {self.speed:.2f}x")
        elif keycode == 256 or chr(keycode) == "Q":
            print("Exit")
            os._exit(0)
        elif keycode == 262:  # Right arrow
            self.time_step += self.dt
        elif keycode == 263:  # Left arrow
            self.time_step -= self.dt
            
    def next_motion(self):
        """Switch to next motion in the dataset."""
        self.current_motion_idx = (self.current_motion_idx + 1) % len(self.motion_keys)
        self.time_step = 0
        print(f"Switched to motion: {self.motion_keys[self.current_motion_idx]}")
        
    def prev_motion(self):
        """Switch to previous motion in the dataset."""
        self.current_motion_idx = (self.current_motion_idx - 1) % len(self.motion_keys)
        self.time_step = 0
        print(f"Switched to motion: {self.motion_keys[self.current_motion_idx]}")
        
    def get_retargeted_state(self, motion_key, frame_idx):
        """Get retargeted robot state for a specific frame."""
        motion = self.motion_data[motion_key]
        
        # Get pose and translation data
        pose_aa = torch.from_numpy(motion['pose_aa']).unsqueeze(0)
        root_trans = torch.from_numpy(motion['root_trans_offset']).unsqueeze(0)
        
        # Perform forward kinematics for retargeting
        fk_result = self.humanoid_fk.fk_batch(pose_aa, root_trans, return_full=True, dt=self.dt)
        
        # Extract frame data
        frame_data = {
            'root_pos': fk_result['global_translation'][0, frame_idx, 0].numpy(),
            'root_rot': fk_result['global_rotation'][0, frame_idx, 0].numpy(),
            'joint_pos': fk_result['global_translation'][0, frame_idx].numpy(),
            'joint_rot': fk_result['global_rotation'][0, frame_idx].numpy(),
            'dof_pos': fk_result['dof_pos'][0, frame_idx].numpy() if 'dof_pos' in fk_result else None,
            'dof_vel': fk_result['dof_vels'][0, frame_idx].numpy() if 'dof_vels' in fk_result else None,
        }
        
        return frame_data
    
    def save_current_retargeted_motion(self):
        """Save the current retargeted motion to a file."""
        motion_key = self.motion_keys[self.current_motion_idx]
        motion = self.motion_data[motion_key]
        
        print(f"Retargeting and saving motion: {motion_key}")
        
        # Get full retargeted motion
        pose_aa = torch.from_numpy(motion['pose_aa']).unsqueeze(0)
        root_trans = torch.from_numpy(motion['root_trans_offset']).unsqueeze(0)
        
        fk_result = self.humanoid_fk.fk_batch(pose_aa, root_trans, return_full=True, dt=self.dt)
        
        # Prepare save data
        save_data = {
            'original_motion_key': motion_key,
            'fps': motion.get('fps', 30),
            'num_frames': pose_aa.shape[1],
            'root_trans': fk_result['global_translation'][0, :, 0].numpy(),
            'root_rot': fk_result['global_rotation'][0, :, 0].numpy(),
            'joint_positions': fk_result['global_translation'][0].numpy(),
            'joint_rotations': fk_result['global_rotation'][0].numpy(),
            'joint_velocities': fk_result['global_velocity'][0].numpy() if 'global_velocity' in fk_result else None,
            'joint_angular_velocities': fk_result['global_angular_velocity'][0].numpy() if 'global_angular_velocity' in fk_result else None,
        }
        
        if 'dof_pos' in fk_result:
            save_data['dof_pos'] = fk_result['dof_pos'][0].numpy()
        if 'dof_vels' in fk_result:
            save_data['dof_vels'] = fk_result['dof_vels'][0].numpy()
            
        # Save to file
        output_dir = Path("retargeted_motions")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"retargeted_{motion_key}.pkl"
        
        joblib.dump(save_data, output_file)
        print(f"Saved retargeted motion to: {output_file}")
        
    def visualize_with_mujoco(self, robot_xml_path):
        """Visualize the retargeted motion using MuJoCo viewer."""
        # Load MuJoCo model
        mj_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        mj_data = mujoco.MjData(mj_model)
        mj_model.opt.timestep = self.dt
        
        print(f"Visualizing motion file: {self.motion_file}")
        print(f"Number of motions: {len(self.motion_keys)}")
        print("Controls:")
        print("  Space: Pause/Resume")
        print("  R: Reset to beginning")
        print("  S: Save current retargeted motion (if enabled)")
        print("  N/P: Next/Previous motion")
        print("  L/K: Speed up/down")
        print("  Left/Right arrows: Step backward/forward")
        print("  Q/Esc: Quit")
        
        with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=self.key_callback) as viewer:
            # Set camera
            viewer.cam.lookat[:] = np.array([0, 0, 0.8])
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            
            # Add visual markers for joints
            for _ in range(30):
                add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 
                                 0.03, np.array([1, 0, 0, 1]))
            
            while viewer.is_running():
                step_start = time.time()
                
                # Get current motion and frame
                motion_key = self.motion_keys[self.current_motion_idx]
                motion = self.motion_data[motion_key]
                num_frames = motion['pose_aa'].shape[0]
                
                # Calculate current frame
                if self.time_step >= num_frames * self.dt:
                    self.time_step = 0
                elif self.time_step < 0:
                    self.time_step = (num_frames - 1) * self.dt
                    
                curr_frame = int(self.time_step / self.dt) % num_frames
                
                # Get retargeted state
                frame_data = self.get_retargeted_state(motion_key, curr_frame)
                
                # Update MuJoCo simulation
                if 'root_trans_offset' in motion:
                    mj_data.qpos[:3] = motion['root_trans_offset'][curr_frame]
                else:
                    mj_data.qpos[:3] = frame_data['root_pos']
                    
                if 'root_rot' in motion:
                    # Convert to MuJoCo quaternion format (wxyz)
                    root_quat = motion['root_rot'][curr_frame]
                    mj_data.qpos[3:7] = root_quat[[3, 0, 1, 2]]
                else:
                    mj_data.qpos[3:7] = frame_data['root_rot'][[3, 0, 1, 2]]
                    
                if 'dof' in motion:
                    mj_data.qpos[7:] = motion['dof'][curr_frame]
                elif frame_data['dof_pos'] is not None:
                    mj_data.qpos[7:] = frame_data['dof_pos']
                
                mujoco.mj_forward(mj_model, mj_data)
                
                # Update joint visualization
                for i in range(min(len(frame_data['joint_pos']), 30)):
                    viewer.user_scn.geoms[i].pos = frame_data['joint_pos'][i]
                
                # Update time
                if not self.paused:
                    self.time_step += self.dt * self.speed
                
                # Display info
                print(f"Motion: {motion_key} | Frame: {curr_frame}/{num_frames} | "
                      f"Time: {self.time_step:.2f}s | Speed: {self.speed:.1f}x", end='\r')
                
                viewer.sync()
                
                # Maintain framerate
                time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    parser = argparse.ArgumentParser(description="Visualize and save retargeted motion from SMPL to robot")
    parser.add_argument("--motion_file", type=str, 
                       default="example/motion_data/Horse-stance_pose.pkl",
                       help="Path to motion data file")
    parser.add_argument("--robot_config", type=str,
                       default="humanoidverse/config/robot/t1/t1_23dof.yaml",
                       help="Path to robot configuration YAML")
    parser.add_argument("--robot_xml", type=str,
                       default="description/robots/t1/t1_23dof_torso_head.xml",
                       help="Path to robot MuJoCo XML file")
    parser.add_argument("--save", action="store_true",
                       help="Enable saving retargeted motion (press 'S' during visualization)")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = RetargetedMotionVisualizer(
        motion_file=args.motion_file,
        robot_config_path=args.robot_config,
        save_retargeted=args.save
    )
    
    # Run visualization
    visualizer.visualize_with_mujoco(args.robot_xml)


if __name__ == "__main__":
    main()