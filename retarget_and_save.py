#!/usr/bin/env python3
"""
Retarget motion data from SMPL to Booster T1 robot and save the results.
This script performs the retargeting without visualization and saves all data.
"""

import os
import sys
import torch
import numpy as np
import joblib
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(os.getcwd())

from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from humanoidverse.utils.motion_lib.motion_lib_robot_WJX import MotionLibRobotWJX


class MotionRetargeter:
    def __init__(self, motion_file, robot_config_path):
        self.motion_file = motion_file
        self.robot_config_path = robot_config_path
        
        # Load robot configuration
        self.robot_cfg = OmegaConf.load(robot_config_path)
        
        # Initialize motion library with full robot motion config
        motion_lib_cfg = self.robot_cfg.robot.motion.copy()
        motion_lib_cfg.motion_file = motion_file
        motion_lib_cfg.step_dt = 1/30.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize motion library
        self.motion_lib = MotionLibRobotWJX(motion_lib_cfg, num_envs=1, device=self.device)
        
        # Load motion data
        print(f"Loading motion data from: {motion_file}")
        self.motion_data = joblib.load(motion_file)
        self.motion_keys = list(self.motion_data.keys())
        
        # Initialize humanoid FK model for retargeting
        print(f"Loading robot model from: {robot_config_path}")
        self.humanoid_fk = Humanoid_Batch(self.robot_cfg.robot.motion)
        
        self.dt = 1/30.0  # Default framerate
        
    def retarget_motion(self, motion_key):
        """Retarget a single motion from SMPL to robot format."""
        motion = self.motion_data[motion_key]
        
        # Get motion parameters
        fps = motion.get('fps', 30)
        dt = 1.0 / fps
        
        # Get pose and translation data
        pose_aa = torch.from_numpy(motion['pose_aa']).unsqueeze(0).to(self.device)
        root_trans = torch.from_numpy(motion['root_trans_offset']).unsqueeze(0).to(self.device)
        
        print(f"  Motion shape: {pose_aa.shape[1]} frames, {pose_aa.shape[2]} joints")
        print(f"  FPS: {fps}")
        
        # Perform forward kinematics for retargeting
        with torch.no_grad():
            fk_result = self.humanoid_fk.fk_batch(pose_aa, root_trans, return_full=True, dt=dt)
        
        # Convert results to numpy and prepare save data
        retargeted_data = {
            'original_motion_key': motion_key,
            'fps': fps,
            'dt': dt,
            'num_frames': pose_aa.shape[1],
            'num_joints': pose_aa.shape[2],
            
            # Original SMPL data
            'original_pose_aa': motion['pose_aa'],
            'original_root_trans_offset': motion['root_trans_offset'],
            
            # Retargeted robot data - root
            'root_trans': fk_result['global_translation'][0, :, 0].cpu().numpy(),
            'root_rot': fk_result['global_rotation'][0, :, 0].cpu().numpy(),
            'root_vel': fk_result['global_velocity'][0, :, 0].cpu().numpy() if 'global_velocity' in fk_result else None,
            'root_ang_vel': fk_result['global_angular_velocity'][0, :, 0].cpu().numpy() if 'global_angular_velocity' in fk_result else None,
            
            # Retargeted robot data - all joints
            'joint_positions': fk_result['global_translation'][0].cpu().numpy(),
            'joint_rotations': fk_result['global_rotation'][0].cpu().numpy(),
            'joint_velocities': fk_result['global_velocity'][0].cpu().numpy() if 'global_velocity' in fk_result else None,
            'joint_angular_velocities': fk_result['global_angular_velocity'][0].cpu().numpy() if 'global_angular_velocity' in fk_result else None,
            
            # Local rotations
            'local_rotations': fk_result['local_rotation'][0].cpu().numpy() if 'local_rotation' in fk_result else None,
        }
        
        # Add DOF data if available
        if 'dof_pos' in fk_result:
            retargeted_data['dof_pos'] = fk_result['dof_pos'][0].cpu().numpy()
            print(f"  DOF positions shape: {retargeted_data['dof_pos'].shape}")
            
        if 'dof_vels' in fk_result:
            retargeted_data['dof_vels'] = fk_result['dof_vels'][0].cpu().numpy()
            print(f"  DOF velocities shape: {retargeted_data['dof_vels'].shape}")
        
        # Copy over original motion metadata if exists
        if 'contact_mask' in motion:
            retargeted_data['contact_mask'] = motion['contact_mask']
            
        if 'beta' in motion:
            retargeted_data['beta'] = motion['beta']
            
        if 'gender' in motion:
            retargeted_data['gender'] = motion['gender']
            
        # Add robot-specific data from original if it exists
        if 'root_rot' in motion:
            retargeted_data['original_root_rot'] = motion['root_rot']
        if 'dof' in motion:
            retargeted_data['original_dof'] = motion['dof']
            
        return retargeted_data
    
    def retarget_and_save_all(self, output_dir="retargeted_motions"):
        """Retarget all motions and save them."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\nRetargeting {len(self.motion_keys)} motions...")
        
        all_retargeted = {}
        
        for motion_key in tqdm(self.motion_keys, desc="Retargeting motions"):
            print(f"\nProcessing: {motion_key}")
            
            try:
                retargeted_data = self.retarget_motion(motion_key)
                all_retargeted[motion_key] = retargeted_data
                
                # Save individual motion
                individual_file = output_path / f"retargeted_{motion_key}.pkl"
                joblib.dump(retargeted_data, individual_file)
                print(f"  Saved to: {individual_file}")
                
            except Exception as e:
                print(f"  Error retargeting {motion_key}: {e}")
                continue
        
        # Save all motions in one file
        combined_file = output_path / "all_retargeted_motions.pkl"
        joblib.dump(all_retargeted, combined_file)
        print(f"\nSaved all retargeted motions to: {combined_file}")
        
        # Save summary
        summary = {
            'num_motions': len(all_retargeted),
            'motion_keys': list(all_retargeted.keys()),
            'robot_config': str(self.robot_config_path),
            'original_motion_file': str(self.motion_file),
        }
        
        summary_file = output_path / "retarget_summary.pkl"
        joblib.dump(summary, summary_file)
        print(f"Saved summary to: {summary_file}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Retarget motion from SMPL to robot and save")
    parser.add_argument("--motion_file", type=str, 
                       default="example/motion_data/Horse-stance_pose.pkl",
                       help="Path to motion data file")
    parser.add_argument("--robot_config", type=str,
                       default="humanoidverse/config/robot/t1/t1_23dof.yaml",
                       help="Path to robot configuration YAML")
    parser.add_argument("--output_dir", type=str,
                       default="retargeted_motions",
                       help="Directory to save retargeted motions")
    
    args = parser.parse_args()
    
    # Create retargeter
    retargeter = MotionRetargeter(
        motion_file=args.motion_file,
        robot_config_path=args.robot_config
    )
    
    # Retarget and save all motions
    output_path = retargeter.retarget_and_save_all(args.output_dir)
    
    print(f"\n✓ Retargeting complete!")
    print(f"  Output directory: {output_path}")
    print(f"  You can now visualize the retargeted motions using:")
    print(f"    python visualize_retargeted.py --motion_dir {output_path}")


if __name__ == "__main__":
    main()