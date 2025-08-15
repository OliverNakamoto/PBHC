#!/usr/bin/env python3
"""
Simple script to save and visualize retargeted motion data.
This directly accesses the motion library's retargeted data.
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

from humanoidverse.utils.motion_lib.motion_lib_robot_WJX import MotionLibRobotWJX


def extract_and_save_retargeted_motion(motion_file, robot_config_path, output_dir):
    """Extract retargeted motion from the motion library."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load robot configuration
    robot_cfg = OmegaConf.load(robot_config_path)
    
    # Initialize motion library with full robot motion config
    motion_lib_cfg = robot_cfg.robot.motion.copy()
    motion_lib_cfg.motion_file = motion_file
    motion_lib_cfg.step_dt = 1/30.0
    
    # Use CPU to avoid device issues
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize motion library
    print("Initializing motion library...")
    motion_lib = MotionLibRobotWJX(motion_lib_cfg, num_envs=1, device=device)
    
    # Load motions
    print("Loading motions...")
    motion_lib.load_motions(random_sample=False, start_idx=0)
    
    # Extract motion data from the library
    print("Extracting retargeted motion data...")
    
    # Get motion metadata
    num_frames = motion_lib._motion_num_frames[0].item()
    fps = motion_lib._motion_fps[0].item()
    dt = motion_lib._motion_dt[0].item()
    motion_length = motion_lib._motion_lengths[0].item()
    
    print(f"Motion info: {num_frames} frames, {fps} FPS, {motion_length:.2f}s duration")
    
    # Extract motion states at each frame
    motion_times = torch.arange(0, num_frames) * dt
    motion_ids = torch.zeros(num_frames, dtype=torch.long)
    
    all_states = []
    for t in tqdm(range(num_frames), desc="Extracting frames"):
        state = motion_lib.get_motion_state(
            motion_ids[:1], 
            motion_times[t:t+1].to(device)
        )
        all_states.append({k: v.cpu().numpy() if torch.is_tensor(v) else v 
                         for k, v in state.items()})
    
    # Combine all frames
    retargeted_data = {
        'num_frames': num_frames,
        'fps': fps,
        'dt': dt,
        'motion_length': motion_length,
        'motion_key': motion_lib.curr_motion_keys[0] if hasattr(motion_lib, 'curr_motion_keys') else 'unknown',
    }
    
    # Stack frame data
    for key in all_states[0].keys():
        if key in ['root_pos', 'root_rot', 'dof_pos', 'root_vel', 'root_ang_vel', 'dof_vel']:
            retargeted_data[key] = np.stack([s[key].squeeze() for s in all_states])
        elif key in ['rg_pos', 'rb_rot', 'body_vel', 'body_ang_vel']:
            # Multi-body data
            retargeted_data[key] = np.stack([s[key].squeeze() for s in all_states])
    
    # Save the retargeted motion
    output_file = output_path / "retargeted_motion.pkl"
    joblib.dump(retargeted_data, output_file)
    print(f"\nSaved retargeted motion to: {output_file}")
    
    # Save summary
    summary = {
        'num_frames': num_frames,
        'fps': fps,
        'motion_length': motion_length,
        'robot_config': str(robot_config_path),
        'original_motion_file': str(motion_file),
        'available_keys': list(retargeted_data.keys()),
    }
    
    summary_file = output_path / "summary.txt"
    with open(summary_file, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Saved summary to: {summary_file}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract and save retargeted motion data")
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
    
    # Extract and save
    output_path = extract_and_save_retargeted_motion(
        args.motion_file,
        args.robot_config,
        args.output_dir
    )
    
    print(f"\n✓ Extraction complete!")
    print(f"  You can now visualize using:")
    print(f"    python visualize_retargeted.py --motion_dir {output_path}")


if __name__ == "__main__":
    main()