#!/usr/bin/env python3
"""
Visualize saved retargeted motion data on the Booster T1 robot.
This script loads pre-retargeted motion data and visualizes it using MuJoCo.
"""

import os
import sys
import numpy as np
import joblib
import mujoco
import mujoco.viewer
import time
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as sRot


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


class RetargetedMotionPlayer:
    def __init__(self, motion_dir, robot_xml):
        self.motion_dir = Path(motion_dir)
        self.robot_xml = robot_xml
        
        # Load retargeted motions
        self.load_retargeted_motions()
        
        # Control variables
        self.paused = False
        self.time_step = 0
        self.speed = 1.0
        self.current_motion_idx = 0
        self.show_joints = True
        self.show_contacts = False
        self.loop_motion = True
        
    def load_retargeted_motions(self):
        """Load all retargeted motion files from directory."""
        # Try to load combined file first
        combined_file = self.motion_dir / "all_retargeted_motions.pkl"
        if combined_file.exists():
            print(f"Loading retargeted motions from: {combined_file}")
            all_data = joblib.load(combined_file)
            self.motions = all_data
            self.motion_keys = list(all_data.keys())
        else:
            # Load individual files
            pkl_files = list(self.motion_dir.glob("retargeted_*.pkl"))
            if not pkl_files:
                raise ValueError(f"No retargeted motion files found in {self.motion_dir}")
            
            print(f"Found {len(pkl_files)} retargeted motion files")
            self.motions = {}
            for pkl_file in pkl_files:
                motion_data = joblib.load(pkl_file)
                key = pkl_file.stem.replace("retargeted_", "")
                self.motions[key] = motion_data
            self.motion_keys = list(self.motions.keys())
        
        print(f"Loaded {len(self.motion_keys)} motions: {self.motion_keys}")
        
    def key_callback(self, keycode):
        """Handle keyboard input during visualization."""
        if chr(keycode) == " ":
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Playing'}")
        elif chr(keycode) == "R":
            self.time_step = 0
            print("Reset to beginning")
        elif chr(keycode) == "N":
            self.next_motion()
        elif chr(keycode) == "P":
            self.prev_motion()
        elif chr(keycode) == "L":
            self.speed = min(self.speed * 1.5, 10.0)
            print(f"Speed: {self.speed:.2f}x")
        elif chr(keycode) == "K":
            self.speed = max(self.speed / 1.5, 0.1)
            print(f"Speed: {self.speed:.2f}x")
        elif chr(keycode) == "J":
            self.show_joints = not self.show_joints
            print(f"Joint visualization: {'ON' if self.show_joints else 'OFF'}")
        elif chr(keycode) == "C":
            self.show_contacts = not self.show_contacts
            print(f"Contact visualization: {'ON' if self.show_contacts else 'OFF'}")
        elif chr(keycode) == "O":
            self.loop_motion = not self.loop_motion
            print(f"Loop motion: {'ON' if self.loop_motion else 'OFF'}")
        elif chr(keycode) == "H":
            self.print_help()
        elif keycode == 256 or chr(keycode) == "Q":
            print("Exit")
            os._exit(0)
        elif keycode == 262:  # Right arrow
            self.time_step += self.get_current_dt()
        elif keycode == 263:  # Left arrow
            self.time_step -= self.get_current_dt()
            
    def print_help(self):
        """Print help information."""
        print("\n=== Controls ===")
        print("  Space:      Pause/Resume")
        print("  R:          Reset to beginning")
        print("  N/P:        Next/Previous motion")
        print("  L/K:        Speed up/down")
        print("  J:          Toggle joint visualization")
        print("  C:          Toggle contact visualization")
        print("  O:          Toggle loop motion")
        print("  ←/→:        Step backward/forward")
        print("  H:          Show this help")
        print("  Q/Esc:      Quit")
        print("================\n")
        
    def next_motion(self):
        """Switch to next motion in the dataset."""
        self.current_motion_idx = (self.current_motion_idx + 1) % len(self.motion_keys)
        self.time_step = 0
        motion_key = self.motion_keys[self.current_motion_idx]
        print(f"Switched to motion [{self.current_motion_idx+1}/{len(self.motion_keys)}]: {motion_key}")
        
    def prev_motion(self):
        """Switch to previous motion in the dataset."""
        self.current_motion_idx = (self.current_motion_idx - 1) % len(self.motion_keys)
        self.time_step = 0
        motion_key = self.motion_keys[self.current_motion_idx]
        print(f"Switched to motion [{self.current_motion_idx+1}/{len(self.motion_keys)}]: {motion_key}")
        
    def get_current_motion(self):
        """Get current motion data."""
        motion_key = self.motion_keys[self.current_motion_idx]
        return self.motions[motion_key], motion_key
        
    def get_current_dt(self):
        """Get timestep for current motion."""
        motion, _ = self.get_current_motion()
        return motion.get('dt', 1/30.0)
        
    def visualize(self):
        """Main visualization loop using MuJoCo viewer."""
        # Load MuJoCo model
        mj_model = mujoco.MjModel.from_xml_path(self.robot_xml)
        mj_data = mujoco.MjData(mj_model)
        
        # Get initial motion
        motion, motion_key = self.get_current_motion()
        mj_model.opt.timestep = self.get_current_dt()
        
        print(f"\n=== Retargeted Motion Visualizer ===")
        print(f"Robot model: {self.robot_xml}")
        print(f"Motion directory: {self.motion_dir}")
        print(f"Number of motions: {len(self.motion_keys)}")
        self.print_help()
        print(f"\nStarting with motion: {motion_key}")
        
        with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=self.key_callback) as viewer:
            # Set camera
            viewer.cam.lookat[:] = np.array([0, 0, 0.8])
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            
            # Add visual markers for joints
            joint_markers = []
            for i in range(30):
                add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 
                                 0.03, np.array([0.2, 0.6, 1.0, 0.8]))
                
            # Add contact markers (feet)
            left_foot_marker = len(joint_markers)
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 
                             0.05, np.array([0, 1, 0, 1]))
            right_foot_marker = left_foot_marker + 1
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 
                             0.05, np.array([0, 1, 0, 1]))
            
            while viewer.is_running():
                step_start = time.time()
                
                # Get current motion
                motion, motion_key = self.get_current_motion()
                dt = self.get_current_dt()
                num_frames = motion['num_frames']
                
                # Update timestep if dt changed
                if mj_model.opt.timestep != dt:
                    mj_model.opt.timestep = dt
                
                # Calculate current frame
                if self.loop_motion:
                    if self.time_step >= num_frames * dt:
                        self.time_step = 0
                    elif self.time_step < 0:
                        self.time_step = (num_frames - 1) * dt
                else:
                    self.time_step = np.clip(self.time_step, 0, (num_frames - 1) * dt)
                    
                curr_frame = int(self.time_step / dt) % num_frames
                
                # Update robot pose
                if 'root_trans' in motion and motion['root_trans'] is not None:
                    mj_data.qpos[:3] = motion['root_trans'][curr_frame]
                elif 'original_root_trans_offset' in motion:
                    mj_data.qpos[:3] = motion['original_root_trans_offset'][curr_frame]
                
                if 'root_rot' in motion and motion['root_rot'] is not None:
                    # Quaternion in xyzw format, convert to wxyz for MuJoCo
                    root_quat = motion['root_rot'][curr_frame]
                    mj_data.qpos[3:7] = root_quat[[3, 0, 1, 2]]
                elif 'original_root_rot' in motion:
                    root_quat = motion['original_root_rot'][curr_frame]
                    mj_data.qpos[3:7] = root_quat[[3, 0, 1, 2]]
                    
                # Update joint positions
                if 'dof_pos' in motion and motion['dof_pos'] is not None:
                    dof_data = motion['dof_pos'][curr_frame]
                    if len(dof_data) <= len(mj_data.qpos[7:]):
                        mj_data.qpos[7:7+len(dof_data)] = dof_data
                elif 'original_dof' in motion:
                    mj_data.qpos[7:] = motion['original_dof'][curr_frame]
                
                mujoco.mj_forward(mj_model, mj_data)
                
                # Update joint visualization
                if self.show_joints and 'joint_positions' in motion:
                    joint_pos = motion['joint_positions'][curr_frame]
                    for i in range(min(len(joint_pos), 30)):
                        viewer.user_scn.geoms[i].pos = joint_pos[i]
                        # Color based on height
                        height = joint_pos[i][2]
                        color_val = np.clip(height / 2.0, 0, 1)
                        viewer.user_scn.geoms[i].rgba = np.array([0.2, 0.6*color_val, 1.0-0.5*color_val, 0.8])
                
                # Update contact visualization
                if self.show_contacts and 'contact_mask' in motion:
                    contacts = motion['contact_mask'][curr_frame]
                    if len(contacts) >= 2:
                        # Left foot (index 6 in joint list typically)
                        left_contact = contacts[0]
                        viewer.user_scn.geoms[30].rgba = np.array([1-left_contact, left_contact, 0, 0.8])
                        
                        # Right foot (index 12 in joint list typically)
                        right_contact = contacts[1]
                        viewer.user_scn.geoms[31].rgba = np.array([1-right_contact, right_contact, 0, 0.8])
                
                # Update time
                if not self.paused:
                    self.time_step += dt * self.speed
                    
                    # Check if we've reached the end and not looping
                    if not self.loop_motion and self.time_step >= (num_frames - 1) * dt:
                        self.paused = True
                        print("Reached end of motion (press Space to restart)")
                
                # Display info
                fps = motion.get('fps', 30)
                time_str = f"{self.time_step:.2f}/{(num_frames-1)*dt:.2f}s"
                print(f"[{self.current_motion_idx+1}/{len(self.motion_keys)}] {motion_key} | "
                      f"Frame: {curr_frame+1}/{num_frames} | Time: {time_str} | "
                      f"FPS: {fps} | Speed: {self.speed:.1f}x", end='\r')
                
                viewer.sync()
                
                # Maintain framerate
                time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    parser = argparse.ArgumentParser(description="Visualize saved retargeted motion on robot")
    parser.add_argument("--motion_dir", type=str,
                       default="retargeted_motions",
                       help="Directory containing retargeted motion files")
    parser.add_argument("--robot_xml", type=str,
                       default="description/robots/t1/t1_23dof.xml",
                       help="Path to robot MuJoCo XML file")
    
    args = parser.parse_args()
    
    # Check if motion directory exists
    if not Path(args.motion_dir).exists():
        print(f"Error: Motion directory '{args.motion_dir}' does not exist!")
        print("Please run 'python retarget_and_save.py' first to generate retargeted motions.")
        sys.exit(1)
    
    # Check if robot XML exists
    if not Path(args.robot_xml).exists():
        print(f"Error: Robot XML file '{args.robot_xml}' does not exist!")
        sys.exit(1)
    
    # Create player and visualize
    player = RetargetedMotionPlayer(
        motion_dir=args.motion_dir,
        robot_xml=args.robot_xml
    )
    
    player.visualize()


if __name__ == "__main__":
    main()