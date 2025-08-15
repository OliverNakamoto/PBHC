#!/usr/bin/env python3
"""
Simple visualizer for retargeted motion data on the Booster T1 robot.
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


class SimpleMotionVisualizer:
    def __init__(self, motion_file, robot_xml):
        self.robot_xml = robot_xml
        
        # Load retargeted motion
        print(f"Loading retargeted motion from: {motion_file}")
        self.motion = joblib.load(motion_file)
        
        # Extract motion info
        self.num_frames = self.motion['num_frames']
        self.fps = self.motion['fps']
        self.dt = self.motion['dt']
        self.motion_length = self.motion['motion_length']
        
        print(f"Motion info: {self.num_frames} frames, {self.fps} FPS, {self.motion_length:.2f}s")
        
        # Control variables
        self.paused = False
        self.time_step = 0
        self.speed = 1.0
        self.show_info = True
        
    def key_callback(self, keycode):
        """Handle keyboard input."""
        if chr(keycode) == " ":
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Playing'}")
        elif chr(keycode) == "R":
            self.time_step = 0
            print("Reset to beginning")
        elif chr(keycode) == "L":
            self.speed = min(self.speed * 1.5, 10.0)
            print(f"Speed: {self.speed:.2f}x")
        elif chr(keycode) == "K":
            self.speed = max(self.speed / 1.5, 0.1)
            print(f"Speed: {self.speed:.2f}x")
        elif chr(keycode) == "I":
            self.show_info = not self.show_info
            print(f"Info display: {'ON' if self.show_info else 'OFF'}")
        elif chr(keycode) == "H":
            self.print_help()
        elif keycode == 256 or chr(keycode) == "Q":
            print("Exit")
            os._exit(0)
        elif keycode == 262:  # Right arrow
            self.time_step = min(self.time_step + self.dt, (self.num_frames - 1) * self.dt)
        elif keycode == 263:  # Left arrow
            self.time_step = max(self.time_step - self.dt, 0)
            
    def print_help(self):
        """Print help information."""
        print("\n=== Controls ===")
        print("  Space:      Pause/Resume")
        print("  R:          Reset to beginning")
        print("  L/K:        Speed up/down")
        print("  I:          Toggle info display")
        print("  ←/→:        Step backward/forward")
        print("  H:          Show this help")
        print("  Q/Esc:      Quit")
        print("================\n")
        
    def visualize(self):
        """Main visualization loop."""
        # Load MuJoCo model
        mj_model = mujoco.MjModel.from_xml_path(self.robot_xml)
        mj_data = mujoco.MjData(mj_model)
        mj_model.opt.timestep = self.dt
        
        print(f"\n=== Retargeted Motion Visualizer ===")
        print(f"Robot model: {self.robot_xml}")
        self.print_help()
        
        with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=self.key_callback) as viewer:
            # Set camera
            viewer.cam.lookat[:] = np.array([0, 0, 0.8])
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            
            while viewer.is_running():
                step_start = time.time()
                
                # Calculate current frame
                if self.time_step >= self.num_frames * self.dt:
                    self.time_step = 0
                elif self.time_step < 0:
                    self.time_step = (self.num_frames - 1) * self.dt
                    
                curr_frame = min(int(self.time_step / self.dt), self.num_frames - 1)
                
                # Update robot pose from retargeted data
                if 'root_pos' in self.motion:
                    mj_data.qpos[:3] = self.motion['root_pos'][curr_frame]
                    
                if 'root_rot' in self.motion:
                    # Quaternion in xyzw format, convert to wxyz for MuJoCo
                    root_quat = self.motion['root_rot'][curr_frame]
                    mj_data.qpos[3:7] = root_quat[[3, 0, 1, 2]]
                    
                if 'dof_pos' in self.motion:
                    dof_data = self.motion['dof_pos'][curr_frame]
                    # Ensure we don't exceed the model's DOF
                    num_dofs = min(len(dof_data), len(mj_data.qpos) - 7)
                    mj_data.qpos[7:7+num_dofs] = dof_data[:num_dofs]
                
                # Forward dynamics
                mujoco.mj_forward(mj_model, mj_data)
                
                # Update time
                if not self.paused:
                    self.time_step += self.dt * self.speed
                
                # Display info
                if self.show_info:
                    time_str = f"{self.time_step:.2f}/{self.motion_length:.2f}s"
                    print(f"Frame: {curr_frame+1}/{self.num_frames} | Time: {time_str} | "
                          f"Speed: {self.speed:.1f}x", end='\r')
                
                viewer.sync()
                
                # Maintain framerate
                time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    parser = argparse.ArgumentParser(description="Visualize retargeted motion on T1 robot")
    parser.add_argument("--motion_file", type=str,
                       default="retargeted_motions_v2/retargeted_motion.pkl",
                       help="Path to retargeted motion pickle file")
    parser.add_argument("--robot_xml", type=str,
                       default="description/robots/t1/t1_23dof.xml",
                       help="Path to robot MuJoCo XML file")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.motion_file).exists():
        print(f"Error: Motion file '{args.motion_file}' does not exist!")
        print("Please run 'python simple_retarget_save.py' first to generate retargeted motion.")
        sys.exit(1)
    
    if not Path(args.robot_xml).exists():
        print(f"Error: Robot XML file '{args.robot_xml}' does not exist!")
        sys.exit(1)
    
    # Create visualizer and run
    visualizer = SimpleMotionVisualizer(
        motion_file=args.motion_file,
        robot_xml=args.robot_xml
    )
    
    visualizer.visualize()


if __name__ == "__main__":
    main()