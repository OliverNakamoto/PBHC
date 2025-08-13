#!/usr/bin/env python3
import os
import sys
sys.path.append('/home/oliver/PBHC')

from isaacgym import gymapi

# Initialize gym
gym = gymapi.acquire_gym()

# Create sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.dt = 0.002
sim_params.substeps = 1
sim_params.use_gpu_pipeline = False

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Load robot asset
asset_root = "/home/oliver/PBHC/description/robots"
asset_file = "t1/t1_23dof.urdf"

asset_options = gymapi.AssetOptions()
asset_options.collapse_fixed_joints = True
asset_options.replace_cylinder_with_capsule = True
asset_options.flip_visual_attachments = False
asset_options.fix_base_link = False
asset_options.density = 0.001
asset_options.angular_damping = 0.0
asset_options.linear_damping = 0.0
asset_options.max_angular_velocity = 1000.0
asset_options.max_linear_velocity = 1000.0
asset_options.armature = 0.001
asset_options.thickness = 0.01
asset_options.default_dof_drive_mode = 3

robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Get body info
num_bodies = gym.get_asset_rigid_body_count(robot_asset)
body_names = gym.get_asset_rigid_body_names(robot_asset)

print(f"Number of bodies: {num_bodies}")
print(f"Body names ({len(body_names)}):")
for i, name in enumerate(body_names):
    print(f"  {i}: {name}")

# Get DOF info
num_dof = gym.get_asset_dof_count(robot_asset)
dof_names = gym.get_asset_dof_names(robot_asset)

print(f"\nNumber of DOFs: {num_dof}")
print(f"DOF names ({len(dof_names)}):")
for i, name in enumerate(dof_names):
    print(f"  {i}: {name}")

# Clean up
gym.destroy_sim(sim)