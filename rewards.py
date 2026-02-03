import numpy as np
import mujoco


def compute_stand_reward(model, data, target_pose, action=None, last_action=None):
    """
    Compute reward for the duck robot standing.
    Args:
        model: mujoco.MjModel object
        data: mujoco.MjData object
        target_pose: np.array of target joint positions
        action: np.array of current actions (torques)
        last_action: np.array of previous actions (for smoothness)
    """
    height = data.body('trunk_assembly').xpos[2]
    up_vector = data.body('trunk_assembly').xmat[8]  # z-component of up direction
    current_joints = data.qpos[7:]  # Current joint positions
    
    # ============ 1. POSE IMITATION REWARD (MOST IMPORTANT) ============
    # Reward for matching the target standing pose
    pose_error = np.sum((current_joints - target_pose) ** 2)
    # Relaxed decay from -2.0 to -0.5 to give gradients even when far
    pose_reward = 10.0 * np.exp(-0.5 * pose_error) 
    
    # ============ 2. UPRIGHT BONUS ============
    upright_reward = 5.0 * max(0, up_vector) ** 2  # Max 5 when perfectly upright
    
    # ============ 3. HEIGHT BONUS ============
    target_height = 0.15  # Target standing height
    # Relaxed decay from -100.0 to -10.0 to strictly encourage lifting
    height_reward = 3.0 * np.exp(-10.0 * (height - target_height) ** 2)
    
    # ============ 4. STABILITY BONUS ============
    # Reward for low velocity (being still while standing)
    joint_vel = np.sum(np.abs(data.qvel[6:]))
    body_vel = np.linalg.norm(data.body('trunk_assembly').cvel[:3])
    if up_vector > 0.5 and height > 0.10:
        stability_reward = 2.0 * np.exp(-0.1 * (joint_vel + body_vel))
    else:
        stability_reward = 0.0
    
    # ============ 5. ACTION SMOOTHNESS (CRITICAL FOR TORQUE CONTROL) ============
    action_penalty = 0.0
    if action is not None and last_action is not None:
        # Penalize large changes in torque
        action_diff = np.sum((action - last_action) ** 2)
        action_penalty = 1.0 * action_diff
        
    # ============ 6. ALIVE BONUS ============
    alive_bonus = 2  # Increased slightly
    
    # ============ 7. ENERGY PENALTY (small) ============
    energy_penalty = 0.0015 * np.sum(np.square(np.array(data.ctrl)))
    
    # ============ 8. FOOT CONTACT REWARD ============
    # Encourage feet to stay close to the ground (z=0)
    left_foot_pos = data.body('left_foot').xpos
    right_foot_pos = data.body('right_foot').xpos
    
    # Target height 0.02m (allow small clearance/thickness)
    foot_target_h = 0.02
    
    # Reward for being close to ground
    left_contact = np.exp(-20.0 * (left_foot_pos[2] - foot_target_h)**2)
    right_contact = np.exp(-20.0 * (right_foot_pos[2] - foot_target_h)**2)
    
    foot_contact_reward = 1.0 * (left_contact + right_contact)

    # ============ 9. CENTRE OF MASS (CoM) STABILITY ============
    # Calculate Center of Mass of the whole robot
    try:
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        com_pos = data.subtree_com[base_id]  # (3,) array
        
        # Calculate center of feet (support center)
        feet_center_xy = (left_foot_pos[:2] + right_foot_pos[:2]) / 2.0
        
        # Minimize horizontal distance between CoM and feet center
        com_dist = np.sum((com_pos[:2] - feet_center_xy) ** 2)
        
        # Reward for centering CoM
        com_reward = 2.0 * np.exp(-10.0 * com_dist)
    except:
        # Fallback if 'base' body not found or logical error
        com_reward = 0.0

    # ============ 10. ORIENTATION PENALTY (Roll/Pitch) ============
    # Penalize tilting away from vertical (minimize Roll and Pitch)
    xmat = data.body('trunk_assembly').xmat
    
    r00, r10, r20 = xmat[0], xmat[3], xmat[6]
    r01, r11, r21 = xmat[1], xmat[4], xmat[7]
    r02, r12, r22 = xmat[2], xmat[5], xmat[8]
    
    sy = np.sqrt(r00*r00 + r10*r10)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(r21, r22)
        pitch = np.arctan2(-r20, sy)
    else:
        roll = np.arctan2(-r12, r11)
        pitch = np.arctan2(-r20, sy)
        
    # Penalty for any deviation from 0
    orientation_penalty = 5.0 * (np.abs(roll) + np.abs(pitch))

    # ============ 11. FEET SPACING PENALTY (Prevent Splits) ============
    feet_spacing = np.linalg.norm(left_foot_pos[:2] - right_foot_pos[:2])
    max_feet_spacing = 0.21  # tested max width
    spacing_penalty = 0.0
    if feet_spacing > max_feet_spacing:
        # Heavily penalize splitting
        spacing_penalty = 30.0 * (feet_spacing - max_feet_spacing) ** 2

    total_reward = (pose_reward + upright_reward + height_reward + stability_reward + 
                   alive_bonus + foot_contact_reward + com_reward - 
                   energy_penalty - action_penalty - orientation_penalty - spacing_penalty)
    
    reward_info = {
        "pose": pose_reward,
        "upright": upright_reward,
        "height": height_reward,
        "stability": stability_reward,
        "alive": alive_bonus,
        "foot": foot_contact_reward,
        "com": com_reward,
        "energy": -energy_penalty,
        "action": -action_penalty,
        "orient": -orientation_penalty,
        "spacing": -spacing_penalty
    }

    return total_reward, reward_info
