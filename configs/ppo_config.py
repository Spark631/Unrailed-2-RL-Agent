# Config 3: Full Game (Balanced)
FULL_GAME_REWARDS = {
    'step_penalty': -0.02,      # Encourages speed
    'goal_reached': 100.0,      # Main goal
    'chop': 0.5,                # Resource gathering
    'mine': 0.5,
    'pickup': 0.1,
    'pickup_rail': 0.5,
    'place_rail': 5.0,          # High reward for placing tracks
    'deposit': 0.1,
    'craft_complete': 2.0,      # Reward for making rails
    'oscillation': -2.0         # Penalty for moving back to the previous position immediately
}

REWARD_WEIGHTS = {
    "success": 100.0,
    "crash": -80.0,
    "track": 5.0,
    "wood": 0.5,
    "stone": 0.5,
    "craft": 2.0,
    "step": -0.02,
    "invalid": -0.2,
    "dist": 0.5,
}

# Config 1: Movement & Navigation
MOVEMENT_REWARDS = {
    'step_penalty': -0.1,
    'wall_bump': -1.0,          
    'goal_reached': 100.0,
    'timeout' : -100.0,
    'mine': 0.0,
    'pickup': 0.0,
    'pickup_rail': 0.0,
    'place_rail': 0.0,
    'deposit': 0.0,
    'craft_complete': 0.0,
    'action_interact': -0.1,
    'action_stay': -0.1,
    'progress_scale' : 0.2
}

# Config 2: Resource Gathering
GATHERING_REWARDS = {
    'step_penalty': -0.01,
    'wall_bump': -0.5,
    'gathering_complete': 100.0,
    'chop': 100.0,
    'mine': 100.0,
    'pickup': 5.0,
    'drop': -6.0,
    'pickup_rail': 0.0,
    'action_interact': -0.05,
    'action_stay': -0.01,
    'place_rail': 0.0,
    'deposit': 0.0,
    'craft_complete': 0.0,
    'progress_scale': 1.0,
    'gathering_progress': 0.5,
    'ready_to_gather': 0.1
}