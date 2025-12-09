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
    'step_penalty': -0.02,
    'wall_bump': -0.1,          
    'goal_reached': 0.0,        
    'chop': 5.0,                
    'mine': 5.0,                
    'pickup': 0.5,              
    'pickup_rail': 0.0,
    'action_interact': 0.0,
    'action_stay': -0.1,
    'place_rail': 0.0,
    'deposit': 2.0,             #
    'craft_complete': 10.0,     #
    'dist_closer': 0.0,        
    'dist_farther': 0.0,
    'oscillation': -0.5         
}