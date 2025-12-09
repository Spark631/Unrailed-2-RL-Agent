def compute_reward(events, reward_config):
    """
    Compute the total reward based on a list of events and a reward configuration.
    
    Args:
        events (list): List of event strings (e.g., ['chop', 'pickup'])
        reward_config (dict): Dictionary mapping event names to reward values
        
    Returns:
        float: Total reward
    """
    reward = reward_config.get('step_penalty', -0.01)
    
    for event in events:
        if event in reward_config:
            reward += reward_config[event]
            
    return reward