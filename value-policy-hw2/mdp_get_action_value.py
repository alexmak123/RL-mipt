def get_action_value(mdp, state_values, state, action, gamma):
    """ Вычисляет Q(s,a) согласно формуле выше """
    
    # Ваша имплементация ниже
    Q = 0
    
    for next_state in mdp.get_next_states(state, action):
        
        transition_prob = mdp.get_transition_prob(state, action, next_state)
        reward = mdp.get_reward(state, action, next_state)
    
        Q += transition_prob * (reward + gamma * state_values[next_state])
    
    return Q
