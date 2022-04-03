import events as e

def rewards(events):

    # punish or reward:
    reward = 0
    if e.COIN_COLLECTED in events:
        reward += 1
    if e.KILLED_OPPONENT in events:
        reward += 5

    return reward