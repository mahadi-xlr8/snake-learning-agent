
def calculate_reward(
    score_delta,
    done,
    won=False,
    old_distance=None,
    new_distance=None,
):
    # Winning
    if won:
        return 20.0

    # Collision or starvation
    if done:
        return -10.0

    # Eating food
    if score_delta > 0:
        return 10.0

    # Small penalty for every ordinary move
    reward = -0.01

    # Reward progress toward food
    if old_distance is not None and new_distance is not None:
        if new_distance < old_distance:
            reward += 0.05
        elif new_distance > old_distance:
            reward -= 0.05

    return reward
