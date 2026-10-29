
def calculate_reward(score_delta, done, won=False):
    # Successfully filled the board
    if won:
        return 20.0

    # Collision or starvation
    if done:
        return -10.0

    # Ate food
    if score_delta > 0:
        return 10.0

    # Small penalty for wasting a move
    return -0.01
