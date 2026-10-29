from game.constants import UP, RIGHT, DOWN, LEFT

DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

STRAIGHT = 0
TURN_RIGHT = 1
TURN_LEFT = 2


def action_to_direction(current_direction, action):
    index = DIRECTIONS.index(current_direction)

    if action == STRAIGHT:
        return DIRECTIONS[index]

    if action == TURN_RIGHT:
        return DIRECTIONS[(index + 1) % 4]

    if action == TURN_LEFT:
        return DIRECTIONS[(index - 1) % 4]

    raise ValueError(f"Invalid action: {action}")