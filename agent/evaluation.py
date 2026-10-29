
from agent.actions import action_to_direction
from agent.exploration import best_action
from agent.state import get_state


# The snake may take this many steps per body segment before
# the episode is cut off for wandering without eating.
STEPS_PER_LENGTH = 100


def starvation_limit(game):
    return STEPS_PER_LENGTH * len(game.snake)


def play_greedy_episode(model, model_type, game, on_step=None):
    """Play one game with no random moves and return the final score.

    `on_step(game)` is called after every move (used for rendering).
    Return False from it to stop early.
    """
    game.reset()
    steps_without_food = 0

    while not game.done:
        state = get_state(game, model_type)
        action = best_action(model, state)

        _, score_delta, _, _ = game.step(
            action_to_direction(game.direction, action)
        )

        if score_delta > 0:
            steps_without_food = 0
        else:
            steps_without_food += 1

        if on_step is not None and on_step(game) is False:
            break

        if steps_without_food >= starvation_limit(game):
            break

    return game.score
