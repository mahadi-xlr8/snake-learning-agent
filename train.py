
import argparse
import csv
import subprocess
import sys

from game.snake_game import SnakeGame

from agent.state import get_state
from agent.actions import action_to_direction
from agent.exploration import choose_action, get_epsilon
from agent.evaluation import play_greedy_episode, starvation_limit
from agent.rewards import calculate_reward
from agent.replay_memory import ReplayMemory
from agent.learning import DQNTrainer

from config import (
    MODEL_TYPE,
    TRAIN_EPISODES,
    SHOW_GAME_WINDOW,
    SHOW_TRAINING_PLOT,
    SAVE_TRAINING_CSV,
    RENDER_FPS,
    RENDER_EVERY_N_EPISODES,
)


# Settings that work well for each model type.
# The CNN learns from raw pixels, so it needs a smaller learning
# rate and a longer exploration phase than the MLP.
MODEL_CONFIG = {
    "mlp": {
        "learning_rate": 0.001,
        "epsilon_decay": 300,
        "batch_size": 256,
    },
    "cnn": {
        "learning_rate": 0.00025,
        "epsilon_decay": 2000,
        "batch_size": 64,
    },
}

MEMORY_CAPACITY = 100_000
LEARNING_STARTS = 1_000
TARGET_UPDATE_STEPS = 1_000

EVAL_EVERY_N_EPISODES = 100
EVAL_EPISODES = 10


def food_distance(game):
    """
    Calculate Manhattan distance between the
    snake's head and the food.
    """
    if game.food is None:
        return None

    head_x, head_y = game.snake[0]
    food_x, food_y = game.food

    return abs(head_x - food_x) + abs(head_y - food_y)


def evaluate(trainer, model_type):
    """Play a few games with no random moves to measure real skill."""
    game = SnakeGame()
    model = trainer.policy_net
    model.eval()

    scores = [
        play_greedy_episode(model, model_type, game)
        for _ in range(EVAL_EPISODES)
    ]

    model.train()
    return scores


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Snake DQN agent")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIG),
        default=MODEL_TYPE,
        help="which network to train",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=TRAIN_EPISODES,
        help="number of training episodes",
    )
    return parser.parse_args()


def train(model_type, episodes):
    settings = MODEL_CONFIG[model_type]

    game = SnakeGame()
    trainer = DQNTrainer(
        model_type=model_type,
        learning_rate=settings["learning_rate"],
    )
    memory = ReplayMemory(capacity=MEMORY_CAPACITY)

    batch_size = settings["batch_size"]
    model_path = f"snake_dqn_{model_type}.pth"
    best_model_path = f"snake_dqn_{model_type}_best.pth"
    history_path = f"training_history_{model_type}.csv"

    history = []
    total_steps = 0
    best_eval_average = -1.0

    pygame = None
    renderer = None
    clock = None
    window_open = SHOW_GAME_WINDOW

    if SHOW_GAME_WINDOW:
        import pygame as pygame_module
        from game.renderer import Renderer

        pygame = pygame_module
        pygame.init()

        renderer = Renderer(game.size)
        clock = pygame.time.Clock()

    print(f"Training the {model_type.upper()} model for {episodes} episodes")

    try:
        for episode in range(episodes):
            game.reset()

            state = get_state(game, model_type)
            epsilon = get_epsilon(
                episode,
                decay=settings["epsilon_decay"],
            )

            done = False
            total_reward = 0.0
            steps = 0
            steps_without_food = 0
            starved = False

            render_episode = (
                window_open
                and episode % RENDER_EVERY_N_EPISODES == 0
            )

            while not done:
                # Measure distance before moving
                old_distance = food_distance(game)

                # Choose an action
                action = choose_action(
                    trainer.policy_net,
                    state,
                    epsilon,
                )

                direction = action_to_direction(
                    game.direction,
                    action,
                )

                # Move the snake
                _, score_delta, done, info = game.step(direction)

                steps += 1
                total_steps += 1

                # Measure distance after moving
                # Only use this when food was not eaten.
                new_distance = None

                if score_delta == 0 and not done:
                    new_distance = food_distance(game)

                # Track moves without food
                if score_delta > 0:
                    steps_without_food = 0
                else:
                    steps_without_food += 1

                # Calculate reward
                reward = calculate_reward(
                    score_delta=score_delta,
                    done=done,
                    won=info["won"],
                    old_distance=old_distance,
                    new_distance=new_distance,
                )

                # Get next state
                next_state = get_state(game, model_type)

                # Store experience.
                # Starvation is a time limit, not a real death, so it
                # is stored as "not done": the future still counts.
                memory.push(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                # Starvation limit: stop the episode
                if (
                    not done
                    and steps_without_food >= starvation_limit(game)
                ):
                    done = True
                    starved = True

                # Train the network
                if len(memory) >= max(batch_size, LEARNING_STARTS):
                    batch = memory.sample(batch_size)
                    trainer.train_step(batch)

                # Update target network
                if total_steps % TARGET_UPDATE_STEPS == 0:
                    trainer.update_target()

                state = next_state
                total_reward += reward

                # Handle Pygame events and rendering
                if window_open:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            window_open = False
                            pygame.display.quit()

                    if render_episode and window_open:
                        renderer.draw(game)
                        clock.tick(RENDER_FPS)

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Score: {game.score} | "
                f"Steps: {steps} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {epsilon:.2f} | "
                f"Starved: {starved}",
                flush=True,
            )

            row = {
                "episode": episode + 1,
                "score": game.score,
                "steps": steps,
                "reward": total_reward,
                "epsilon": epsilon,
                "starved": starved,
                "eval_average": "",
            }

            # Measure skill without random moves
            if (episode + 1) % EVAL_EVERY_N_EPISODES == 0:
                scores = evaluate(trainer, model_type)
                average = sum(scores) / len(scores)
                row["eval_average"] = average

                print(
                    f"  Evaluation (no random moves) | "
                    f"Average: {average:.1f} | "
                    f"Best: {max(scores)}",
                    flush=True,
                )

                if average > best_eval_average:
                    best_eval_average = average
                    trainer.save(best_model_path)
                    print(f"  New best model saved to {best_model_path}")

            history.append(row)

    finally:
        if pygame is not None:
            pygame.quit()

    # Save training statistics
    if SAVE_TRAINING_CSV:
        with open(history_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "episode",
                    "score",
                    "steps",
                    "reward",
                    "epsilon",
                    "starved",
                    "eval_average",
                ],
            )

            writer.writeheader()
            writer.writerows(history)

        print(f"Training history saved to {history_path}")

    # Save trained model
    trainer.save(model_path)

    print(f"Model saved to {model_path}")

    # Show graph
    if SHOW_TRAINING_PLOT:
        if SAVE_TRAINING_CSV:
            subprocess.run(
                [sys.executable, "plot_training.py", history_path],
                check=True,
            )
        else:
            print(
                "Training plot skipped: "
                "SAVE_TRAINING_CSV is disabled."
            )

    print("Training complete!")


if __name__ == "__main__":
    args = parse_args()
    train(args.model, args.episodes)
