
import csv
import subprocess
import sys

import torch

from game.snake_game import SnakeGame
from agent.state import get_state
from agent.actions import action_to_direction
from agent.exploration import choose_action, get_epsilon
from agent.rewards import calculate_reward
from agent.replay_memory import ReplayMemory
from agent.learning import DQNTrainer

from config import (
    TRAIN_EPISODES,
    SHOW_GAME_WINDOW,
    SHOW_TRAINING_PLOT,
    SAVE_TRAINING_CSV,
    RENDER_FPS,
    RENDER_EVERY_N_EPISODES,
)


MAX_STEPS_WITHOUT_FOOD = 200


def train():
    game = SnakeGame()
    trainer = DQNTrainer()
    memory = ReplayMemory(capacity=10_000)

    batch_size = 64
    target_update_frequency = 10

    history = []

    pygame = None
    renderer = None
    clock = None
    window_open = SHOW_GAME_WINDOW

    # Initialize visualization only when enabled
    if SHOW_GAME_WINDOW:
        import pygame as pygame_module
        from game.renderer import Renderer

        pygame = pygame_module
        pygame.init()

        renderer = Renderer(game.size)
        clock = pygame.time.Clock()

    try:
        for episode in range(TRAIN_EPISODES):
            game.reset()

            state = get_state(game)
            epsilon = get_epsilon(episode)

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
                # 1. Choose an action
                action = choose_action(
                    trainer.policy_net,
                    state,
                    epsilon,
                )

                # 2. Convert action to direction
                direction = action_to_direction(
                    game.direction,
                    action,
                )

                # 3. Move the snake
                _, score_delta, done, info = game.step(direction)

                steps += 1

                # 4. Track how long since the last food
                if score_delta > 0:
                    steps_without_food = 0
                else:
                    steps_without_food += 1

                # 5. End episode if snake wanders too long
                if (
                    not done
                    and steps_without_food >= MAX_STEPS_WITHOUT_FOOD
                ):
                    done = True
                    starved = True

                # 6. Calculate reward
                reward = calculate_reward(
                    score_delta,
                    done,
                    info["won"],
                )

                # 7. Observe the next state
                next_state = get_state(game)

                # 8. Store experience
                memory.push(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                # 9. Train the CNN
                if len(memory) >= batch_size:
                    batch = memory.sample(batch_size)
                    trainer.train_step(batch)

                state = next_state
                total_reward += reward

                # 10. Render selected episodes
                if window_open:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            window_open = False
                            pygame.display.quit()

                    if render_episode and window_open:
                        renderer.draw(game)
                        clock.tick(RENDER_FPS)

            # Update target network
            if (episode + 1) % target_update_frequency == 0:
                trainer.update_target()

            # Print progress
            print(
                f"Episode {episode + 1}/{TRAIN_EPISODES} | "
                f"Score: {game.score} | "
                f"Steps: {steps} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {epsilon:.2f} | "
                f"Starved: {starved}",
                flush=True,
            )

            # Save episode statistics
            history.append({
                "episode": episode + 1,
                "score": game.score,
                "steps": steps,
                "reward": total_reward,
                "epsilon": epsilon,
                "starved": starved,
            })

    finally:
        if pygame is not None:
            pygame.quit()

    # Save CSV if enabled
    if SAVE_TRAINING_CSV:
        with open("training_history.csv", "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "episode",
                    "score",
                    "steps",
                    "reward",
                    "epsilon",
                    "starved",
                ],
            )
            writer.writeheader()
            writer.writerows(history)

        print("Training history saved to training_history.csv")

    # Save trained model
    torch.save(
        trainer.policy_net.state_dict(),
        "snake_dqn.pth",
    )

    print("Model saved to snake_dqn.pth")

    # Show graph if enabled
    if SHOW_TRAINING_PLOT:
        if SAVE_TRAINING_CSV:
            subprocess.run(
                [sys.executable, "plot_training.py"],
                check=True,
            )
        else:
            print(
                "Training plot skipped: "
                "SAVE_TRAINING_CSV is disabled."
            )

    print("Training complete!")


if __name__ == "__main__":
    train()
