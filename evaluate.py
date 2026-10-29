
import argparse
import statistics

from game.snake_game import SnakeGame

from agent.evaluation import play_greedy_episode
from agent.model import MODELS, load_model

from config import MODEL_TYPE, RENDER_FPS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch or score a trained Snake agent"
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default=MODEL_TYPE,
        help="which network to load",
    )
    parser.add_argument(
        "--checkpoint",
        help="path to the .pth file (default: snake_dqn_<model>.pth)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="number of games to play",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="show the game window",
    )
    return parser.parse_args()


def evaluate(model_type, checkpoint, episodes, render):
    checkpoint = checkpoint or f"snake_dqn_{model_type}.pth"
    model = load_model(model_type, checkpoint)
    game = SnakeGame()

    on_step = None
    pygame = None
    window = {"closed": False}

    if render:
        import pygame as pygame_module
        from game.renderer import Renderer

        pygame = pygame_module
        renderer = Renderer(game.size)
        clock = pygame.time.Clock()

        def on_step(current_game):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    window["closed"] = True
                    return False

            renderer.draw(current_game)
            clock.tick(RENDER_FPS)
            return True

    print(f"Evaluating {checkpoint} ({model_type.upper()}) for {episodes} games")

    scores = []

    try:
        for episode in range(episodes):
            score = play_greedy_episode(model, model_type, game, on_step)
            scores.append(score)
            print(f"Game {episode + 1}/{episodes} | Score: {score}", flush=True)

            if window["closed"]:
                break
    finally:
        if pygame is not None:
            pygame.quit()

    print(
        f"Average: {statistics.mean(scores):.1f} | "
        f"Median: {statistics.median(scores)} | "
        f"Max: {max(scores)}"
    )


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.checkpoint, args.episodes, args.render)
