
import os

from dotenv import load_dotenv


load_dotenv()


def env_bool(name, default=False):
    value = os.getenv(name, str(default))
    return value.strip().lower() in ("true", "1", "yes", "on")


# Which network to train and evaluate: "mlp" or "cnn"
MODEL_TYPE = os.getenv("MODEL_TYPE", "mlp").strip().lower()

TRAIN_EPISODES = int(os.getenv("TRAIN_EPISODES", "500"))

SHOW_GAME_WINDOW = env_bool("SHOW_GAME_WINDOW")
SHOW_TRAINING_PLOT = env_bool("SHOW_TRAINING_PLOT")
SAVE_TRAINING_CSV = env_bool("SAVE_TRAINING_CSV", True)

RENDER_FPS = int(os.getenv("RENDER_FPS", "15"))
RENDER_EVERY_N_EPISODES = max(
    1,
    int(os.getenv("RENDER_EVERY_N_EPISODES", "1")),
)
