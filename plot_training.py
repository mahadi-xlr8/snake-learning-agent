import csv
import sys

import matplotlib.pyplot as plt


# Smooth the noisy per-episode scores over this many episodes
WINDOW = 100


def moving_average(values, window):
    averages = []
    total = 0.0

    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        averages.append(total / min(i + 1, window))

    return averages


def load_history(path):
    episodes = []
    scores = []
    eval_episodes = []
    eval_scores = []

    with open(path, newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            episodes.append(int(row["episode"]))
            scores.append(int(row["score"]))

            if row.get("eval_average"):
                eval_episodes.append(int(row["episode"]))
                eval_scores.append(float(row["eval_average"]))

    return episodes, scores, eval_episodes, eval_scores


def main(paths):
    for path in paths:
        episodes, scores, eval_episodes, eval_scores = load_history(path)
        label = path.removeprefix("training_history_").removesuffix(".csv")

        line, = plt.plot(
            episodes,
            moving_average(scores, WINDOW),
            label=f"{label}: training score ({WINDOW}-episode average)",
        )

        if eval_scores:
            plt.plot(
                eval_episodes,
                eval_scores,
                "o--",
                color=line.get_color(),
                label=f"{label}: evaluation score (no random moves)",
            )

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Snake AI Training Progress")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:] or ["training_history_mlp.csv"])
