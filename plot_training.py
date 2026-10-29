import csv
import matplotlib.pyplot as plt


episodes = []
scores = []
rewards = []

with open("training_history.csv", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        episodes.append(int(row["episode"]))
        scores.append(int(row["score"]))
        rewards.append(float(row["reward"]))

plt.plot(episodes, scores, label="Score")
plt.plot(episodes, rewards, label="Reward")

plt.xlabel("Episode")
plt.ylabel("Value")
plt.title("Snake AI Training Progress")
plt.legend()
plt.grid(True)

plt.show()