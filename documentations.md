# Snake Learning Agent: Documentation from the Ground Up

This guide explains every idea used in this project, starting from zero.
You don't need to know reinforcement learning or neural networks before reading it.
Every concept is tied to the real code, so you can open the file and see it in action.

**Table of contents**

1. [What is an AI agent?](#1-what-is-an-ai-agent)
2. [Reinforcement learning basics](#2-reinforcement-learning-basics)
3. [Q-learning: putting a number on every choice](#3-q-learning-putting-a-number-on-every-choice)
4. [Deep Q-Networks (DQN) and the tricks that make them work](#4-deep-q-networks-dqn-and-the-tricks-that-make-them-work)
5. [Exploration vs. exploitation](#5-exploration-vs-exploitation)
6. [Reward design](#6-reward-design)
7. [How the snake "sees": MLP vs. CNN](#7-how-the-snake-sees-mlp-vs-cnn)
8. [Walkthrough of one training step](#8-walkthrough-of-one-training-step)
9. [Case study: why the first version scored 0](#9-case-study-why-the-first-version-scored-0)
10. [How to run, evaluate, and read the results](#10-how-to-run-evaluate-and-read-the-results)
11. [Why "winning" is hard, and where to go next](#11-why-winning-is-hard-and-where-to-go-next)
12. [Glossary](#12-glossary)

---

## 1. What is an AI agent?

An **agent** is anything that:

1. **observes** the world,
2. **decides** what to do, and
3. **acts**, which changes the world.

Then the loop repeats. The "world" is called the **environment**.

```
            ┌───────────── action ─────────────┐
            │                                  ▼
     ┌─────────────┐                    ┌──────────────┐
     │    AGENT    │                    │ ENVIRONMENT  │
     │ (the brain) │                    │ (Snake game) │
     └─────────────┘                    └──────────────┘
            ▲                                  │
            └──── new state + reward ──────────┘
```

In this project:

| Concept | In Snake | In the code |
|---|---|---|
| Environment | The 20×20 board with walls, obstacles, food, and the snake | `game/snake_game.py` → `SnakeGame` |
| State | What the agent looks at before it decides | `agent/state.py` → `get_state` |
| Action | Go straight, turn right, or turn left | `agent/actions.py` |
| Reward | A number that says "that was good" or "that was bad" | `agent/rewards.py` |
| Policy | The rule for picking an action from a state | The neural network in `agent/model.py` |

### Why relative actions?

The snake has only **3** actions instead of 4 (up, down, left, right):

```
STRAIGHT = 0      keep going the way you face
TURN_RIGHT = 1    turn 90° clockwise
TURN_LEFT = 2     turn 90° counter-clockwise
```

There are two reasons:

- Reversing into your own neck is always death, so we remove that option completely.
- "Turn right" means the same thing for the snake no matter which way it is facing.
  That makes learning easier, **as long as the state is also described relative to the snake**.
  This detail turned out to be the main bug in the first version (see [section 9](#9-case-study-why-the-first-version-scored-0)).

`action_to_direction` in `agent/actions.py:10` converts a relative action into a real screen direction.

---

## 2. Reinforcement learning basics

**Reinforcement learning (RL)** means learning by trial and error.
Nobody tells the agent the right move. It tries things, gets rewards, and slowly figures out which behavior leads to more reward.

Compare this with *supervised learning*, where you have labeled examples ("this picture is a cat").
In RL there are no labels, only rewards, and they often arrive **late**:
the move that led to eating food might have happened 15 steps earlier.

### Key words

- **Step**: one move of the snake.
- **Episode**: one full game, from reset until death (or until a time limit).
- **Return**: the total reward collected from now until the end of the episode.

### Discounting: why the future counts a little less

We want the agent to care about future rewards, but rewards that are closer should count more.
To do that, we multiply each future reward by a **discount factor** γ (gamma), once for every step it lies in the future:

```
Return = r₀ + γ·r₁ + γ²·r₂ + γ³·r₃ + ...
```

This project uses `gamma = 0.99` (`agent/learning.py`).

Example: food (+10) is 5 steps away and each step costs −0.01.

```
Return ≈ −0.01 − 0.0099 − 0.0098 − 0.0097 − 0.0096 + 0.99⁵ · 10
       ≈ −0.05 + 9.51
       ≈ 9.46
```

With γ close to 1, the agent is "far-sighted": food 50 steps away still matters (0.99⁵⁰ ≈ 0.6).
With γ = 0.5, food just 10 steps away would be worth almost nothing (0.5¹⁰ ≈ 0.001).

---

## 3. Q-learning: putting a number on every choice

### What is a Q-value?

**Q(state, action)** answers this question:
*"If I am in this state and I take this action, then play well afterwards, how much total (discounted) reward will I get?"*

If we knew Q exactly, playing perfectly would be easy: **pick the action with the highest Q**.

```
State: food is to the right, a wall is straight ahead

  Q(state, STRAIGHT)   = −9.8   ← crash into the wall
  Q(state, TURN_RIGHT) = +8.1   ← toward the food
  Q(state, TURN_LEFT)  = +2.3   ← safe, but away from the food

  Best action = TURN_RIGHT
```

### The Bellman equation: how Q is learned

We don't know Q, but we know a rule that the true Q must follow:

```
Q(s, a) = r + γ · max Q(s', a')
              a'
```

In words: *the value of an action = the reward you get right now + the (discounted) value of the best action in the next state.*

So we can **improve our guess** after each step:

1. We were in state `s`, took action `a`, got reward `r`, and landed in `s'`.
2. The **target** is `r + γ · max Q(s', ·)`.
3. We move our guess `Q(s, a)` a little toward that target.

### Worked example

Say the current guess is `Q(s, TURN_RIGHT) = 2.0`.
We take TURN_RIGHT and get reward `r = 0.04` (we moved closer to the food).
In the new state the best Q-value is `5.0`.

```
target = 0.04 + 0.99 × 5.0 = 4.99
error  = target − guess = 4.99 − 2.0 = 2.99
```

Our guess was too low, so we nudge it up.
Repeat this millions of times and the guesses become accurate.

**Special case: the game ended.** There is no next state, so the target is just `r`.
In code, this is the `(1 - dones)` factor in `agent/learning.py`: it zeroes out the future part.

### Why not just use a table?

Classic Q-learning stores Q in a table: one row per state, one column per action.
A 20×20 Snake board has an astronomical number of possible states, so the table would never fit, and the agent would almost never visit the same state twice.

The fix is to use a **neural network** that *approximates* Q.
States that look alike get similar Q-values, so learning in one situation carries over to similar ones.
A Q-function made of a deep neural network is a **Deep Q-Network (DQN)**.

```
         state ──►  [ neural network ]  ──►  Q(straight), Q(right), Q(left)
```

---

## 4. Deep Q-Networks (DQN) and the tricks that make them work

Putting a neural network in place of the table is unstable on its own.
DQN adds a few tricks. Each one fixes a specific problem.

### 4.1 Training a neural network (a 60-second version)

- A network is a function with millions of adjustable numbers called **weights**.
- We define a **loss**: one number measuring how wrong the network's prediction was.
- **Backpropagation** works out, for every weight, which direction would reduce the loss.
- The **optimizer** (Adam in this project) nudges every weight a little in that direction.
  The size of the nudge is the **learning rate**.

In `DQNTrainer.train_step` (`agent/learning.py:36`), this is:

```python
loss = self.loss_fn(predicted, targets)   # how wrong were we?
self.optimizer.zero_grad()                # clear old gradients
loss.backward()                           # backpropagation
self.optimizer.step()                     # update the weights
```

### 4.2 Experience replay (`agent/replay_memory.py`)

**Problem:** consecutive steps are almost identical (the snake moved one cell).
Training on them in order is like studying one page of a textbook 100 times in a row: the network overfits to what just happened and forgets the rest.

**Fix:** store every experience `(state, action, reward, next_state, done)` in a big memory,
and train on a **random batch** drawn from it.

```
memory: [exp1, exp2, exp3, ..., exp100000]
                 │      │            │
                 └──────┴── random ──┴──► batch of 256 ──► one training step
```

Benefits: every experience can be learned from many times, and the batch mixes old and new situations.
This project keeps the last 100,000 experiences (`MEMORY_CAPACITY` in `train.py`).

### 4.3 Target network

**Problem:** the target `r + γ · max Q(s', ·)` uses the same network we are training.
Each update changes the target too, like chasing your own shadow.

**Fix:** keep a frozen **copy** of the network (the *target network*) to compute targets,
and copy the trained weights into it only every 1,000 steps (`TARGET_UPDATE_STEPS`).
Targets stay still long enough for learning to make progress.

### 4.4 Double DQN

**Problem:** `max` over noisy estimates is biased upward.
If all three Q-values have random errors, the maximum tends to pick whichever one is *accidentally* too high.
The agent then becomes overconfident about some actions.

**Fix (Double DQN):** split the job in two.

- The **policy network** *chooses* the best next action.
- The **target network** *scores* that action.

```python
next_actions = self.policy_net(*next_state_inputs).argmax(dim=1, keepdim=True)
next_q = self.target_net(*next_state_inputs).gather(1, next_actions).squeeze(1)
```

Both networks would have to make the same mistake for it to get through, which is much less likely.

### 4.5 Smooth L1 loss (Huber loss)

The loss compares the prediction with the target.
Squared error (`error²`) explodes for big errors: one surprising crash could cause a huge, destabilizing update.
**Smooth L1** acts like squared error for small errors and like plain absolute error for big ones, so large mistakes can't blow up training.

### 4.6 Gradient clipping

As an extra seatbelt, `clip_grad_norm_(..., 10.0)` caps how big a single update can be.

---

## 5. Exploration vs. exploitation

At the start, the network knows nothing. If it always picked its "best" action, it would repeat the same random-looking behavior and never discover that food is good.

- **Exploration**: try random actions to discover new things.
- **Exploitation**: use what you've learned to collect reward.

### Epsilon-greedy (`agent/exploration.py`)

```
with probability ε  → pick a random action        (explore)
otherwise           → pick the action with max Q  (exploit)
```

ε (epsilon) starts at 1.0 (100% random) and shrinks to 0.01 (1% random):

```
ε
1.0 ┤█
    │ ██
    │   ███
    │      ███
    │         ███
0.01┤            ████████████████████████
    └──────────────────────────────────── episode
     0       decay
```

`decay` is how many episodes it takes to get there: 300 for the MLP and 2,000 for the CNN (`MODEL_CONFIG` in `train.py`).

### Why greedy evaluation matters

During training, the printed score includes the random moves.
Even ε = 0.3 means about one move in three is random, and a random turn next to a wall is often fatal.
That makes a decent agent *look* bad.

So every 100 episodes `train.py` plays 10 games with **ε = 0** (no random moves) and prints:

```
  Evaluation (no random moves) | Average: 31.4 | Best: 52
```

**That is the real measure of skill.** The best one is saved to `snake_dqn_<model>_best.pth`.

---

## 6. Reward design

Rewards are the *only* way you tell the agent what you want.
The agent will maximize exactly what you reward, including in ways you didn't intend.

`agent/rewards.py`:

| Event | Reward | Why |
|---|---|---|
| Win (board full) | +20 | The ultimate goal |
| Death (wall, obstacle, own body) | −10 | Strongly discourage crashing |
| Eat food | +10 | The main objective |
| Normal move | −0.01 | A gentle push not to waste time |
| Move closer to food | +0.05 | **Reward shaping**: a hint |
| Move away from food | −0.05 | **Reward shaping**: a hint |

### Reward shaping

Without the closer/farther hint, a new agent moving randomly almost never finds food, so it rarely sees the +10 and learns very slowly.
The small distance bonus gives it a breadcrumb trail. It's kept small (0.05 vs. 10) so it guides the agent without becoming the goal.

### Starvation is a *time limit*, not a death

If the snake goes `100 × length` steps without eating (`agent/evaluation.py:12`), we stop the episode so it can't loop forever.

Earlier, this counted as a death: reward −10, `done = True`.
That's misleading: the snake did nothing wrong *in that exact state*.
The state doesn't contain a step counter, so the network sees the same situation sometimes end in a crash and sometimes continue normally, and it can't make sense of that.

Now starvation is stored with a normal step reward and `done = False`.
In RL this is called **truncation**: the game was *cut off*, not *lost*, so the future value still counts.

---

## 7. How the snake "sees": MLP vs. CNN

The **state** is what you feed into the network. It largely decides how fast (and whether) the agent learns.
This project offers two approaches, and you can train and compare both.

### 7.1 The MLP: hand-crafted features

**Idea:** we, the humans, pick the facts that matter and give the network just those numbers.
`agent/features.py` builds 17 numbers:

| # | Feature | Values |
|---|---|---|
| 0–2 | **Danger** if I go straight / right / left (wall, obstacle, or body) | 0 or 1 |
| 3–6 | Current heading: up / right / down / left (one-hot) | 0 or 1 |
| 7–10 | Food is ahead / behind / to my right / to my left | 0 or 1 |
| 11–13 | **Free space** reachable after going straight / right / left | 0 to 1 |
| 14 | Snake length (as a fraction of the board) | 0 to 1 |
| 15–16 | Distance to the food: forward and sideways (scaled) | −1 to 1 |

Everything food-related is **relative to the heading**, just like the actions.

```
Snake facing UP, food up and to the right:

   . . . . F .        food_ahead = 1
   . . . . . .        food_right = 1
   . . H . . .        food_behind = 0, food_left = 0
   . . B . . .
   . . B . . .
```

#### Free space (flood fill): the anti-trap feature

Danger only looks one step ahead. A long snake can turn into a pocket that is safe *now* but has no exit:

```
   # # # # # #
   # . B B B #       Turning right enters a 1-cell pocket.
   # . H ? B #       danger_right = 0 (it looks safe!)
   # . B B B #       free_space_right = 1/318 ≈ 0.003 (a trap!)
   # . . . . #
```

`_reachable_space` (`agent/features.py:44`) runs a **breadth-first search (BFS)**:
starting from the cell the snake would move into, it spreads to every connected empty cell and counts them.
This is the same idea as the "paint bucket" tool in an image editor.

#### What is an MLP?

A **multi-layer perceptron** is the simplest neural network: layers of "neurons", where every neuron is connected to every input of the layer below.

```
17 features ──► [256 neurons] ──ReLU──► [256 neurons] ──ReLU──► [3 outputs]
                                                                 Q(straight)
                                                                 Q(right)
                                                                 Q(left)
```

- Each `nn.Linear(a, b)` layer computes `output = W · input + bias`, where `W` is a `b × a` matrix of weights.
- **ReLU** (`max(0, x)`) is the *non-linearity*. Without it, stacking layers would collapse into one linear function and the network couldn't learn "if danger AND food-right then...".

See `SnakeMLP` in `agent/model.py:43`.

### 7.2 The CNN: learning from the raw board

**Idea:** give the network the whole board as an image and let it figure out what matters.
No human-designed features.

#### One-hot channels (`agent/observation.py`)

The board is a 20×20 grid of cell types. We split it into 5 **channels**, like the red, green, and blue layers of a photo:

```
channel 0: walls        channel 1: obstacles    channel 2: body
channel 3: head         channel 4: food

Each channel is a 20×20 grid of 0s and 1s.
Shape fed to the network: [5, 20, 20]
```

#### What a convolution does

A **convolution** slides a small 3×3 window over the image, and at each position computes a weighted sum.
The 3×3 weights form a **filter**, and the network *learns* what each filter should detect.
One filter might learn "wall directly above", another "body to the left".

```
   input patch          filter            output
   0 0 1                0 0 1
   0 H 1        ×       0 0 1      =    high value: "wall to the right!"
   0 0 1                0 0 1
```

Two stacked convolutions let the network detect patterns of patterns (for example "a corridor").
**Max pooling** (`nn.MaxPool2d(2)`) then shrinks 20×20 to 10×10 by keeping the largest value in each 2×2 block.
That means fewer numbers for the final layers to handle.

```
[5,20,20] ─conv─► [16,20,20] ─conv─► [32,20,20] ─pool─► [32,10,10] ─flatten─► 3200
                                                       + 4 direction values ─► 256 ─► 3 Q-values
```

See `SnakeCNN` in `agent/model.py:8`.

#### Rotating the board so the snake always faces up

This is the most important detail of the CNN.

The actions are *relative* ("turn right"). If the board were always shown in the same screen orientation, "turn right" would mean:

- facing up → go to the screen's right
- facing right → go down
- facing down → go left
- facing left → go up

So the CNN would have to learn **4 separate versions** of every situation.

So `get_cnn_state` (`agent/state.py:13`) **rotates the board** so the snake's heading always points up:

```
Before (snake heading RIGHT):        After rotation (heading points UP):

  . . . . .                            . . F . .
  . . . . .                            . . . . .
  B B H . F           ──rot90──►       . . H . .
  . . . . .                            . . B . .
  . . . . .                            . . B . .
```

Now "straight" is always *up on the image*, "right" is always *right on the image*, and the network only has to learn each situation once.

### 7.3 MLP vs. CNN: which is better?

| | MLP + features | CNN + board |
|---|---|---|
| What it sees | 17 hand-picked numbers | The whole board |
| Speed to learn | Fast (hundreds of episodes) | Slow (thousands or more) |
| Training speed per step | Very fast | Much slower |
| Who designs the "vision" | You | The network learns it |
| Ceiling | Limited by your features | Higher in theory: it can spot patterns you didn't think of |
| Lesson | Good features are powerful | This is how "deep" RL works on raw pixels (Atari, etc.) |

A good way to learn is to **train both and compare the plots** (section 10).

---

## 8. Walkthrough of one training step

Here is one iteration of the loop in `train.py` (starting at line 155), in order:

```
                 ┌────────────────────────────────────────────────┐
  state ──► 1. choose_action ──► 2. game.step ──► 3. calculate_reward
                                                        │
          7. state = next_state ◄── 6. train_step ◄── 4–5. get next_state, memory.push
```

1. **Choose an action** (`train.py:160`): epsilon-greedy. Either random, or `argmax` of the network's Q-values.
2. **Act** (`train.py:172`): convert the relative action into a direction and move the snake.
   The game reports `score_delta` (1 if food was eaten) and `done` (crashed or won).
3. **Reward** (`train.py:191`): turn what happened into a number (section 6).
4. **Observe** (`train.py:200`): build the new state `s'`.
5. **Remember** (`train.py:205`): store `(s, a, r, s', done)` in replay memory.
   Then check the starvation time limit.
6. **Learn** (`train.py:224`): once memory has 1,000+ experiences, sample a random batch and run one `train_step`.
   It computes Double DQN targets → Smooth L1 loss → backprop → Adam update.
   Every 1,000 steps the target network gets a fresh copy.
7. **Move on**: `state = next_state`, and repeat until the episode ends.

After each episode the score is printed and saved. Every 100 episodes there's a greedy evaluation.

### How both models share the same code

A state is always a **tuple of arrays**:

- CNN: `(board[5,20,20], direction[4])`
- MLP: `(features[17],)`

`states_to_tensors` (`agent/state.py:41`) turns a batch of these into tensors, and the network is called as `model(*tensors)`.
So the trainer, the exploration code, and the training loop never need to know which model they are using.

---

## 9. Case study: why the first version scored 0

After 3,000 episodes the original agent still scored 0–1 and died within about 10–170 steps.
`Starved: False` on almost every episode told us it wasn't wandering. It was **crashing**.
Here's what was wrong, and what we changed:

| Problem | Effect | Fix |
|---|---|---|
| CNN saw a fixed-orientation board, but actions were relative | It had to learn every situation 4 times; a 4-number direction hint was not enough | Rotate the board to face the heading (7.2); add the MLP (7.1) |
| ε decayed over all 3,000 episodes | At episode 2,230 it was still 30% random, so random turns kept killing it | Decay over 300 (MLP) or 2,000 (CNN) episodes, down to 0.01 |
| No greedy evaluation | Logged scores included the random moves, which hid real progress | Evaluate with ε = 0 every 100 episodes |
| Small replay memory (10k), low learning rate, target sync per episode | Slow, unstable learning | 100k memory, tuned learning rates, target sync every 1,000 steps |
| Plain DQN, no clipping | Overconfident Q-values, occasional huge updates | Double DQN + gradient clipping |
| Huge CNN dense layer (12,804 → 128) | Lots of weights to learn from little data | Max pooling → 3,204 → 256 |
| Starvation = death (−10, terminal) | Confusing targets | Treat it as truncation (`done = False`) |
| Fixed 200-step starvation limit | A long snake needs more room to maneuver | `100 × length` |

**Lesson:** in RL, a "not learning" problem is rarely a single bug.
Check, in this order: (1) can the state actually express what's needed? (2) is exploration sensible? (3) are you measuring the right thing? (4) are the learning settings stable?

---

## 10. How to run, evaluate, and read the results

### Train

```bash
python train.py --model mlp --episodes 500     # fast: a few minutes
python train.py --model cnn --episodes 3000    # slow: can take hours on CPU
```

You can also set `MODEL_TYPE=mlp` or `MODEL_TYPE=cnn` and `TRAIN_EPISODES` in `.env`.
Use `SHOW_GAME_WINDOW=true` to watch training. It's much slower, so try it with `RENDER_EVERY_N_EPISODES=50`.

Each model writes its own files, so they don't overwrite each other:

| File | What it is |
|---|---|
| `snake_dqn_<model>.pth` | Weights at the end of training |
| `snake_dqn_<model>_best.pth` | Weights with the best greedy evaluation |
| `training_history_<model>.csv` | Per-episode score, steps, reward, epsilon, and evaluation |

### Evaluate (check a trained model)

```bash
python evaluate.py --model mlp --episodes 50
python evaluate.py --model mlp --checkpoint snake_dqn_mlp_best.pth --render
python evaluate.py --model cnn --render
```

This plays with no random moves and prints the average, median, and max score.

### Plot

```bash
python plot_training.py training_history_mlp.csv
python plot_training.py training_history_mlp.csv training_history_cnn.csv   # compare both
```

The **solid line** is the training score, smoothed over 100 episodes (it includes random moves).
The **dots** are greedy evaluation scores, which show real skill.

### What to expect

- **MLP**: in a test run of 500 episodes, the greedy evaluation averages were 6 → 42 → 43 → 72 → 74 (at episodes 100–500), with single games reaching about 90–97.
  Evaluating the best checkpoint over 30 games gave an average of 71, a median of 74, and a max of 90.
- **CNN**: stays near 0 for a long time while ε is high, then slowly rises. It needs many more episodes than the MLP.
- Results vary between runs, because RL is noisy. Some runs are just luckier.

### Experiments to try (one at a time)

| Change | Where | What you'll learn |
|---|---|---|
| Remove the free-space features | `agent/features.py` | How much self-trapping matters |
| Set the distance shaping to 0 | `agent/rewards.py` | How much shaping speeds up learning |
| `gamma = 0.9` vs. `0.99` | `agent/learning.py` | Short-sighted vs. far-sighted behavior |
| A shorter or longer `epsilon_decay` | `MODEL_CONFIG` in `train.py` | Too little vs. too much exploration |
| Turn off the CNN rotation (`k = 0`) | `agent/state.py` | How much the orientation fix matters |
| `TARGET_UPDATE_STEPS = 1` | `train.py` | Why a target network exists |

---

## 11. Why "winning" is hard, and where to go next

Winning means filling **all 318 free cells**.
Near the end the snake is so long that a single wrong turn hundreds of steps earlier can make the game impossible to finish.
A plain DQN with one-step-ahead features almost never gets there.
This project's MLP reaches about 70–95 (roughly a quarter of the board). Past that, the snake usually traps itself.

Ideas to push further, roughly from easiest to hardest:

1. **Better features**: the direction toward the tail, whether the tail is reachable from each move, or a longer lookahead.
2. **More training and tuning**: larger networks, a learning-rate schedule, prioritized replay (replay surprising experiences more often).
3. **Curriculum learning**: start on a small board (say 8×8) and grow it as the agent improves.
4. **Better algorithms**: PPO (a policy-gradient method) usually beats DQN on Snake with enough compute.
5. **Hybrid AI + algorithm**: let the network chase food, but only allow moves where a safe path back to the tail still exists.
   A guaranteed-win strategy is a **Hamiltonian cycle**, a path that visits every cell exactly once and loops.
   Following it never fails; the AI's job is to take shortcuts safely.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Agent** | The decision maker (our neural network + action-selection rule) |
| **Environment** | The world the agent acts in (the Snake game) |
| **State** | What the agent observes before deciding |
| **Action** | A choice the agent can make (straight, right, left) |
| **Reward** | A number given after each action: good or bad |
| **Episode** | One full game |
| **Return** | The sum of (discounted) rewards from now to the end |
| **γ (gamma)** | Discount factor: how much the future matters (0.99 here) |
| **Policy** | The rule that maps states to actions |
| **Q-value** | The expected return for taking an action in a state |
| **Bellman equation** | Q(s,a) = r + γ·max Q(s′,·): the rule used to learn Q |
| **DQN** | Deep Q-Network: a neural network that approximates Q |
| **Replay memory** | A storage of past experiences, sampled randomly for training |
| **Target network** | A frozen copy of the network used to compute stable targets |
| **Double DQN** | The policy net picks the next action, the target net scores it; this reduces overestimation |
| **ε-greedy** | Act randomly with probability ε, otherwise pick the best action |
| **Greedy evaluation** | Testing with ε = 0 to measure real skill |
| **Reward shaping** | Extra small rewards that guide learning (distance to food) |
| **Truncation** | An episode cut off by a time limit (not a real ending) |
| **MLP** | Multi-layer perceptron: stacked fully connected layers |
| **CNN** | Convolutional neural network: learns spatial patterns with sliding filters |
| **One-hot** | Encoding a category as a vector with a single 1 (for example, heading = [0,1,0,0]) |
| **ReLU** | max(0, x), the non-linearity between layers |
| **Loss** | How wrong a prediction is; training minimizes it |
| **Learning rate** | How big each weight update is |
| **Batch** | A group of experiences trained on together |
| **Flood fill / BFS** | Counting all cells reachable from a starting cell |
| **Hyperparameter** | A setting you choose (learning rate, γ, ε decay), as opposed to a weight the network learns |
