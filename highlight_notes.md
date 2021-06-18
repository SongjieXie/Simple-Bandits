# Bandit Algorithms

Online decision-making involves a chioce of **exploration-expliotation** and bandits model have widely applications likes: ...

The goal is: maximize expected cumulative reward.

The reward model:

1. **IID reward:**
2. **Adversarial reward:**
3. **Constrained adversarial reward:**
4. **Stochastic reward(beyond IID):** random process, e.g., a random walk.

## Stochastic bandits

In each round, the agent can only observe the reward of chosen action and the rewards are bounded. Every time an action $a$ is chosen, the reward is sampled independently from the reward distribution associated with the action $a$.

A multi-armed bandit is the tuple $<\mathcal{A}, \mathcal{R}>$ where $\mathcal{A}$ is the set of actions(arms) and the $\mathcal{R}$ is the unknown reward distribution, $\mathcal{R}^a(r)= \mathbb{P}[r|a]$. At each round $t$, an agent select the action $a_t \in \mathcal{R}$, receive a reward $r_t \sim \mathcal{R}^a(t)$

## Greedy algorithm

## UCB algorithm

## Thompson sampling

## Gradient bandit

## Gaussian process regression

## Gaussian processes bandit

### GP-UCB

### GP-TS
