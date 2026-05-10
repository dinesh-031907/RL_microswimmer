# Reinforcement Learning for Optimal Microswimmer Gaits

In this project I studied the use of **Reinforcement Learning (RL)** to discover optimal swimming strategies for microswimmers operating in the **low Reynolds number regime**. The swimmer dynamics are modeled using the **Oseen tensor approximation**, while the optimal actuation policies are learned using **Q-learning**.

The main objective is to learn non-reciprocal stroke cycles that maximize long-time net displacement of an \(N\)-sphere microswimmer.

---

# Physics Background

At low Reynolds numbers, inertia becomes negligible and viscous forces dominate the dynamics. In this regime, reciprocal motion cannot generate net propulsion due to the **Scallop Theorem**. Efficient swimming therefore requires non-reciprocal motion.

This project models a generalized \(N\)-sphere swimmer connected through \(N-1\) extensible arms. Each arm can exist in either:

- Contracted state
- Extended state

---

# Features

- Generalized \(N\)-sphere swimmer framework
- Oseen tensor hydrodynamic interactions
- Automatic generation of:
  - States
  - Actions
  - Transition matrices
  - Reward matrices
- Q-learning to find optimal swimming cycles
- Scalable to any total sphere sizes.

---

# State Space

For an \(N\)-sphere swimmer:

- Number of arms = `N-1`

- Total states   = `2 ** (N-1)`
  
Each state represents a unique arm configuration.

Example for \(N=3\):

```python
(10,10)
(10,6)
(6,10)
(6,6)
```

---

# Actions

For an `N`-sphere swimmer:

- Number of arms = `N-1`

Therefore, the number of possible actions available at each state is:

```python
N - 1
```

However, the total number of possible actions is:

```python
2 * (N - 1)
```

The reason is that:
- arms in the contracted state cannot be further contracted
- arms in the extended state cannot be further extended
---
# Q-Matrix Representation

The swimmer policy is learned using a Q-table representation.

For an `N`-sphere swimmer:

- Number of states = `2^(N-1)`
- Number of actions = `2*(N-1)`

Hence, the Q-matrix dimensions are:

```python
Q.shape = (2^(N-1), 2*(N-1))
```

Each entry:

```python
Q(s, a)
```

represents the expected long-time displacement obtained by taking action `a` in state `s`.

Invalid actions are masked during learning.

An epsilon-greedy Q-learning algorithm is used to optimize swimmer propulsion.

The update rule is:

```python
Q(s,a) = Q(s,a) + alpha * (
    r + gamma * max(Q(s',a')) - Q(s,a)
)
```

where:

- `alpha` = learning rate
- `gamma` = discount factor
- `r` = displacement reward
- `s'` = next state

The learned policy corresponds policy that maximizes the long-time average displacement per stroke.

--- 

# Physical Interpretation

The swimmer learns non-reciprocal cycles, allowing them to propel in the low Reynolds number regime.

This work was based on the following paper:

- `Tsang, E. C., & Tong, P. (2020). Self-learning how to swim at low Reynolds number. Physical Review Fluids, 5(7), 074101. doi:10.1103/PhysRevFluids.5.074101`
- `Najafi, A., & Golestanian, R. (2004). Simple swimmer at low Reynolds number: Three linked spheres. Physical Review E, 69(6), 062901. doi.org`



---

# NN based Actor Critic model

- I also solved the same problem using Actor Critic RL algorithm.

- Results are similar for 3 sphere case.

- As before, this can also extended to multiple sphere case.
