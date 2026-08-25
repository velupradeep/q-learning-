# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.

## Q LEARNING ALGORITHM
### Step 1:
Initialize Q-table and hyperparameters.

### Step 2:
Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

### Step 3:
After training, derive the optimal policy from the Q-table.

### Step 4:
Implement the Monte Carlo method to estimate state values.

### Step 5:
Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION

```py
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n

    pi_track = []

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # epsilon-greedy action selection
    select_action = lambda state, Q, epsilon: (
        np.argmax(Q[state])
        if np.random.random() > epsilon
        else np.random.randint(len(Q[state]))
    )

    # decay schedules
    alphas = decay_schedule(
        init_alpha,
        min_alpha,
        alpha_decay_ratio,
        n_episodes
    )

    epsilons = decay_schedule(
        init_epsilon,
        min_epsilon,
        epsilon_decay_ratio,
        n_episodes
    )

    # training loop
    for e in tqdm(range(n_episodes), leave=False):

        state, done = env.reset(), False

        while not done:

            # choose action
            action = select_action(state, Q, epsilons[e])

            # take action
            next_state, reward, done, _ = env.step(action)

            # TD target
            td_target = reward + gamma * Q[next_state].max() * (not done)

            # TD error
            td_error = td_target - Q[state][action]

            # update Q-table
            Q[state][action] = Q[state][action] + alphas[e] * td_error

            # move to next state
            state = next_state

        # store tracking info
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    # final value function
    V = np.max(Q, axis=1)

    # final policy
    pi = lambda s: {
        s: a for s, a in enumerate(np.argmax(Q, axis=1))
    }[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
<img width="933" height="712" alt="image" src="https://github.com/user-attachments/assets/779270c7-eaa0-408f-9aaa-f234cf95e28c" /> <br>
<img width="846" height="130" alt="image" src="https://github.com/user-attachments/assets/e0d403fa-917a-481b-947d-eca8ddf9f44b" /><br>
<img width="929" height="688" alt="image" src="https://github.com/user-attachments/assets/42c42164-c1d3-4593-84c4-71e15dd81bed" /><br>
<img width="857" height="117" alt="image" src="https://github.com/user-attachments/assets/61934c1c-8316-49c2-bb51-9457b228854a" /><br>
<img width="925" height="677" alt="image" src="https://github.com/user-attachments/assets/f7303c90-4b1c-4d14-9dec-c7e5f23dd0f4" /><br>
<img width="927" height="421" alt="image" src="https://github.com/user-attachments/assets/4d4f1ba3-c6c1-4973-af04-539f756cdeba" /><br>
<img width="919" height="413" alt="image" src="https://github.com/user-attachments/assets/ff46cdae-43e9-4ea7-b8e4-23c383897d1d" /><br>


## RESULT:

Therefore a python program has been successfully developed to find the optimal policy for the given RL environment using Q-Learning and compared the state values with the Monte Carlo method.
