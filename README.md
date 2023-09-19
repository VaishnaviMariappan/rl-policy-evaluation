# POLICY EVALUATION

## AIM:
To develop a Python program to evaluate the given policy.

## PROBLEM STATEMENT:
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States
The environment has 7 states:

* Two Terminal States: G: The goal state & H: A hole state.
* Five Transition states / Non-terminal States including S: The starting state.

### Actions
The agent can take two actions:

* R: Move right.
* L: Move left.

### Transition Probabilities
The transition probabilities for each action are as follows:

* 50% chance that the agent moves in the intended direction.
* 33.33% chance that the agent stays in its current state.
* 16.66% chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
![image](https://github.com/Aashima02/rl-policy-evaluation/assets/93427086/ee9c6dcf-b579-4b1c-9663-47c8b17a08b4)

## POLICY EVALUATION FUNCTION

### Formula
![image](https://github.com/Aashima02/rl-policy-evaluation/assets/93427086/574fb688-7c9f-409f-b07f-e75441d8f4b3)

## Program:
### Policy Evaluation:
```python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V = np.zeros(len(P))
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s] += prob * (reward + gamma *  prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V - V)) < theta:
        break
      prev_V = V.copy()
    return V

## Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)

## Code to evaluate the second policy
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)

## Comparing policies based on state value function
### The state value function of the second policy V2 is greater than that of the first policy V1, so we conclude that the second policy is the best policy.

V1
print_state_value_function(V1, P, n_cols=7, prec=5)
V2
print_state_value_function(V2, P, n_cols=7, prec=5)
V1>=V2
if(np.sum(V1>=V2)==7):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```

## OUTPUT:
### Policy 1:
![image](https://github.com/VaishnaviMariappan/rl-policy-evaluation/assets/94169913/16549213-2e9f-44c9-a6b2-2142fbfbd681)
![image](https://github.com/VaishnaviMariappan/rl-policy-evaluation/assets/94169913/22e65e1b-34dd-447d-8da1-b76211e7c4f9)
![image](https://github.com/VaishnaviMariappan/rl-policy-evaluation/assets/94169913/35244019-cc1d-49db-be3c-7fbe60e68ada)




### Policy 2:
![image](https://github.com/VaishnaviMariappan/rl-policy-evaluation/assets/94169913/eb49e578-118d-4b15-99bf-7fd1161c468c)
![image](https://github.com/VaishnaviMariappan/rl-policy-evaluation/assets/94169913/55cb5655-ccf1-4450-a0fc-127c49f4c396)
![image](https://github.com/VaishnaviMariappan/rl-policy-evaluation/assets/94169913/90ec3b30-40f8-4b1a-8c4a-279f69a31118)


### Comparison:
![image](https://github.com/VaishnaviMariappan/rl-policy-evaluation/assets/94169913/1d2328ff-b161-4d1a-bd3e-1334efeddbc3)



### Conclusion:
![image](https://github.com/VaishnaviMariappan/rl-policy-evaluation/assets/94169913/a574f228-aa94-4fd7-960a-7242849d729b)



## RESULT:
Thus, a Python program is developed to evaluate the given policy.
