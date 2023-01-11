import random
from gymnasium import Env
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import relu, linear
import numpy as np

# --------------------------------------------------------------
# Adapted from:
# https://shiva-verma.medium.com/solving-lunar-lander-openaigym-reinforcement-learning-785675066197
# --------------------------------------------------------------


class DQN:
    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space: int, state_space: int, config: dict) -> None:
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = config['epsilon']
        self.gamma = config['gamma']
        self.batch_size = 64
        self.epsilon_min = config['epsilon_min']
        self.learning_rate = config['learning_rate']
        self.epsilon_decay = config['epsilon_decay']
        self.memory = deque(maxlen=config['memory'])
        self.model = self.build_model()

    def build_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, terminated, truncated) -> None:
        self.memory.append((state, action, reward, next_state, terminated, truncated))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, model_id: str) -> None:
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        # Model save? with model_id
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(env: Env, episode: int, config: dict, model_id: str) -> list:
    _loss = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0], config=config)
    for e in range(episode):
        state, _ = env.reset(seed=0)
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 3000
        for i in range(max_steps):
            action = agent.act(state)
            env.render()
            next_state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.remember(state, action, reward, next_state, terminated, truncated)
            state = next_state
            agent.replay(model_id=model_id)
            if terminated or truncated:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        _loss.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(_loss[-100:])
        if is_solved > 200:
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return _loss
