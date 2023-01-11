import datetime
import gymnasium as gym
import glob
import random
import numpy as np
import os
from keras import Sequential, callbacks
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import relu, linear
import tensorflow as tf


# --------------------------------------------------------------
# https://shiva-verma.medium.com/solving-lunar-lander-openaigym-reinforcement-learning-785675066197
# --------------------------------------------------------------


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space, model_save_path, result_save_path, video_save_path):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .99
        self.batch_size = 64
        self.epsilon_min = .01
        self.lr = 0.001
        self.epsilon_decay = .996
        self.memory = deque(maxlen=1000000)
        self.iteration = 0
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.video_save_path = video_save_path
        self.model = self.load_model()

    def load_model(self):
        model = self.build_model()
        modelfiles = glob.glob("%s/model0*.h5" % self.model_save_path)
        modelfiles.sort()
        if len(modelfiles) >= 1:
            model = tf.keras.models.load_model(modelfiles[-1])
            self.iteration = int(modelfiles[-1].split("/")[3].replace(".h5","").replace("model","")) + 1
            print(f"Model loaded from previous state at episode {self.iteration}.")
        return model

    def build_model(self):

        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def remember(self, state, action, reward, next_state, terminated, truncated):
        self.memory.append((state, action, reward, next_state, terminated, truncated))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

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

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets
    
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(env, args):

    loss = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0], args.model_save_path, args.result_save_path, args.video_save_path)
    for e in range(agent.iteration, args.n_episodes):
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
            agent.replay()
            if terminated or truncated:
                print("episode: {}/{}, score: {}".format(e+1, args.n_episodes, score))
                break
        loss.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        if is_solved > 200:
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))

        # Checkpoint for models
        if e+1 % 50 == 0:
            agent.model.save(args.model_save_path + "/model%09d" % e+'.h5')
            print(f"Saved model at episode {e+1}.")

    return loss