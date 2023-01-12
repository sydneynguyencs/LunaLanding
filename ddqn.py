import glob
import random
from gymnasium import Env
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import relu, linear
import numpy as np
import tensorflow as tf


# --------------------------------------------------------------
# Adapted from:
# https://github.com/anh-nn01/Lunar-Lander-Double-Deep-Q-Networks/blob/master/Code%20source/Lunar_Lander_v2.py
# --------------------------------------------------------------


class DDQN:
    """ Implementation of double deep q learning algorithm """

    def __init__(self, args, action_space: int, state_space: int, params: dict) -> None:
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = params['epsilon']
        self.gamma = params['gamma']
        self.batch_size = 32
        self.epsilon_min = .01
        self.learning_rate = params['learning_rate']
        self.epsilon_decay = .996
        self.memory = deque(maxlen=2000)
        self.iteration = 0
        self.model_save_path = args.model_save_path
        self.result_save_path = args.result_save_path
        self.model = self.load_model()
        self.model_target = self.load_model()  # Second (target) neural network
        self.update_target_from_model()  # Update weights

    def load_model(self):
        model = self.build_model()
        modelfiles = glob.glob("%s/model0*.h5" % self.model_save_path)
        modelfiles.sort()
        if len(modelfiles) >= 1:
            model = tf.keras.models.load_model(modelfiles[-1])
            self.iteration = int(modelfiles[-1].split("/")[-1].replace(".h5", "").replace("model", ""))
            print(f"Model loaded from previous state at episode {self.iteration}.")
        return model

    def build_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_from_model(self):
        # Update the target model from the base model
        self.model_target.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        # take a mini-batch from replay experience
        cur_batch_size = min(len(self.memory), self.batch_size)
        mini_batch = random.sample(self.memory, cur_batch_size)

        # batch data
        sample_states = np.ndarray(shape=(cur_batch_size, self.state_space))
        sample_actions = np.ndarray(shape=(cur_batch_size, 1))
        sample_rewards = np.ndarray(shape=(cur_batch_size, 1))
        sample_next_states = np.ndarray(shape=(cur_batch_size, self.state_space))
        sample_dones = np.ndarray(shape=(cur_batch_size, 1))

        temp = 0
        for exp in mini_batch:
            sample_states[temp] = exp[0]
            sample_actions[temp] = exp[1]
            sample_rewards[temp] = exp[2]
            sample_next_states[temp] = exp[3]
            sample_dones[temp] = exp[4]
            temp += 1

        sample_qhat_next = self.model_target.predict(sample_next_states, verbose=0)

        # set all Q values terminal states to 0
        sample_qhat_next = sample_qhat_next * (np.ones(shape=sample_dones.shape) - sample_dones)
        # choose max action for each state
        sample_qhat_next = np.max(sample_qhat_next, axis=1)

        sample_qhat = self.model.predict(sample_states, verbose=0)

        for i in range(cur_batch_size):
            a = sample_actions[i, 0]
            sample_qhat[i, int(a)] = sample_rewards[i] + self.gamma * sample_qhat_next[i]

        q_target = sample_qhat

        self.model.fit(sample_states, q_target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_ddqn(args, env: Env, params: dict) -> (list, int):
    _loss = []
    is_solved = 0
    agent = DDQN(args, action_space=env.action_space.n, state_space=env.observation_space.shape[0], params=params)
    for e in range(agent.iteration, args.n_episodes):
        state, _ = env.reset(seed=42)
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 3000
        for i in range(max_steps):
            action = agent.act(state)
            env.render()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = False
            if terminated or truncated:
                done = True
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, args.n_episodes, score))
                break
        _loss.append(score)
        agent.update_target_from_model()  # Update the weights after each episode

        # Write scores into file
        scorefile = open(args.result_save_path + "/scores.txt", "a+")
        scorefile.write(f"Episode: {e}, Score: {score} \n")
        scorefile.flush()
        scorefile.close()

        # Average score of last 100 episode
        is_solved = np.mean(_loss[-100:])
        
        # Log every 20 episodes
        if e % 20 == 0:
            print("episode: {}/{}, score: {}".format(e + 1, args.n_episodes, score))   
            print("Average over last 100 episode: {0:.2f} \n".format(is_solved)) 

        # Checkpoint for models
        if e % 50 == 0 or e == args.n_episodes - 1:
            agent.model.save(args.model_save_path + "/model%09d" % e + '.h5')
            print(f"Saved model at episode {e}.")
        
        if is_solved > 200:
            agent.model.save(args.model_save_path + "/model%09d" % e + '.h5')
            print(f"Saved model at episode {e}.")
            print('\n Task Completed! \n')
            break

    return _loss, is_solved
