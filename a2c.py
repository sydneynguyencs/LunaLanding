import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reinforce import Reinforce
from gymnasium import Env
import glob

NUM_ACTIONS = 4
STATE_DIM = 8

class A2C(Reinforce):
    def __init__(self, args, params: dict):        
        self.iteration = 0
        self.gamme = params['gamma']
        self.n = params['n']
        self.learning_rate = params['learning_rate']
        self.critic_learning_rate = params['critic_learning_rate']
        self.model_save_path = args.model_save_path
        self.critic_model_save_path = args.model_save_path.replace("model","critic_model")
        self.result_save_path = args.result_save_path
        self.video_save_path = args.video_save_path
        self.model = self.load_model()
        self.critic_model = self.load_model('critic')

        
    def load_model(self, mode:str='model'):
        if mode == 'model':
            model = self.build_model()
            modelfiles = glob.glob("%s/model0*.h5" % self.model_save_path)
        if mode == 'critic':
            model = self.build_critic_model()
            modelfiles = glob.glob("%s/model0*.h5" % self.critic_model_save_path)

        modelfiles.sort()
        if len(modelfiles) >= 1:
            model = tf.keras.models.load_model(modelfiles[-1])
            self.iteration = int(modelfiles[-1].split("/")[-1].replace(".h5", "").replace("model", ""))
            print(f"Model loaded from previous state at episode {self.iteration}.")
        return model

    # TODO: Define any training operations and optimizers here, initialize
    #       your variables, or alternately compile your model here.
    
    def build_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(16, input_dim=STATE_DIM, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))
        
        actor_optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        self.model.compile(loss=get_actor_loss(), optimizer=actor_optimizer)
        model = self.model
        return model
    
    def build_critic_model(self) -> Sequential:
        critic_model = Sequential()
        critic_model.add(Dense(32, input_dim=STATE_DIM, kernel_initializer='VarianceScaling', activation='relu', use_bias=True))
        critic_model.add(Dense(64, kernel_initializer='VarianceScaling', activation='relu', use_bias=True))
        critic_model.add(Dense(64, kernel_initializer='VarianceScaling', activation='relu', use_bias=True))
        critic_model.add(Dense(32, kernel_initializer='VarianceScaling', activation='relu', use_bias=True))
        critic_model.add(Dense(1, kernel_initializer='VarianceScaling', use_bias=True))
        critic_optimizer = keras.optimizers.Adam(lr=critic_learning_rate)
        self.critic_model.compile(loss=keras.losses.mean_squared_error, optimizer=critic_optimizer)
        model = self.critic_model
        return model

    def multinomial_sample(self, policy):
        num_classes = policy.shape[0]
        thresholds = np.zeros(num_classes + 1)
        for i in range(num_classes):
            thresholds[i+1] = thresholds[i] + policy[i]
        rand = np.random.uniform()
        for i in range(num_classes):
            if rand >= thresholds[i] and rand <= thresholds[i+1]:
                return i
        return num_classes - 1
    
    def generate_episode(self, env, render=False):
        states = []
        actions = []
        rewards = []
        state = env.reset()
        done = False
        total_rewards = 0
        while not done:
            if render:
                env.render()
            policy = self.model.predict(state.reshape((1, STATE_DIM))).reshape(NUM_ACTIONS)
            action = self.multinomial_sample(policy)
            observation, reward, done, _ = env.step(action)
            total_rewards += reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = observation

        return states, actions, rewards

    def train_step(self, env, update_actor=False):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, actions, rewards = self.generate_episode(env)
        reward = [0.01 * i for i in rewards]
        R = np.zeros((len(reward), 1))
        state = np.zeros((len(reward), STATE_DIM))
        action = np.zeros(len(reward))
        V = self.critic_model.predict(state)
        # print(np.max(V))
        for i in range(R.shape[0]):
            multi = 1
            state[i] = states[i]
            action[i] = actions[i]
            for j in range(self.n):
                r = 0 if (i + j >= R.shape[0]) else reward[i+j]
                R[i, 0] += multi * r
                multi *= self.gamma
            v = 0 if (i + self.n >= R.shape[0]) else V[i+self.n, 0]
            R[i, 0] += multi * v
        '''
        multi = 1
        actor_R = np.copy(R)
        for i in range(R.shape[0]):
            actor_R[i, 0] *= multi
            multi *= self.gamma
        '''
        # train actor model
        actor_loss = 0
        if update_actor:
            y_true = keras.utils.to_categorical(action, num_classes=NUM_ACTIONS)
            y_true = (R - V) * y_true
            actor_loss = self.model.train_on_batch(state, y_true)
        # train critic model
        critic_loss = self.critic_model.train_on_batch(state, R)
        total_rewards = sum(rewards)
        return actor_loss, critic_loss, total_rewards


def train_a2c(args, env: Env, params: dict) -> (list, int):
    score = []
    is_solved = 0

    agent = A2C(args, params=params)
    for e in range(agent.iteration, args.n_episodes):
        actor_loss, critic_loss, score = agent.train_step(env, update_actor=True)

        scorefile = open(args.result_save_path + "/scores.txt", "a+")
        scorefile.write(f"Episode: {e}, Score: {score} \n")
        scorefile.flush()
        scorefile.close()

        # Average score of last 100 episode
        is_solved = np.mean(score[-100:])
        if is_solved > 200:
            agent.model.save(args.model_save_path + "/model%09d" % e + '.h5')
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))

        # Checkpoint for models
        if e % 1 == 0:
            agent.model.save(args.model_save_path + "/model%09d" % e + '.h5')
            print(f"Saved model at episode {e}.")
            agent.critic_model.save(agent.critic_model_save_path + "/model%09d" % e + '.h5')
            print(f"Saved critic model at episode {e}.")

    return score, is_solved
