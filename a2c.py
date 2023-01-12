import sys
import glob
import torch  
import gymnasium as gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
hidden_size = 256


""" class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist """

class ActorCritic(nn.Module):
    def __init__(self, env, hidden_size):
        super(ActorCritic, self).__init__()
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.n
        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.dp1 = nn.Dropout(p=0.5)
        self.critic_linear2 = nn.Linear(hidden_size, 8)
        self.critic_linear3 = nn.Linear(8, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear3 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear4 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(x))
        value = self.dp1(value)
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)

        policy_dist = F.relu(self.actor_linear1(x))
        policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist = F.relu(self.actor_linear3(policy_dist))
        policy_dist = F.softmax(self.actor_linear4(policy_dist),dim=-1)

        return value, policy_dist

def load_model(args, env):
    model = ActorCritic(env, hidden_size)
    modelfiles = glob.glob("%s/model0*.h5" % args.model_save_path)
    modelfiles.sort()
    iteration = 0
    if len(modelfiles) >= 1:
        model = torch.load(modelfiles[-1])
        iteration = int(modelfiles[-1].split("/")[-1].replace(".h5", "").replace("model", ""))
        print(f"Model loaded from previous state at episode {iteration}.")
    return model, iteration

def train_a2c(args, env, params: dict) -> (list, int):
    agent, iteration = load_model(args, env)
    ac_optimizer = optim.Adam(agent.parameters(), lr=params['learning_rate'])
    _loss = []
    all_lengths = []
    entropy_term = 0

    for e in range(iteration, args.n_episodes):
        score = 0
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []
        max_steps = 3000
        for i in range(max_steps):
            value, policy_dist = agent.forward(state)
            value = value.detach().numpy()[0,0]
            dist = policy_dist.detach().numpy() 
            action = np.random.choice(env.action_space.n, p=np.squeeze(dist))
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = False
            if terminated or truncated:
                done = True
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = next_state
            score = np.sum(rewards)
            _loss.append(score)
            
            if done: # or i == max_steps-1:
                Qval, _ = agent.forward(next_state)
                Qval = Qval.detach().numpy()[0,0]
                all_lengths.append(i)   
                print("episode: {}/{}, score: {}".format(e + 1, args.n_episodes, score))    
                break

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + params['gamma'] * Qval
            Qvals[t] = Qval
  
        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

        scorefile = open(args.result_save_path + "/scores.txt", "a+")
        scorefile.write(f"Episode: {e}, Score: {score} \n")
        scorefile.flush()
        scorefile.close()

        # Average score of last 100 episode
        is_solved = np.mean(_loss[-100:])
        if is_solved > 200:
            torch.save(agent, args.model_save_path + "/model%09d" % e + '.h5')
            # agent.save(args.model_save_path + "/model%09d" % e + '.h5')
            print(f"Saved model at episode {e}.")

            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))

        # Checkpoint for models
        if e % 50 == 0:
            torch.save(agent, args.model_save_path + "/model%09d" % e + '.h5')
            print(f"Saved model at episode {e}.")
            

    return _loss, is_solved



