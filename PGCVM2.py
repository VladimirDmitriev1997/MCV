###
# REINFORCE agent for arbitrary environment
# CV with net for regression of a
###
import math
from scipy.integrate import quad
from scipy import special
import sys 
import os
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.utils as utils
from torch.distributions.categorical import Categorical
from gym import core, spaces

from gym import Env, spaces
from contextlib import closing


from io import StringIO


from gym.envs.toy_text import discrete


from gym.utils import seeding
pi = Variable(torch.FloatTensor([math.pi])).cpu()

class REINFORCE_Agent(nn.Module):
    def __init__(self, num_states, sampler_parameters_dim, hidden_size, learning_rate=3e-4):
        super(REINFORCE_Agent, self).__init__()

        self.linear1 = nn.Linear(num_states, hidden_size)
        torch.nn.init.normal_(self.linear1.weight,mean = 0.0, std = 0.01) 
        self.linear2 = nn.Linear(hidden_size, 1)
        torch.nn.init.normal_(self.linear2.weight,mean = 0.0, std = 0.01) 
        self.linear2_ = nn.Linear(hidden_size, 1)
        torch.nn.init.normal_(self.linear2_.weight,mean = 0.0, std = 0.01) 
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    
    

    
    
    
    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)
        sigma_sq = F.softplus(sigma_sq)
        

        return torch.transpose(torch.stack([mu, sigma_sq]),0,1)

        return output

    
    def get_action(self, state):
        ### Take state in numpy
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        
        highest_prob_action, rands = self.distribution_model(probs)
        
        highest_prob_action = torch.tensor(rands, dtype = torch.float32) * torch.sqrt(probs[:,1]**2) +  probs[:,0]
        log_prob = torch.log(torch.exp(-((highest_prob_action-probs[:,0])**2)/(2*probs[:,1]**2))*torch.sqrt(2*np.pi*probs[:,1]**2))
        return highest_prob_action, log_prob, rands, probs, None
    
    def distribution_model(self, probs):
        #This function set distribution of decisions
        rands = np.random.normal(size = probs.shape[0])
        highest_prob_action = torch.tensor(rands, dtype = torch.float32) * torch.sqrt(probs[:,1]**2) +  probs[:,0]

        return highest_prob_action, rands

    def compute_loss(self, rewards, log_probs, state_dim, action_dim, trajectory_len, gamma, states=None, actions=None, params=None, rands=None, cv_constructor=None):
        
        discounted_rewards = []
        l = []
        
        
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        return torch.stack(policy_gradient).sum()
        #self.optimizer.zero_grad()
        #policy_gradient = torch.stack(policy_gradient).mean() # sum()
        #policy_gradient.backward()
        #utils.clip_grad_norm(self.parameters(), 40)
       # self.optimizer.step()
        #return rewards
    
    def clean_arrays(self):
        return 0

        
    def parser(self, extra_):
        return 0



class REINFORCE_lake_Agent(nn.Module):
    def __init__(self, num_states, sampler_parameters_dim, hidden_size, max_drift,l2_lambda=0.01, entropy_factor = 0.0, learning_rate=3e-4):
        super(REINFORCE_lake_Agent, self).__init__()
        
        
        self.max_drift = max_drift
        self.values = []
        self.entropies = []
        self.entropy_factor = entropy_factor
        self.l2_lambda = l2_lambda
        self.linear1 = nn.Linear(num_states, 4)
        torch.nn.init.normal_(self.linear1.weight,mean = 0.0, std = 0.01)
        self.linear0 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 4)
        self.soft = nn.Softmax(dim=1,)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.nn_list = [self.linear1, self.linear0,  self.linear2, self.soft]
    
    

    
    
    
    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        a = self.soft(x)
        
        return a

        

    
    def get_action(self, state):
        ### Take state in numpy
        state = torch.from_numpy(state).float().unsqueeze(0)
        input = Variable(state)
        a = self.forward(input)
        
        tmp_prob = a.clone().detach() 
        starter = -torch.ones(tmp_prob.shape[0]).reshape(-1)
        tmp_prob[:,0] = 2*tmp_prob[:,0] + starter
        for i in range(1,tmp_prob.shape[1]):
            tmp_prob[:,i]=tmp_prob[:,i-1]+2*tmp_prob[:,i]
      
        aug_tmp_prob = torch.cat([-torch.ones(tmp_prob.shape[0]).reshape((1,-1)), tmp_prob[:,:-1].T]).T
        
        rands = np.random.uniform(size = aug_tmp_prob.shape[0], low=-1, high=1)
        #print(rands)
        mask1 = tmp_prob.ge(torch.tensor(rands).reshape((-1,1)))
        mask2 = aug_tmp_prob.le(torch.tensor(rands).reshape((-1,1)))
        mask = mask1.long()*mask2.long()
        highest_prob_action = torch.argmax(mask,dim=1)
        tmp_action = highest_prob_action.numpy()
        act = tmp_action
        stoch = False
        
        if stoch == True:
            act = np.random.choice([(tmp_action[0]-1)%4,(tmp_action[0])%4,(tmp_action[0]-1)%4], p = [1/5, 3/5, 1/5])

        self.entropies.append((a*torch.log(a)).sum())
        #highest_prob_action = torch.tensor(rands, dtype = torch.float32) * torch.sqrt(probs[:,1]**2) +  probs[:,0]
        log_prob = torch.log(a[:,int(act)])#[:,mask.detach()]
        highest_prob_action = torch.tensor(act)
     
        return highest_prob_action, log_prob, rands, None, None
    
    def distribution_model(self, probs):
        #This function set distribution of decisions
        rands = np.random.normal(size = probs.shape[0])
        highest_prob_action = torch.tensor(rands, dtype = torch.float32) * torch.sqrt(probs[:,1]**2) +  probs[:,0]

        return highest_prob_action, rands

    def compute_loss(self, rewards, log_probs, state_dim, action_dim, trajectory_len, gamma, states=None, actions=None, params=None, rands=None, cv_constructor=None):
        
        discounted_rewards = []
        l = []
        
        v_loss = 0

        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
        deltas = discounted_rewards

        
        if cv_constructor != None:
            if cv_constructor.status == 'learning':
                deltas = cv_constructor.get_cv_correction(1, states, actions, deltas)
            if cv_constructor.status == 'work':
                deltas = cv_constructor.get_cv_correction(1, states, actions, deltas)
        
        #deltas = np.array(deltas).reshape((-1,1))
        #if cv_constructor != None:
        #    if cv_constructor.status == 'learning':
        #        deltas = cv_constructor.learn_regression(1, actions, states, deltas)
        #        return 0
        #    if cv_constructor.status == 'work':
        #        deltas = cv_constructor.get_cv_correction(states, actions, deltas)
        
        
        
        

        test = 0
        entropy = 0
        for i in range(len(rewards)):
            v_loss = v_loss - (log_probs[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32))).cpu()) + self.entropy_factor*self.entropies[i]
            entropy += self.entropies[i]
            test = test - (log_probs[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32))).cpu()).detach()

        loss = v_loss + (test/self.entropy_factor)*entropy
        l2_lambda = self.l2_lambda
        l2_reg = torch.tensor(0.).cpu()
        for model in self.nn_list:
          for param in model.parameters():
            l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
        return loss
     
        #self.optimizer.zero_grad()
        #policy_gradient = torch.stack(policy_gradient).mean() # sum()
        #policy_gradient.backward()
        #utils.clip_grad_norm(self.parameters(), 40)
       # self.optimizer.step()
        #return rewards
    
    def clean_arrays(self):
        self.entropies = []
        return 0

        
    def parser(self, extra_):
        return 0

class REINFORCE_lake_Tab(nn.Module):
    def __init__(self, num_states, sampler_parameters_dim, hidden_size, max_drift,l2_lambda=0.01, entropy_factor = 0.0, learning_rate=3e-4):
        super(REINFORCE_lake_Tab, self).__init__()
        
        
        self.max_drift = max_drift
        self.values = []
        self.entropies = []
        self.entropy_factor = entropy_factor
        self.l2_lambda = l2_lambda
        self.linear1 = nn.Linear(num_states, 4)
        torch.nn.init.normal_(self.linear1.weight,mean = 0.0, std = 0.01)
        self.linear0 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 4)
        self.soft = nn.Softmax(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.nn_array = []
        for i in range(4):
            self.nn_array.append([])
            for j in range(4):
                layer1 = nn.Linear(1, 4, bias = False)
                torch.nn.init.normal_(layer1.weight.data,mean = 0.0, std = 0.01) 
                
                
                layer2 = nn.Softmax(dim=1)
                self.nn_array[-1].append(nn.Sequential(layer1, layer2))
        
    
    

    
    
    
    def forward(self, inputs):
        x = inputs[1]
        inputs = inputs[0]
        
        ap = self.nn_array[int(inputs[0,0])][int(inputs[0,1])]
        a = ap(x)
        #x = self.linear1(x)
        #a = self.soft(x)
        
        return a

        

    
    def get_action(self, state):
        ### Take state in numpy
        state = torch.from_numpy(state).float().unsqueeze(0)
        input = Variable(state)
        un = torch.tensor(1, dtype=torch.float32).reshape((1,1)).requires_grad_(True)
        a = self.forward([input,un])
        
        tmp_prob = a.clone().detach() 
        starter = -torch.ones(tmp_prob.shape[0]).reshape(-1)
        tmp_prob[:,0] = 2*tmp_prob[:,0] + starter
        for i in range(1,tmp_prob.shape[1]):
            tmp_prob[:,i]=tmp_prob[:,i-1]+2*tmp_prob[:,i]
      
        aug_tmp_prob = torch.cat([-torch.ones(tmp_prob.shape[0]).reshape((1,-1)), tmp_prob[:,:-1].T]).T
        
        rands = np.random.uniform(size = aug_tmp_prob.shape[0], low=-1, high=1)
        
        mask1 = tmp_prob.ge(torch.tensor(rands).reshape((-1,1)))
        mask2 = aug_tmp_prob.le(torch.tensor(rands).reshape((-1,1)))
        mask = mask1.long()*mask2.long()

        highest_prob_action = torch.argmax(mask,dim=1)

        tmp_action = highest_prob_action.numpy()
        act = tmp_action
        stoch = False
        
        if stoch == True:
            act = np.random.choice([(tmp_action[0]-1)%4,(tmp_action[0])%4,(tmp_action[0]-1)%4], p = [1/3, 1/3, 1/3])

        self.entropies.append((-a*torch.log(a)).sum())
        #highest_prob_action = torch.tensor(rands, dtype = torch.float32) * torch.sqrt(probs[:,1]**2) +  probs[:,0]
        log_prob = torch.log(a[:,int(act)])#[:,mask.detach()]
        highest_prob_action = torch.tensor(act)
     
        return highest_prob_action, log_prob, rands, None, None
    
    def distribution_model(self, probs):
        #This function set distribution of decisions
        rands = np.random.normal(size = probs.shape[0])
        highest_prob_action = torch.tensor(rands, dtype = torch.float32) * torch.sqrt(probs[:,1]**2) +  probs[:,0]

        return highest_prob_action, rands

    def compute_loss(self, rewards, log_probs, state_dim, action_dim, trajectory_len, gamma, states=None, actions=None, params=None, rands=None, cv_constructor=None):
        
        discounted_rewards = []
        l = []
        
        v_loss = 0

        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
        deltas = discounted_rewards
        if deltas[-1] == 0:
            return torch.tensor(0., dtype = torch.float32).cpu().requires_grad_(True)
        
        if cv_constructor != None:
            if cv_constructor.status == 'learning':
                deltas = cv_constructor.get_cv_correction(1, states, actions, deltas)
            if cv_constructor.status == 'work':
                deltas = cv_constructor.get_cv_correction(1, states, actions, deltas)
        
        #deltas = np.array(deltas).reshape((-1,1))
        #if cv_constructor != None:
        #    if cv_constructor.status == 'learning':
        #        deltas = cv_constructor.learn_regression(1, actions, states, deltas)
        #        return 0
        #    if cv_constructor.status == 'work':
        #        deltas = cv_constructor.get_cv_correction(states, actions, deltas)
        
        
        
        

        test = 0
        entropy = 0
        for i in range(len(rewards)):
            v_loss = v_loss - (log_probs[i]*(Variable(torch.tensor(1, dtype = torch.float32))).cpu())
            entropy += self.entropies[i]
            test = test - (log_probs[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32))).cpu()).detach()

        loss = v_loss + (test/self.entropy_factor)*entropy
       
        l2_lambda = self.l2_lambda
        l2_reg = torch.tensor(0.).cpu()
        for i in self.nn_array:
            for model in i:
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                    loss += l2_lambda * l2_reg
        
        return loss
     
        #self.optimizer.zero_grad()
        #policy_gradient = torch.stack(policy_gradient).mean() # sum()
        #policy_gradient.backward()
        #utils.clip_grad_norm(self.parameters(), 40)
       # self.optimizer.step()
        #return rewards
    
    def clean_arrays(self):
        self.entropies = []
        return 0

        
    def parser(self, extra_):
        return 0

class A2C_Agent(nn.Module):
    def __init__(self, num_states, sampler_parameters_dim, hidden_size,l2_lambda=0.01, entropy_factor = 0.0, learning_rate=3e-4):
        super(A2C_Agent, self).__init__()



        

        self.values = []
        self.entropies = []
        self.entropy_factor = entropy_factor


        self.linear1 = nn.Linear(num_states, hidden_size).cpu()
        self.linear2 = nn.Linear(hidden_size, 1).cpu()
        self.linear2_ = nn.Linear(hidden_size, 1).cpu()
        self.linear3 = nn.Linear(hidden_size, hidden_size).cpu()
        self.linear4_v = nn.Linear(num_states, hidden_size).cpu()
        self.linear5 = nn.Linear(hidden_size, hidden_size).cpu()
        self.linear_v5 = nn.Linear(hidden_size, 1).cpu()
        self.nn_list = [self.linear1,self.linear2,self.linear2_,self.linear4_v, self.linear_v5]
        self.l2_lambda = l2_lambda
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        x = inputs.cpu()
        y = inputs.cpu()
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq= self.linear2_(x)
        sigma_sq = F.softplus(sigma_sq)
        
        y = F.relu(self.linear4_v(y))
        value= self.linear_v5(y)
        return value, torch.transpose(torch.stack([mu, sigma_sq]),0,1)
    

    def get_action(self, state):
        ### Take state in numpy
        state = torch.from_numpy(state).float().unsqueeze(0)
        value, probs = self.forward(Variable(state).cpu())
        
        mu = probs[:,0]
        sigma_sq = probs[:,1]
        highest_prob_action, rands = self.distribution_model(probs)
        self.entropies.append(-0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1))
        self.values.append(value)
        highest_prob_action = torch.tensor(rands, dtype = torch.float32).cpu() * torch.sqrt(probs[:,1]**2) +  probs[:,0]
        log_prob = -((highest_prob_action-probs[:,0])**2)/(2*probs[:,1]**2) + torch.log(torch.sqrt(2*np.pi*probs[:,1]**2))
        return highest_prob_action, log_prob, rands, probs, None
    
    def distribution_model(self, probs):
        #This function set distribution of decisions
        rands = np.random.normal(size = probs.shape[0])
        highest_prob_action = torch.tensor(rands, dtype = torch.float32).cpu() * torch.sqrt(probs[:,1]**2) +  probs[:,0]

        return highest_prob_action, rands

    def compute_loss(self, rewards, log_probs, state_dim, action_dim, trajectory_len, gamma, states=None, actions=None, params=None, rands=None, cv_constructor=None):
        
        p_loss = 0
        v_loss = 0
        deltas = []


        if cv_constructor != None:
            if cv_constructor.status == 'learning':
                rewards = cv_constructor.learn_regression(1, actions, states, rewards)
                return 0
            if cv_constructor.status == 'work':
                rewards = cv_constructor.get_cv_correction(states, actions, rewards)
        

        for i in range(len(rewards)):
            if i!=len(rewards)-1: 
                deltas.append((gamma**i)*(rewards[i] + gamma*self.values[i+1].detach() - self.values[i].detach()).reshape(-1))
            else:
                deltas.append((gamma**i)*(rewards[i] - self.values[i].detach()).reshape(-1))
        
        #deltas = np.array(deltas).reshape((-1,1))
        #if cv_constructor != None:
        #    if cv_constructor.status == 'learning':
        #        deltas = cv_constructor.learn_regression(1, actions, states, deltas)
        #        return 0
        #    if cv_constructor.status == 'work':
        #        deltas = cv_constructor.get_cv_correction(states, actions, deltas)
        
        
        
        

     
        for i in range(len(rewards)):
            p_loss = p_loss - self.values[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32)).cpu())
            v_loss = v_loss - (log_probs[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32))).cpu()) - (self.entropy_factor*self.entropies[i].cpu()).sum()
    
        loss = (v_loss + p_loss)
        l2_lambda = self.l2_lambda
        l2_reg = torch.tensor(0.).cpu()
        for model in self.nn_list:
          for param in model.parameters():
            l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
        return loss
         


        #self.optimizer.zero_grad()
        #loss.backward()
        #utils.clip_grad_norm(self.parameters(), 40)
        #self.optimizer.step()
        #self.entropies = []
        #self.values = []

    def clean_arrays(self):
        self.values = []
        self.entropies = []
        return 0  
      
    def parser(self, extra_):

        return 0


class A2C_Agent_Pole(nn.Module):
    def __init__(self, num_states, sampler_parameters_dim, hidden_size, max_drift,l2_lambda=0.01, entropy_factor = 0.0, learning_rate=3e-4):
        super(A2C_Agent_Pole, self).__init__()

        self.action_bound = 1

        
        
        self.max_drift = max_drift
        self.values = []
        self.entropies = []
        self.entropy_factor = entropy_factor


        self.linear1 = nn.Linear(num_states, hidden_size).cpu()
        self.linear2 = nn.Linear(hidden_size, 1).cpu()
        self.linear2_ = nn.Linear(hidden_size, 1).cpu()
        self.linear3 = nn.Linear(hidden_size, hidden_size).cpu()
        self.linear4_v = nn.Linear(num_states, hidden_size).cpu()
        self.linear5 = nn.Linear(hidden_size, hidden_size).cpu()
        self.linear_v5 = nn.Linear(hidden_size, 1).cpu()
        self.nn_list = [self.linear1,self.linear2,self.linear2_,self.linear4_v, self.linear_v5]
        self.l2_lambda = l2_lambda

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        x = inputs.cpu()
        y = inputs.cpu()
        x = F.relu(self.linear1(x))
        mu = F.tanh(self.linear2(x))
        #sigma_sq= self.linear2_(x)
        sigma_sq = self.max_drift*torch.ones_like(mu)#self.max_drift #F.softplus(sigma_sq)
        
        y = F.relu(self.linear4_v(y))
        value= self.linear_v5(y)
        return value, torch.transpose(torch.stack([mu, sigma_sq]),0,1)
    

    def get_action(self, state):
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        value, probs = self.forward(Variable(state).cpu())
        
        mu = probs[:,0]
        sigma_sq = probs[:,1]
        highest_prob_action, rands = self.distribution_model(probs)
        self.entropies.append(-0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1))
        self.values.append(value)
        highest_prob_action = self.action_bound*F.tanh(torch.tensor(rands, dtype = torch.float32).cpu() * torch.sqrt(probs[:,1]**2) +  probs[:,0])
        
        log_prob = -((highest_prob_action-probs[:,0])**2)/(2*probs[:,1]**2) + torch.log(torch.sqrt(2*np.pi*probs[:,1]**2))
        return highest_prob_action, log_prob, rands, probs, None
    
    def distribution_model(self, probs):
        #This function set distribution of decisions
        rands = np.random.normal(size = probs.shape[0])
        highest_prob_action = torch.tensor(rands, dtype = torch.float32).cpu() * torch.sqrt(probs[:,1]**2) +  probs[:,0]

        return highest_prob_action, rands

    def compute_loss(self, rewards, log_probs, state_dim, action_dim, trajectory_len, gamma, states=None, actions=None, params=None, rands=None, cv_constructor=None):
        
        p_loss = 0
        v_loss = 0
        deltas = []


        if cv_constructor != None:
            if cv_constructor.status == 'learning':
                rewards = cv_constructor.learn_regression(1, actions, states, rewards)
                return 0
            if cv_constructor.status == 'work':
                rewards = cv_constructor.get_cv_correction(states, actions, rewards)
        

        for i in range(len(rewards)):
            if i!=len(rewards)-1: 
                deltas.append((gamma**i)*(rewards[i] + gamma*self.values[i+1].detach() - self.values[i].detach()).reshape(-1))
            else:
                deltas.append((gamma**i)*(rewards[i] - self.values[i].detach()).reshape(-1))
        
        #deltas = np.array(deltas).reshape((-1,1))
        #if cv_constructor != None:
        #    if cv_constructor.status == 'learning':
        #        deltas = cv_constructor.learn_regression(1, actions, states, deltas)
        #        return 0
        #    if cv_constructor.status == 'work':
        #        deltas = cv_constructor.get_cv_correction(states, actions, deltas)
        
        
        
        

     
        for i in range(len(rewards)):
            p_loss = p_loss - self.values[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32)).cpu())
            v_loss = v_loss - (log_probs[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32))).cpu()) - (self.entropy_factor*self.entropies[i].cpu()).sum()
    
        loss = (v_loss + p_loss)
        l2_lambda = self.l2_lambda
        l2_reg = torch.tensor(0.).cpu()
        for model in self.nn_list:
          for param in model.parameters():
            l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
        return loss
         



    def clean_arrays(self):
        self.values = []
        self.entropies = []
        return 0  
      
    def parser(self, extra_):

        return 0

class A2C_Agent_Pole4(nn.Module):
    def __init__(self, num_states, sampler_parameters_dim, hidden_size, max_drift,l2_lambda=0.01, entropy_factor = 0.0, learning_rate=3e-4):
        super(A2C_Agent_Pole4, self).__init__()

        self.action_bound = 1

        
        
        self.max_drift = max_drift
        self.values = []
        self.entropies = []
        self.entropy_factor = entropy_factor


        self.linear1 = nn.Linear(num_states, hidden_size).cpu()
        self.linear1_ = nn.Linear(num_states, hidden_size).cpu()
        self.linear2 = nn.Linear(hidden_size, 1).cpu()
        self.linear2_ = nn.Linear(hidden_size, 1).cpu()
        self.linear3 = nn.Linear(hidden_size, hidden_size).cpu()
        self.linear4_v = nn.Linear(num_states, hidden_size).cpu()
        self.linear5 = nn.Linear(hidden_size, hidden_size).cpu()
        self.linear_v5 = nn.Linear(hidden_size, 1).cpu()
        self.nn_list = [self.linear1,self.linear2,self.linear2_,self.linear3,self.linear4_v, self.linear_v5]
        self.l2_lambda = l2_lambda

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        x = inputs.cpu()
        y = inputs.cpu()
        z = inputs.cpu()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear3(x))
        mu = self.action_bound*F.tanh(self.linear2(x))
        y = F.relu(self.linear1_(y))
        y = F.relu(self.linear2_(y))
        sigma_sq = F.softplus(y)+ 1e-7 #self.max_drift*torch.ones_like(mu)#self.max_drift #F.softplus(sigma_sq)        ATTENTION CRUCIAL
        
        z = F.relu(self.linear4_v(z))
        value= self.linear_v5(z)
        return value, torch.transpose(torch.stack([mu, sigma_sq]),0,1)
    

    def get_action(self, state):
        ### Take state in numpy
        state = torch.from_numpy(state).float().unsqueeze(0)
        value, probs = self.forward(Variable(state).cpu())
        
        mu = probs[:,0]
        sigma_sq = probs[:,1]
        highest_prob_action, rands = self.distribution_model(probs)
        self.entropies.append(-0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1))
        self.values.append(value)
        highest_prob_action = self.action_bound*F.tanh(torch.tensor(rands, dtype = torch.float32).cpu() * torch.sqrt(probs[:,1]**2) +  probs[:,0])
        
        log_prob = -((highest_prob_action-probs[:,0])/(np.sqrt(2)*probs[:,1]))**2 - torch.log(np.sqrt(2*np.pi)*probs[:,1])
        return highest_prob_action, log_prob, rands, probs, None
    
    def distribution_model(self, probs):
        #This function set distribution of decisions
        rands = np.random.normal(size = probs.shape[0])
        highest_prob_action = torch.tensor(rands, dtype = torch.float32).cpu() * torch.sqrt(probs[:,1]**2) +  probs[:,0]

        return highest_prob_action, rands

    def compute_loss(self, rewards, log_probs, state_dim, action_dim, trajectory_len, gamma, states=None, actions=None, params=None, rands=None, cv_constructor=None):
        
        p_loss = 0
        v_loss = 0
        deltas = []



        

        for i in range(len(rewards)):
            if i!=len(rewards)-1: 
                deltas.append((gamma**i)*(rewards[i] + gamma*self.values[i+1].detach() - self.values[i].detach()).reshape(-1))
            else:
                deltas.append((gamma**i)*(rewards[i] - self.values[i].detach()).reshape(-1))
        

        if cv_constructor != None:
            if cv_constructor.status == 'learning':
                deltas = cv_constructor.get_cv_correction(1, states, actions, deltas)
            if cv_constructor.status == 'work':
                deltas = cv_constructor.get_cv_correction(1, states, actions, deltas)
        
        #deltas = np.array(deltas).reshape((-1,1))
        #if cv_constructor != None:
        #    if cv_constructor.status == 'learning':
        #        deltas = cv_constructor.learn_regression(1, actions, states, deltas)
        #        return 0
        #    if cv_constructor.status == 'work':
        #        deltas = cv_constructor.get_cv_correction(states, actions, deltas)
        
        
        
        

     
        for i in range(len(rewards)):
            p_loss = p_loss - self.values[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32)).cpu())
            v_loss = v_loss - (log_probs[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32))).cpu()) - (self.entropy_factor*self.entropies[i].cpu()).sum()
    
        loss = (v_loss + p_loss)
        l2_lambda = self.l2_lambda
        l2_reg = torch.tensor(0.).cpu()
        for model in self.nn_list:
          for param in model.parameters():
            l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
        return loss
         


        #self.optimizer.zero_grad()
        #loss.backward()
        #utils.clip_grad_norm(self.parameters(), 40)
        #self.optimizer.step()
        #self.entropies = []
        #self.values = []

    def clean_arrays(self):
        self.values = []
        self.entropies = []
        return 0  
      
    def parser(self, extra_):

        return 0



class A2C_Agent_Discrete(nn.Module):
    def __init__(self, num_states, sampler_parameters_dim, hidden_size, action_num, entropy_factor = 0.0,learning_rate=3e-4):
        super(A2C_Agent_Discrete, self).__init__()



        

        self.values = []
        self.entropies = []
        self.entropy_factor = entropy_factor

        self.action_num = action_num
        self.linear1 = nn.Linear(num_states, hidden_size).cpu()
        self.linear2 = nn.Linear(hidden_size, action_num).cpu()
        #self.linear2_ = nn.Linear(hidden_size, 1).cpu()
        #self.linear3 = nn.Linear(hidden_size, hidden_size).cpu()
        self.linear4_v = nn.Linear(num_states, hidden_size).cpu()
        self.linear5 = nn.Linear(hidden_size, hidden_size).cpu()
        self.linear_v5 = nn.Linear(hidden_size, 1).cpu()
        

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        x = inputs.cpu()
        y = inputs.cpu()
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        
        y = F.relu(self.linear4_v(y))
        value= self.linear_v5(y)
        return value, F.softmax(x)
    

    def get_action(self, state):
        ### Take state in numpy
        state = torch.from_numpy(state).float().unsqueeze(0)
        value, probs = self.forward(Variable(state).cpu())
        

        
        #self.entropies.append(-0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1))
        self.entropies.append(-probs*probs.log().sum(1, keepdim=True))
        self.values.append(value)
        m = Categorical(torch.tensor(probs.detach().cpu()))
        
        highest_prob_action = m.sample().reshape((-1,1)).cpu()
        log_prob = torch.gather(probs,1,highest_prob_action).log()
        return highest_prob_action, log_prob, None, probs, None
    
    def distribution_model(self, probs):
        #This function set distribution of decisions
        rands = np.random.normal(size = probs.shape[0])
        highest_prob_action = torch.tensor(rands, dtype = torch.float32).cpu() * torch.sqrt(probs[:,1]**2) +  probs[:,0]

        return highest_prob_action, rands

    def compute_loss(self, rewards, log_probs, state_dim, action_dim, trajectory_len, gamma, states=None, actions=None, params=None, rands=None, cv_constructor=None):
        
        p_loss = 0
        v_loss = 0
        deltas = []

        if cv_constructor != None:
            if cv_constructor.status == 'learning':
                rewards = cv_constructor.learn_regression(1, actions, states, rewards)
                return 0
            if cv_constructor.status == 'work':
                rewards = cv_constructor.get_cv_correction(states, actions, rewards)
        
        for i in range(len(rewards)):
            if i!=len(rewards)-1: 
                deltas.append((gamma**i)*(rewards[i] + gamma*self.values[i+1].detach() - self.values[i].detach()).reshape(-1))
            else:
                deltas.append((gamma**i)*(rewards[i] - self.values[i].detach()).reshape(-1))
        
        
        
        

     
        for i in range(len(rewards)):
            p_loss = p_loss - self.values[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32)).cpu())
            v_loss = v_loss - (log_probs[i]*(Variable(torch.tensor(deltas[i], dtype = torch.float32))).cpu()) - (self.entropy_factor*self.entropies[i].cpu()).sum()
    
        loss = (v_loss + p_loss)/ len(rewards)
    
        return loss
        #self.optimizer.zero_grad()
        #loss.backward()
        #utils.clip_grad_norm(self.parameters(), 40)
        #self.optimizer.step()
   
    

    def clean_arrays(self):
        self.entropies = []
        self.values = []
        return 0
        
    def parser(self, extra_):

        return 0


class ULA():
    def __init__(self, X, trajectory_len, gamma, state_dim, f, lag, K, burn_in, burn_off, polynomial, a, b,  lr = 0.001):
        super(ULA, self).__init__()
        
        
       
        self.lr = lr
        self.states_history = []
        self.trajectory_len = trajectory_len # maximal trajectory length
        self.t = 0 # current step
        self.rands = []
        self.state_dim = state_dim
        self.f = f
        self.gamma = gamma
        self.state = X
        self.loss = []
        self.function = []
        self.archive = None
        #self.states_history.append(self.state)
        #self.function.append(self.f(self.state))

        self.lag = lag
        self.K = K
        self.Nets = []
        self.burn_in = burn_in
        self.burn_off = burn_off
        self.polynomial = polynomial
        self.H = np.zeros((self.trajectory_len, self.K)) 
        self.c_1 = a
        self.c_2 = b
    def clean(self):
        self.archive = None
        self.H = np.zeros((self.trajectory_len, self.K)) 
        self.function = []
        self.t = 0 # current step
        self.rands = []
        self.states_history = []
        

    def step(self):
        if self.t>=self.trajectory_len:
            return 0
         #entropy, rand, param, value
        rand = np.random.normal()
        
        self.get_H(self.t,rand)
        
        next_state = (1 - self.gamma)*self.state + np.sqrt(2*self.gamma)*rand
        self.state = next_state
        self.function.append(self.f(self.state))
        state = torch.Tensor([next_state])

        
        self.states_history.append(state.cpu().detach().numpy().reshape(-1))
        self.rands.append(rand)
        
        self.t=self.t+1
        
        return 1
    

    def init_regression(self, states_num):
        for i in range(self.lag+1):
            Nets_ = []
            for j in range(self.K):
                net = Approx_net(states_num).cpu()
                Nets_.append([net, nn.MSELoss(), optim.Adam(net.parameters(), lr=self.lr)])
            self.Nets.append(Nets_)
    
    def construct_Q(self, states, y):
        ### get actions, states and target value to rearrange them in convenient way
        #actions = actions.reshape((-1,self.action_dim))
        states = states.reshape((-1,self.state_dim))
        y = np.array(y).reshape((-1,1))
        q = np.hstack((states,y))
        q = torch.from_numpy(q)
        Q = torch.tensor(q,dtype = torch.float32, requires_grad=True).cpu()
        return Q
    
    
  
        
    def get_H(self, t, rand):
        for k in range(1, self.K+1):
            self.H[t,k-1] = self.polynomial(k,t,rand)

        #self.trajectory_len = self.trajectory_len + 1
        
    
    
        

    def a_net(self, x, q, k, Nets):
        # propagation of single state-action pair to get regression
        x_ = x
        output = Nets[q][k][0](x_)
        return output



    def finalize_trajectory(self):
        a = []
        for i in self.states_history:
            a.append(float(i))
        self.states_history = np.array(a)
        self.rands = np.array(self.rands)
        self.function = np.array(self.function)
        Q = self.construct_Q(self.states_history, self.function)
        trajectory_len = Q.shape[0]
        list_of_cvs = [ [] for _ in range(self.lag+1) ]
        n_epoches = 1
        for epoch in range(n_epoches):
            for t in range(trajectory_len):
                c = 0
                for q in range(max(t-self.lag,0),t+1):
                    for k in range(1,self.K+1):
                        r = t - q
                        x = torch.tensor(Q[t-r,:self.state_dim].detach().clone(),dtype = torch.float32, requires_grad=True).detach().cpu()
                        x = torch.tensor(x,dtype = torch.float32, requires_grad=True).reshape((1,-1)).cpu()
                        if q!=trajectory_len-1:
                            b_2 = self.a_net(x, r, k-1, self.Nets)*self.H[q,k-1]
                        else:
                            self.a_net(x, r, k-1, self.Nets)*self.H[q-1,k-1]*0
                        if k == 1:
                            list_of_cvs[r].append(b_2.detach().cpu().numpy())
                        else:
                            list_of_cvs[r][-1] = list_of_cvs[r][-1] + b_2.detach().cpu().numpy()
                        if c == 0:
                            c = 1
                            CV_ = b_2
                        else:
                            CV_ = torch.cat((b_2, CV_))
              
                CV1 = torch.sum(CV_, dim = 0, keepdim = True)
                if t == 0:
                    Deltas_shifted = Q[t,-1].detach().clone() - CV1#torch.tensor((Q[t,0]-CV1),dtype = torch.float32, requires_grad=True).reshape((1,-1)).cpu()
                else:
                    Deltas_shifted = torch.cat((Deltas_shifted,Q[t,-1].detach().clone() - CV1)).cpu()



            loss = self.Loss(Deltas_shifted, Q[:,-1].detach().clone())
            loss.backward()
            
            for i in range(self.lag+1):
                for j in range(self.K):
                    self.Nets[i][j][2].step()
                    self.Nets[i][j][2].zero_grad()
            self.archive = [list_of_cvs, Q[:,-1].cpu().detach().numpy(),Deltas_shifted.reshape((-1,1)).cpu().detach().clone().numpy()] #[] Deltas_shifted.reshape((-1,1)).cpu().detach().clone().numpy()

        self.loss.append(loss)
        
    def Loss(self, R, ref):
        # Loss for control variate
        c_1 = self.c_1
        c_2 = self.c_2
        cv = (ref-R)
        expectation = torch.mean(ref)
        L =  c_1*torch.mean(torch.mul(R-expectation,R - expectation)) + c_2*torch.mean(torch.mul(cv,cv))
        #self.loss_archive = L
        return L




class T_ULA():
    def __init__(self, X, trajectory_len, gamma, state_dim, f, lag, K, burn_in, burn_off, polynomial, a, b, c, sigma, batch_size, lr = 0.001):
        super(T_ULA, self).__init__()
        
        
        self.sigma1 = sigma
        self.lr = lr
        self.states_history = []
        self.trajectory_len = trajectory_len # maximal trajectory length
        self.t = 0 # current step
        self.rands = []
        self.state_dim = state_dim
        self.f = f
        self.gamma = gamma
        self.state = X
        
        self.loss = []
        self.function = []
        self.archive = []
        #self.states_history.append(self.state)
        #self.function.append(self.f(self.state))
        
        self.batch_size = batch_size

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -2.0
        self.max_action = 2.0

        # Angle at which to fail the episode
        self.tresh_phi = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.tresh_phidot = self.force_mag


        self.lag = lag
        self.K = K
        self.Nets = []
        self.burn_in = int(burn_in)
        self.burn_off = int(burn_off)
        self.polynomial = polynomial
        self.H = np.zeros((self.trajectory_len, self.K)) 
        self.c_1 = a
        self.c_2 = b
        self.c_4 = c
    def clean(self, X):
        #self.archive = None
        self.H = np.zeros((self.trajectory_len, self.K)) 
        self.function = []
        self.t = 0 # current step
        self.rands = []
        self.states_history = []
        self.state = X
        
    def get_action(self, x):
      a = - np.sin((np.pi/2)*(x[2]/self.tresh_phi))*np.cos(np.pi*(x[3]/self.tresh_phidot))
      #a = -x[2]/(self.tresh_phi)
      return a
    
    def dynamics(self, force):
      x, x_dot, theta, theta_dot = self.state
      costheta = math.cos(theta)
      sintheta = math.sin(theta)
      temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
      thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
      xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
      x = x + self.tau * x_dot
      x_dot = x_dot + self.tau * xacc
      theta = theta + self.tau * theta_dot
      theta_dot = theta_dot + self.tau * thetaacc
      return (x, x_dot, theta, theta_dot)

    def step(self):
        if self.t>=self.trajectory_len:
            return 0
         #entropy, rand, param, value
        rand = np.random.normal()
        a = self.get_action(self.state)
        self.get_H(self.t,rand)
        next_state = self.dynamics((a+self.sigma1*rand)*self.force_mag)
        
        self.state = next_state
        self.function.append(self.f(self.state))
        state = torch.Tensor([next_state])

        
        self.states_history.append(state.cpu().detach().numpy().reshape(-1))
        self.rands.append(rand)
        
        self.t=self.t+1
        
        x = self.state[0]
        theta = self.state[2]

        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.tresh_phi \
            or theta > self.tresh_phi
        done = bool(done)
        if done:
          return 0
        else:
          return 1
    

    def init_regression(self, states_num):
        for i in range(self.lag+1):
            Nets_ = []
            for j in range(self.K):
                net = Approx_net(states_num).cpu()
                Nets_.append([net, nn.MSELoss(), optim.Adam(net.parameters(), lr=self.lr)])
            self.Nets.append(Nets_)
    
    def construct_Q(self, states, y):
        ### get actions, states and target value to rearrange them in convenient way
        #actions = actions.reshape((-1,self.action_dim))
        states = states.reshape((-1,self.state_dim))
        y = np.array(y).reshape((-1,1))
        q = np.hstack((states,y))
        q = torch.from_numpy(q)
        Q = torch.tensor(q,dtype = torch.float32, requires_grad=True).cpu()
        return Q
    
    
  
        
    def get_H(self, t, rand):
        for k in range(1, self.K+1):
            self.H[t,k-1] = self.polynomial(k,t,rand)

        #self.trajectory_len = self.trajectory_len + 1
        
    
    
        

    def a_net(self, x, q, k, Nets):
        # propagation of single state-action pair to get regression
        x_ = x
        output = Nets[q][k][0](x_)
        return output



    def finalize_trajectory(self):
        a = []
        for i in self.states_history:
            a.append(i.reshape(-1))
        self.states_history = np.array(a)
        self.rands = np.array(self.rands)
        self.function = np.array(self.function)
        Q = self.construct_Q(self.states_history, self.function)
        trajectory_len = Q.shape[0]
        #print(Q.shape)
        if (trajectory_len - self.burn_in - self.burn_off <2):
          return 0
        list_of_cvs = [ [] for _ in range(self.lag+1) ]
        n_epoches = 1
        batch_size = int(self.batch_size)
        while int((trajectory_len-self.burn_in - self.burn_off)%batch_size) == 1:
          batch_size = batch_size+1
        if trajectory_len -self.burn_in - self.burn_off < batch_size:
          batch_size = trajectory_len -self.burn_in - self.burn_off
        n_iter = int(math.ceil((trajectory_len-self.burn_in - self.burn_off)/batch_size))
        #print(n_iter)
        #print(batch_size)
        for epoch in range(n_epoches):
          for n_i in range(n_iter):
            c = 0
            for r in range(0, self.lag+1):
              for k in range(1,self.K+1):
                if n_i == n_iter-1  and (r>0 or self.burn_off>0):
                  batch = Q[self.burn_in+n_i*batch_size-r:-self.burn_off-r,:self.state_dim].detach().clone()
                  #print('a')
                  #print(n_i)
                  #print(batch.shape)
                  b_2 = self.a_net(batch, r, k-1, self.Nets)*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:trajectory_len-self.burn_off-r,k-1]).reshape(-1,1).cpu()
                elif n_i == n_iter-1  and (self.burn_off==0 and r==0):
                  batch = Q[self.burn_in+n_i*batch_size:,:self.state_dim].detach().clone()
                  #print('b')
                  #print(n_i)
                  #print(batch.shape)
                  b_2 = self.a_net(batch, r, k-1, self.Nets)*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:trajectory_len-r,k-1]).reshape(-1,1).cpu()
                else:
                  batch = Q[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,:self.state_dim].detach().clone()
                  #print('c')
                  #print(n_i)
                  #print(batch.shape)
                  b_2 = self.a_net(batch, r, k-1, self.Nets)*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,k-1]).reshape(-1,1).cpu()
                if k == 1:
                  list_of_cvs[r].append(b_2.detach().cpu().numpy())
                else:
                  list_of_cvs[r][-1] = list_of_cvs[r][-1] + b_2.detach().cpu().numpy()
                if c == 0:
                  c = 1
                  CV_ = b_2
                else:
                  CV_ +=b_2
            #print(CV_.shape)
            if n_i == 0:
              CV1 = CV_
            else:
              CV1 = torch.cat((CV1,CV_)).cpu()
            #print(Q[:,-1].detach().clone().shape)
            #print(CV1.shape)
          #print('aaa')
          Deltas_shifted = Q[:,-1].detach().clone().reshape(-1,1)  - torch.cat([torch.zeros(self.burn_in).reshape(-1,1).cpu(),CV1,torch.zeros(self.burn_off).reshape(-1,1).cpu()])

          loss = self.Loss(Deltas_shifted, Q[:,-1].detach().clone())
          loss.backward()
          for i in range(self.lag+1):
            for j in range(self.K):
              self.Nets[i][j][2].step()
              self.Nets[i][j][2].zero_grad()
          self.archive.append([list_of_cvs, Q[:,-1].cpu().detach().numpy(),Deltas_shifted.reshape((-1,1)).cpu().detach().clone().numpy()]) #[] Deltas_shifted.reshape((-1,1)).cpu().detach().clone().numpy()

        self.loss.append(loss)
        
    def Loss(self, R, ref):
        # Loss for control variate
        c_1 = self.c_1
        c_2 = self.c_2
        c_4 = self.c_4
        cv = (ref-R)
        expectation = torch.mean(ref)
        L =  c_1*torch.mean((R - expectation)**2) + c_2*torch.mean(torch.mul(cv,cv)) + c_4*(torch.var(R)**2)/(torch.var(ref)**2)
        #self.loss_archive = L
        return L
    

class Frozen_ULA():
    def __init__(self, X, trajectory_len, gamma, state_dim, f, lag, K, burn_in, burn_off, polynomial, a, b, lr = 0.001):
        super(Frozen_ULA, self).__init__()
        
        
        
        self.lr = lr
        self.states_history = []
        self.trajectory_len = trajectory_len # maximal trajectory length
        self.t = 0 # current step
        self.rands = []
        self.state_dim = state_dim
        self.f = f
        self.gamma = gamma
        self.state = X
        self.loss = []
        self.function = []
        self.archive = None
        #self.states_history.append(self.state)
        #self.function.append(self.f(self.state))

        self.lag = lag
        self.K = K
        self.Nets = []
        self.burn_in = burn_in
        self.burn_off = burn_off
        self.polynomial = polynomial
        self.H = np.zeros((self.trajectory_len, self.K)) 
        self.c_1 = a
        self.c_2 = b
    def clean(self):
        self.archive = None
        self.H = np.zeros((self.trajectory_len, self.K)) 
        self.function = []
        self.t = 0 # current step
        self.rands = []
        self.states_history = []
        

    def step(self):
        if self.t>=self.trajectory_len:
            return 0
         #entropy, rand, param, value
        rand = np.random.normal()
        
        self.get_H(self.t,rand)
        
        next_state = (1 - self.gamma)*self.state + np.sqrt(2*self.gamma)*rand
        self.state = next_state
        self.function.append(self.f(self.state))
        state = torch.Tensor([next_state])

        
        self.states_history.append(state.cpu().detach().numpy().reshape(-1))
        self.rands.append(rand)
        
        self.t=self.t+1
        
        return 1
    

    def init_regression(self, states_num):
        for i in range(self.lag+1):
            Nets_ = []
            for j in range(self.K):
                net = Approx_net(states_num).cpu()
                Nets_.append([net, nn.MSELoss(), optim.Adam(net.parameters(), lr=self.lr)])
            self.Nets.append(Nets_)
    
    def construct_Q(self, states, y):
        ### get actions, states and target value to rearrange them in convenient way
        #actions = actions.reshape((-1,self.action_dim))
        states = states.reshape((-1,self.state_dim))
        y = np.array(y).reshape((-1,1))
        q = np.hstack((states,y))
        q = torch.from_numpy(q)
        Q = torch.tensor(q,dtype = torch.float32, requires_grad=True).cpu()
        return Q
    
    
  
        
    def get_H(self, t, rand):
        for k in range(1, self.K+1):
            self.H[t,k-1] = self.polynomial(k,t,rand)

        #self.trajectory_len = self.trajectory_len + 1
        
    
    
        

    def a_net(self, x, q, k, Nets):
        # propagation of single state-action pair to get regression
        x_ = x
        output = Nets[q][k][0](x_)
        return output



    def finalize_trajectory(self, lag):
        a = []
        for i in self.states_history:
            a.append(float(i))
        self.states_history = np.array(a)
        self.rands = np.array(self.rands)
        self.function = np.array(self.function)
        Q = self.construct_Q(self.states_history, self.function)
        trajectory_len = Q.shape[0]
        list_of_cvs = [ [] for _ in range(self.lag+1) ]
        n_epoches = 1
        for epoch in range(n_epoches):
            for t in range(trajectory_len):
                c = 0
                for q in range(max(t-self.lag,0),t+1):
                    for k in range(1,self.K+1):
                        r = t - q
                        x = torch.tensor(Q[t-r,:self.state_dim].detach().clone(),dtype = torch.float32, requires_grad=True).detach().cpu()
                        x = torch.tensor(x,dtype = torch.float32, requires_grad=True).reshape((1,-1)).cpu()
                        if q!=trajectory_len-1:
                            b_2 = self.a_net(x, r, k-1, self.Nets)*self.H[q,k-1]
                        else:
                            b_2 = self.a_net(x, r, k-1, self.Nets)*0
                        if r!=lag:
                            b_2 = b_2.detach()
                        if k == 1:
                            list_of_cvs[r].append(b_2.detach().cpu().numpy())
                        else:
                            list_of_cvs[r][-1] = list_of_cvs[r][-1] + b_2.detach().cpu().numpy()
                        if c == 0:
                            c = 1
                            CV_ = b_2
                        else:
                            CV_ = torch.cat((b_2, CV_))
              
                CV1 = torch.sum(CV_, dim = 0, keepdim = True)
                if t == 0:
                    Deltas_shifted = Q[t,-1].detach().clone() - CV1#torch.tensor((Q[t,0]-CV1),dtype = torch.float32, requires_grad=True).reshape((1,-1)).cpu()
                else:
                    Deltas_shifted = torch.cat((Q[t,-1].detach().clone() - CV1, Deltas_shifted)).cpu()



            loss = self.Loss(Deltas_shifted, Q[:,-1].detach().clone())
            loss.backward()
            
            for i in range(self.lag+1):
                for j in range(self.K):
                    self.Nets[i][j][2].step()
                    self.Nets[i][j][2].zero_grad()
            self.archive = [list_of_cvs, Q[:,-1].cpu().detach().numpy(),Deltas_shifted.reshape((-1,1)).cpu().detach().clone().numpy()] #[] Deltas_shifted.reshape((-1,1)).cpu().detach().clone().numpy()

        self.loss.append(loss)
        
    def Loss(self, R, ref):
        # Loss for control variate
        c_1 = self.c_1
        c_2 = self.c_2
        cv = (ref-R)
        expectation = torch.mean(ref)
        L =  c_1*torch.mean(torch.mul(R-expectation,R - expectation)) + c_2*torch.mean(torch.mul(cv,cv))
        #self.loss_archive = L
        return L

class MDP():
    def __init__(self, env, agent, trajectory_len, gamma, state_dim, action_dim, CV=None):
        super(MDP, self).__init__()
        
        self.env = env
        self.agent = agent
        self.CV = CV
        self.states_history = []
        self.actions_history = []
        self.trajectory_len = trajectory_len # maximal trajectory length
        self.t = 0 # current step
        self.done = False # has env reach the final step
        self.log_probs = []
        self.rewards = []
        self.params = []
        self.rands = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        s =  env.reset()
        self.state = torch.Tensor([env.reset()]).cpu().detach().numpy().reshape(-1)
        self.loss = []
        self.states_history.append(self.state)
        
    def step(self):
        if (self.done) or (self.t>=self.trajectory_len):
            return 0
        action, log_prob, rand, param, extra_  = self.agent.get_action(self.state) #entropy, rand, param, value
        action = action.cpu().detach().numpy()
        self.actions_history.append(action)
        if self.CV != None:
            self.agent.parser(extra_)
            self.CV.get_H(self.t,rand)

        
        next_state, reward, done, _ = self.env.step(action.reshape((1,-1))[0])
        self.state = next_state
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        state = torch.Tensor([next_state])

        self.params.append(param)
        self.states_history.append(state.cpu().detach().numpy().reshape(-1))
        self.rands.append(rand)
        
        self.t=self.t+1
        self.done = done
        return 1
    
    def optimize_policy(self):
        loss = torch.stack(self.loss).mean()
        self.agent.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.agent.parameters(), 40)
        self.agent.optimizer.step()
        self.loss = []
    
    def finalize_trajectory(self):

        self.states_history = np.array(self.states_history)
        self.params = np.array(self.params)
        self.actions_history = np.array(self.actions_history)
        self.rands = np.array(self.rands)

        loss = self.agent.compute_loss(self.rewards, self.log_probs, self.state_dim, self.action_dim, self.t, self.gamma, self.states_history, self.actions_history, self.params, self.rands, self.CV)
        self.agent.clean_arrays()
        self.loss.append(loss)
        if CV == None:
            return np.sum(self.rewards), np.mean(self.rewards)
        else:
            return np.sum(self.rewards), np.mean(self.rewards)#, np.sum(c_rewards), np.mean(c_rewards)
    
    
    


def herm(k, l, xi):
  return special.hermitenorm(k)(xi)/np.sqrt(special.gamma(k+1))

def legd(k, l, xi):
  return special.legendre(k)(xi)*np.sqrt((2*k+1)/2)

class Approx_net(nn.Module):
    def __init__(self, input_dim):
        super(Approx_net, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        torch.nn.init.normal_(self.linear1.weight,mean = 0.0, std = 0.01) #0.01
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 256)
        torch.nn.init.normal_(self.linear2.weight,mean = 0.0, std = 0.01)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(256)
        self.linear21 = nn.Linear(256, 128)
        torch.nn.init.normal_(self.linear21.weight,mean = 0.0, std = 0.01)
        self.relu21 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(128)
        self.linear22 = nn.Linear(128, 128)
        torch.nn.init.normal_(self.linear22.weight,mean = 0.0, std = 0.01)
        self.relu22 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 1)
        torch.nn.init.normal_(self.linear3.weight,mean = 0.0, std = 0.01)
        

    def forward(self, x):
        x = self.bn1(self.relu1(self.linear1(x)))
        x = self.bn2(self.relu2(self.linear2(x)))
        x = self.bn3(self.relu21(self.linear21(x)))
        #x = self.linear22(self.relu21(self.linear21(self.relu2(self.linear2(self.relu1(self.linear1(x)))))))
        x = self.linear3(x)
  
        return x


class Approx_net2(nn.Module):
    def __init__(self, input_dim):
        super(Approx_net2, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        torch.nn.init.normal_(self.linear1.weight,mean = 0.0, std = 0.01) #0.01
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 128)
        torch.nn.init.normal_(self.linear2.weight,mean = 0.0, std = 0.01)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(128)
        self.linear21 = nn.Linear(128, 1)
        torch.nn.init.normal_(self.linear21.weight,mean = 0.0, std = 0.01)
        self.relu21 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(128)
        self.linear22 = nn.Linear(128, 128)
        torch.nn.init.normal_(self.linear22.weight,mean = 0.0, std = 0.01)
        self.relu22 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 1)
        torch.nn.init.normal_(self.linear3.weight,mean = 0.0, std = 0.01)
        

    def forward(self, x):
        
        x = self.bn1(self.relu1(self.linear1(x)))
        
        x = self.bn2(self.relu2(self.linear2(x)))
       
        #x = self.bn3(self.relu21(self.linear21(x)))
        #x = self.linear22(self.relu21(self.linear21(self.relu2(self.linear2(self.relu1(self.linear1(x)))))))
        x = self.linear3(x)
  
        return x



class CV():
    def __init__(self, lag, K, burn_in, burn_off, polynomial, trajectory_len, state_dim, action_dim, status, C_1, C_2, C_3, C_4, batch_size, lr = 0.001):
        super(CV, self).__init__()
        self.c_4 = C_4
        self.polynomial = polynomial
        self.K = K
        self.lag = lag
        self.burn_in = burn_in
        self.burn_off = burn_off
        self.lr = lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_trajectory_len = trajectory_len
        
        self.c_3 = C_3
        self.c_2 = C_2
        self.c_1 = C_1
        self.status = status

        
        self.H = np.zeros((self.max_trajectory_len, self.K))

        self.trajectory_len = 0
        self.Nets = []
        self.archive = None
        self.loss_archive = None
        self.batch_size = batch_size
    
    def init_regression(self, states_num, actions_num):
        for i in range(self.lag+1):
            Nets_ = []
            for j in range(self.K):
                net = Approx_net(states_num+actions_num).cpu()
                Nets_.append([net, nn.MSELoss(), optim.Adam(net.parameters(), lr=self.lr)])
            self.Nets.append(Nets_)
    
    def construct_Q(self, states,actions, y):
        ### get actions, states and target value to rearrange them in convenient way
        actions = actions.reshape((-1,self.action_dim))
        states = states.reshape((-1,self.state_dim))
        y = np.array(y).reshape((-1,1))
        q = np.hstack((states[:-1],actions,y))
        q = torch.from_numpy(q)
        Q = torch.tensor(q,dtype = torch.float32, requires_grad=True).cpu()
        return Q
    
    
  
        
    def get_H(self, t, rand):
        for k in range(1, self.K+1):
            self.H[t,k-1] = self.polynomial(k,t,rand)

        #self.trajectory_len = self.trajectory_len + 1
        
    
    
        

    def a_net(self, x, q, k, Nets):
        # propagation of single state-action pair to get regression
        x_ = x
        output = Nets[q][k][0](x_)
        return output
    
    
    
    
    def get_cv_correction(self, n_epoches, states, actions, y):
        A = 0
        Q = self.construct_Q(states,actions, y)
        trajectory_len = Q.shape[0]
        #print(Q.shape)
        if (trajectory_len - self.burn_in - self.burn_off <2):
          self.archive = [0, Q[:,-1].cpu().detach().numpy(), Q[:,-1].cpu().detach().numpy()]
          return y
        list_of_cvs = [ [] for _ in range(self.lag+1) ]
        n_epoches = 1
        batch_size = int(self.batch_size)
        while int((trajectory_len-self.burn_in - self.burn_off)%batch_size) == 1:
          batch_size = batch_size+1
        if trajectory_len -self.burn_in - self.burn_off < batch_size:
          batch_size = trajectory_len -self.burn_in - self.burn_off
        n_iter = int(math.ceil((trajectory_len-self.burn_in - self.burn_off)/batch_size))
        
        
        for epoch in range(n_epoches):
          for n_i in range(n_iter):
            c = 0
            for r in range(0, self.lag+1):
              for k in range(1,self.K+1):
                if n_i == n_iter-1  and (r>0 or self.burn_off>0):
                  batch = Q[self.burn_in+n_i*batch_size-r:-self.burn_off-r,:self.state_dim + self.action_dim].detach().clone()
                  #print('a')
                  #print(n_i)
                  #print(batch.shape)
                  b = self.a_net(batch, r, k-1, self.Nets)
                  A += torch.sum(b**2).reshape(-1)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:trajectory_len-self.burn_off-r,k-1], dtype = torch.float32).reshape(-1,1).cpu()
                  #print(b, 'b')
                  #print(torch.tensor(self.H[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,k-1], dtype = torch.float32).reshape(-1,1).cpu(), 'legd')
                elif n_i == n_iter-1  and (self.burn_off==0 and r==0):
                  batch = Q[self.burn_in+n_i*batch_size:,:self.state_dim + self.action_dim].detach().clone()
                  #print('b')
                  #print(n_i)
                  #print(batch.shape)
                  b = self.a_net(batch, r, k-1, self.Nets)
                  A += torch.sum(b**2).reshape(-1)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:trajectory_len-r,k-1], dtype = torch.float32).reshape(-1,1).cpu()
                  #print(b, 'b')
                  #print(torch.tensor(self.H[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,k-1], dtype = torch.float32).reshape(-1,1).cpu(), 'legd')
                else:
                  batch = Q[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,:self.state_dim + self.action_dim].detach().clone()
                  #print('c')
                  #print(n_i)
                  #print(batch.shape)
                  b = self.a_net(batch, r, k-1, self.Nets)
                  A += torch.sum(b**2).reshape(-1)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,k-1], dtype = torch.float32).reshape(-1,1).cpu()
                  #print(b, 'b')
                  #print(torch.tensor(self.H[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,k-1], dtype = torch.float32).reshape(-1,1).cpu(), 'legd')
                if k == 1:
                  list_of_cvs[r].append(b_2.detach().cpu().numpy())
                else:
                  list_of_cvs[r][-1] = list_of_cvs[r][-1] + b_2.detach().cpu().numpy()
                
                if c == 0:
                  c = 1
                  CV_ = b_2
                else:
                  CV_ +=b_2
            #print(CV_.shape)
            if n_i == 0:
              CV1 = CV_
            else:
              CV1 = torch.cat((CV1,CV_)).cpu()
            #print(Q[:,-1].detach().clone().shape)
            #print(CV1.shape)
          #print('aaa')
          Deltas_shifted = Q[:,-1].detach().clone().reshape(-1,1)  - torch.cat([torch.zeros(self.burn_in).reshape(-1,1).cpu(),CV1,torch.zeros(self.burn_off).reshape(-1,1).cpu()])

          loss = self.Loss(Deltas_shifted, Q[:,-1].detach().clone(), A)
          loss.backward()
          for i in range(self.lag+1):
            for j in range(self.K):
              self.Nets[i][j][2].step()
              self.Nets[i][j][2].zero_grad()
          self.archive = [list_of_cvs, Q[:,-1].cpu().detach().numpy(), Deltas_shifted.reshape((-1,1)).cpu().detach().clone().numpy()]
        return  Deltas_shifted.reshape((-1,1))

        



            
            
    
    
    
    def get_cv_correction_old(self, states, actions, y):
        n_epoches = 1
        Q = self.construct_Q(states,actions, y)
        trajectory_len = Q.shape[0]
        #print(Q.shape)
        if (trajectory_len - self.burn_in - self.burn_off <2):
          return 0
        list_of_cvs = [ [] for _ in range(self.lag+1) ]
        n_epoches = 1
        batch_size = int(self.batch_size)
        while int((trajectory_len-self.burn_in - self.burn_off)%batch_size) == 1:
          batch_size = batch_size+1
        if trajectory_len -self.burn_in - self.burn_off < batch_size:
          batch_size = trajectory_len -self.burn_in - self.burn_off
        n_iter = int(math.ceil((trajectory_len-self.burn_in - self.burn_off)/batch_size))
        #print(n_iter)
        #print(batch_size)
        for epoch in range(n_epoches):
          for n_i in range(n_iter):
            c = 0
            for r in range(0, self.lag+1):
              for k in range(1,self.K+1):
                if n_i == n_iter-1  and (r>0 or self.burn_off>0):
                  batch = Q[self.burn_in+n_i*batch_size-r:-self.burn_off-r,:self.state_dim + self.action_dim].detach().clone()
                  b = self.a_net(batch, r, k-1, self.Nets)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:trajectory_len-self.burn_off-r,k-1]).reshape(-1,1).cpu()
                elif n_i == n_iter-1  and (self.burn_off==0 and r==0):
                  batch = Q[self.burn_in+n_i*batch_size:,:self.state_dim + self.action_dim].detach().clone()
                  b = self.a_net(batch, r, k-1, self.Nets)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:trajectory_len-r,k-1]).reshape(-1,1).cpu()
                else:
                  batch = Q[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,:self.state_dim + self.action_dim].detach().clone()
                  b = self.a_net(batch, r, k-1, self.Nets)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,k-1]).reshape(-1,1).cpu()
                if k == 1:
                  list_of_cvs[r].append(b_2.detach().cpu().numpy())
                else:
                  list_of_cvs[r][-1] = list_of_cvs[r][-1] + b_2.detach().cpu().numpy()
                if c == 0:
                  c = 1
                  CV_ = b_2
                else:
                  CV_ +=b_2
            #print(CV_.shape)
            if n_i == 0:
              CV1 = CV_
            else:
              CV1 = torch.cat((CV1,CV_)).cpu()
            #print(Q[:,-1].detach().clone().shape)
            #print(CV1.shape)
          #print('aaa')
          Deltas_shifted = Q[:,-1].detach().clone().reshape(-1,1)  - torch.cat([torch.zeros(self.burn_in).reshape(-1,1).cpu(),CV1,torch.zeros(self.burn_off).reshape(-1,1).cpu()])
        
        self.archive = [list_of_cvs, Q[:,-1].cpu().detach().numpy(), Deltas_shifted.reshape((-1,1)).cpu().detach().clone().numpy()]
        return  Deltas_shifted.reshape((-1,1))
    
    
    
    
    def clean_cv(self, status):
        
        self.status = status

        
        self.H = np.zeros((self.max_trajectory_len, self.K))

        self.trajectory_len = 0
        
        self.archive = None
        self.loss_archive = None
    
    
    def Loss(self, R, ref, A):
        # Loss for control variate
        c_1 = self.c_1
        c_2 = self.c_2
        c_3 = self.c_3
        c_4 = self.c_4
        N = R.shape[0]
        cv = (ref-R)
        expectation = torch.mean(ref)
        L =  c_1*torch.mean((R - expectation)**2) + c_2*torch.mean(torch.mul(cv,cv)) + c_3*A/N + c_4*(torch.var(R)/torch.var(ref))#torch.mean(torch.mul(R-expectation,R - expectation))
        #self.loss_archive = L
        return L

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartPoleEnvSmooth(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, threshold = 100.0):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.a_threshold = threshold # to tune

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        #high_a = np.array(-self.a_threshold, self.a_threshold, dtype=np.float32)

        self.action_space = spaces.Box(-self.a_threshold, self.a_threshold, shape=(1,), dtype=np.float32)#spaces.Discrete(2) ### Changes in action space
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #err_msg = "%r (%s) invalid" % (action, type(action))
        #assert self.action_space.contains(action), err_msg
        action = float(action)
        #if action>self.a_threshold:
        #    action = self.a_threshold
        #if action<-self.a_threshold:
        #    action = -self.a_threshold
        action = -np.tanh(action+2)*np.tanh(action-2)
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag*action#self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        xi_0 = 0.1*np.random.randn()
        xi_2 = 0.1*np.random.randn()
        x_dot = (1+xi_0) * x_dot
        theta_dot = (1+xi_2) * theta_dot
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        sigma1 = self.x_threshold
        sigma2 = self.theta_threshold_radians
        if not done:
            reward = np.cos((np.pi/2)*self.state[2]/sigma2)*np.cos((np.pi/2)*self.state[0]/sigma1)#1
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = np.cos((np.pi/2)*self.state[2]/sigma2)*np.cos((np.pi/2)*self.state[0]/sigma1)#1
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# Buffer replay and Q_prop
import random
import numpy as np
from collections import deque

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def sample_sequence(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        min_start = len(self.buffer) - batch_size
        start = np.random.randint(0, min_start)

        for sample in range(start, start + batch_size):
            state, action, reward, next_state, done = sample
            #state, action, reward, next_state, done = sample
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)
    
    def sample_sequence_ET(self, batch_size, E):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        min_start = len(self.buffer) - batch_size
        start = len(self.buffer)-1
        
        while E>0:
            state, action, reward, next_state, done = self.buffer[start]
            #state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
            start-=1
            if done == True:
                E-=1

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)


    def __len__(self):
        return len(self.buffer)



class Q_net(nn.Module):
    def __init__(self, input_dim):
        super(Approx_net, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        torch.nn.init.normal_(self.linear1.weight,mean = 0.0, std = 0.1) #0.01
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        torch.nn.init.normal_(self.linear2.weight,mean = 0.0, std = 0.1)
        self.relu2 = nn.ReLU()
        self.linear21 = nn.Linear(128, 128)
        torch.nn.init.normal_(self.linear21.weight,mean = 0.0, std = 0.1)
        self.relu21 = nn.ReLU()
        self.linear22 = nn.Linear(128, 128)
        torch.nn.init.normal_(self.linear22.weight,mean = 0.0, std = 0.1)
        self.relu22 = nn.ReLU()
        self.linear3 = nn.Linear(128, 1)
        torch.nn.init.normal_(self.linear3.weight,mean = 0.0, std = 1)
        

    def forward(self, x):
        x = self.linear22(self.relu21(self.linear21(self.relu2(self.linear2(self.relu1(self.linear1(x)))))))
        x = self.linear3(self.relu22(x))
  
        return x


class MDP_Q_prop():
    def __init__(self, env, agent, trajectory_len, gamma, state_dim, action_dim, Buff, CV=None):
        super(MDP_Q_prop, self).__init__()
        self.Buff = Buff
        self.env = env
        self.agent = agent
        self.CV = CV
        self.states_history = []
        self.actions_history = []
        self.trajectory_len = trajectory_len # maximal trajectory length
        self.t = 0 # current step
        self.done = False # has env reach the final step
        self.log_probs = []
        self.rewards = []
        self.params = []
        self.rands = []
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.state = torch.Tensor([env.reset()]).cpu().detach().numpy().reshape(-1)
        self.loss = []
        self.states_history.append(self.state)
        
    def step(self):
        if (self.done) or (self.t>=self.trajectory_len):
            return 0
        action, log_prob, rand, param, extra_  = self.agent.get_action(self.state) #entropy, rand, param, value
        action = action.cpu().detach().numpy()
        self.actions_history.append(action)
        if self.CV != None:
            self.agent.parser(extra_)
            self.CV.get_H(self.t,rand)
            
        next_state, reward, done, _ = self.env.step(action.reshape((1,-1))[0])
        self.Buff.push(self.state, action, reward, next_state, done)
        self.state = next_state
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        state = torch.Tensor([next_state])

        self.params.append(param)
        self.states_history.append(state.cpu().detach().numpy().reshape(-1))
        self.rands.append(rand)
        
        self.t=self.t+1
        self.done = done
        return 1
    def redo(self):
        self.done = False


    def optimize_policy(self):
        loss = torch.stack(self.loss).mean()
        self.agent.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.agent.parameters(), 40)
        self.agent.optimizer.step()
        self.loss = []
    
    
    def finalize_trajectory(self):

        self.states_history = np.array(self.states_history)
        self.params = np.array(self.params)
        self.actions_history = np.array(self.actions_history)
        self.rands = np.array(self.rands)

        loss = self.agent.compute_loss(self.rewards, self.log_probs, self.state_dim, self.action_dim, self.t, self.gamma, self.states_history, self.actions_history, self.params, self.rands, self.CV)
        self.agent.clean_arrays()
        self.loss.append(loss)
        if CV == None:
            return np.sum(self.rewards), np.mean(self.rewards)
        else:
            return np.sum(self.rewards), np.mean(self.rewards)#, np.sum(c_rewards), np.mean(c_rewards)
    
#    def optimize_Q_w(self, E):
#        S, A, R, S, D = self.Buff.sample(T)
#        for i in range(T):
#            y - 

    """
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Continuous version by Ian Danforth
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -2.0
        self.max_action = 2.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)
        sigma1 = self.x_threshold
        sigma2 = self.theta_threshold_radians
        if not done:
            reward = np.cos((np.pi/2)*self.state[2]/sigma2)*np.cos((np.pi/2)*self.state[0]/sigma1)#1
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = np.cos((np.pi/2)*self.state[2]/sigma2)*np.cos((np.pi/2)*self.state[0]/sigma1)#1
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()


"""classic Acrobot task"""



from gym.utils import seeding

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class ContinuousAcrobotEnv(core.Env):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    dt = .2

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    #AVAIL_TORQUE = [-1., 0., +1]
   
    torque_noise_max = 0.

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self):
        self.max_action= 2.0
        self.min_action = -2.0
        self.viewer = None
        
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.state = None
        self.seed()
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        return self._get_ob()

    def step(self, a):
        s = self.state
        assert self.action_space.contains(a), \
            "%r (%s) invalid" % (a, type(a))
        torque = a

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = -np.tanh((-np.cos(s[0]) - np.cos(s[1] + s[0])-2))**2 if not terminal else 0.
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        s = self.state
        return np.array([np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]])

    def _terminal(self):
        s = self.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
                (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None

        p1 = [-self.LINK_LENGTH_1 *
              np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]- pi/2, s[0]+s[1]-pi/2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range
    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    Args:
        x: scalar
    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi, ti)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0


    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout








class MyAcrobotEnv(core.Env):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    dt = .2

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    #AVAIL_TORQUE = [-1., 0., +1]
   
    torque_noise_max = 0.

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self):
        self.max_action= 2.0
        self.min_action = -2.0
        self.viewer = None
        
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.state = None
        self.seed()
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        return self._get_ob()

    def step(self, a):
        s = self.state
        assert self.action_space.contains(a), \
            "%r (%s) invalid" % (a, type(a))
        torque = a

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = -1 if not terminal else 0.
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        s = self.state
        return np.array([np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]])

    def _terminal(self):
        s = self.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
                (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None

        p1 = [-self.LINK_LENGTH_1 *
              np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]- pi/2, s[0]+s[1]-pi/2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range
    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    Args:
        x: scalar
    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi, ti)``
        y0: initial state vector
        t: sample times
        args: additional arguments passed to the derivative function
        kwargs: additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0


    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] != 'H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]

class MyDiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed()
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        s0 = int(self.s)//4
        s1 = int(self.s)%4
        #s2 = np.zeros(8)
        #s2[s0] = 1
        #s2[s1+4] = 1
        s2 = np.array([s0,s1])
        return s2

    def step(self, a):
        
        a = int(a[0])
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a

        s0 = int(s)//4
        s1 = int(s)%4
        #s2 = np.zeros(8)
        #print(s0,s1,s)
        #s2[s0] = 1
        #s2[s1+4] = 1
        s2 = np.array([s0,s1])
        return (s2, r, d, {"prob": p})
class MyFrozenLakeEnv(MyDiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage, so
    it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", is_slippery=False):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b'GH'
            reward = float(newletter == b'G')
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append((
                                    1. / 3.,
                                    *update_probability_matrix(row, col, b)
                                ))
                        else:
                            li.append((
                                1., *update_probability_matrix(row, col, a)
                            ))

        super(MyFrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = gym.utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
    

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()





class CV2():
    def __init__(self, lag, K, burn_in, burn_off, polynomial, trajectory_len, state_dim, action_dim, status, C_1, C_2, C_3, C_4, batch_size, lr = 0.001):
        super(CV2, self).__init__()
        self.c_4 = C_4
        self.polynomial = polynomial
        self.K = K
        self.lag = lag
        self.burn_in = burn_in
        self.burn_off = burn_off
        self.lr = lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_trajectory_len = trajectory_len
        
        self.c_3 = C_3
        self.c_2 = C_2
        self.c_1 = C_1
        self.status = status

        
        self.H = np.zeros((self.max_trajectory_len, self.K))

        self.trajectory_len = 0
        self.Nets = []
        self.archive = None
        self.loss_archive = None
        self.batch_size = batch_size
    
    def init_regression(self, states_num, actions_num):
        for i in range(self.lag+1):
            Nets_ = []
            for j in range(self.K):
                net = Approx_net2(states_num+actions_num).cpu()
                Nets_.append([net, nn.MSELoss(), optim.Adam(net.parameters(), lr=self.lr)])
            self.Nets.append(Nets_)
    
    def construct_Q(self, states,actions, y):
        ### get actions, states and target value to rearrange them in convenient way
        actions = actions.reshape((-1,self.action_dim))
        states = states.reshape((-1,self.state_dim))
        y = np.array(y).reshape((-1,1))
        q = np.hstack((states[:-1],actions,y))
        q = torch.from_numpy(q)
        Q = torch.tensor(q,dtype = torch.float32, requires_grad=True).cpu()
        return Q
    
    
  
        
    def get_H(self, t, rand):
        for k in range(1, self.K+1):
            self.H[t,k-1] = self.polynomial(k,t,rand)

        #self.trajectory_len = self.trajectory_len + 1
        
    
    
        

    def a_net(self, x, q, k, Nets):
        # propagation of single state-action pair to get regression
        x_ = x
        output = Nets[q][k][0](x_)
        return output
    
    
    
    
    def get_cv_correction(self, n_epoches, states, actions, y):
        A = 0
        Q = self.construct_Q(states,actions, y)
        trajectory_len = Q.shape[0]
        #print(Q.shape)
        if (trajectory_len - self.burn_in - self.burn_off <2):
          self.archive = [0, Q[:,-1].cpu().detach().numpy(), Q[:,-1].cpu().detach().numpy()]
          return y
        list_of_cvs = [ [] for _ in range(self.lag+1) ]
        n_epoches = 1
        batch_size = int(self.batch_size)
        while int((trajectory_len-self.burn_in - self.burn_off)%batch_size) == 1:
          batch_size = batch_size+1
        if trajectory_len -self.burn_in - self.burn_off < batch_size:
          batch_size = trajectory_len -self.burn_in - self.burn_off
        n_iter = int(math.ceil((trajectory_len-self.burn_in - self.burn_off)/batch_size))
        
        
        for epoch in range(n_epoches):
          for n_i in range(n_iter):
            c = 0
            for r in range(0, self.lag+1):
              for k in range(1,self.K+1):
                if n_i == n_iter-1  and (r>0 or self.burn_off>0):
                  batch = Q[self.burn_in+n_i*batch_size-r:-self.burn_off-r,:self.state_dim + self.action_dim].detach().clone()
                  #print('a')
                  #print(n_i)
                  #print(batch.shape)
                  b = self.a_net(batch, r, k-1, self.Nets)
                  A += torch.sum(b**2).reshape(-1)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:trajectory_len-self.burn_off-r,k-1], dtype = torch.float32).reshape(-1,1).cpu()
                  #print(b, 'b')
                  #print(torch.tensor(self.H[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,k-1], dtype = torch.float32).reshape(-1,1).cpu(), 'legd')
                elif n_i == n_iter-1  and (self.burn_off==0 and r==0):
                  batch = Q[self.burn_in+n_i*batch_size:,:self.state_dim + self.action_dim].detach().clone()
                  #print('b')
                  #print(n_i)
                  #print(batch.shape)
                  b = self.a_net(batch, r, k-1, self.Nets)
                  A += torch.sum(b**2).reshape(-1)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:trajectory_len-r,k-1], dtype = torch.float32).reshape(-1,1).cpu()
                  #print(b, 'b')
                  #print(torch.tensor(self.H[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,k-1], dtype = torch.float32).reshape(-1,1).cpu(), 'legd')
                else:
                  batch = Q[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,:self.state_dim + self.action_dim].detach().clone()
                  #print('c')
                  #print(n_i)
                  #print(batch.shape)
                  b = self.a_net(batch, r, k-1, self.Nets)
                  A += torch.sum(b**2).reshape(-1)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,k-1], dtype = torch.float32).reshape(-1,1).cpu()
                  #print(b, 'b')
                  #print(torch.tensor(self.H[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,k-1], dtype = torch.float32).reshape(-1,1).cpu(), 'legd')
                if k == 1:
                  list_of_cvs[r].append(b_2.detach().cpu().numpy())
                else:
                  list_of_cvs[r][-1] = list_of_cvs[r][-1] + b_2.detach().cpu().numpy()
                
                if c == 0:
                  c = 1
                  CV_ = b_2
                else:
                  CV_ +=b_2
            #print(CV_.shape)
            if n_i == 0:
              CV1 = CV_
            else:
              CV1 = torch.cat((CV1,CV_)).cpu()
            #print(Q[:,-1].detach().clone().shape)
            #print(CV1.shape)
          #print('aaa')
          Deltas_shifted = Q[:,-1].detach().clone().reshape(-1,1)  - torch.cat([torch.zeros(self.burn_in).reshape(-1,1).cpu(),CV1,torch.zeros(self.burn_off).reshape(-1,1).cpu()])

          loss = self.Loss(Deltas_shifted, Q[:,-1].detach().clone(), A)
          
          loss.backward()
          for i in range(self.lag+1):
            for j in range(self.K):
              self.Nets[i][j][2].step()
              self.Nets[i][j][2].zero_grad()
          self.archive = [list_of_cvs, Q[:,-1].cpu().detach().numpy(), Deltas_shifted.reshape((-1,1)).cpu().detach().clone().numpy()]
        return  Deltas_shifted.reshape((-1,1))

        



            
            
    
    
    
    def get_cv_correction_old(self, states, actions, y):
        n_epoches = 1
        Q = self.construct_Q(states,actions, y)
        trajectory_len = Q.shape[0]
        #print(Q.shape)
        if (trajectory_len - self.burn_in - self.burn_off <2):
          return 0
        list_of_cvs = [ [] for _ in range(self.lag+1) ]
        n_epoches = 1
        batch_size = int(self.batch_size)
        while int((trajectory_len-self.burn_in - self.burn_off)%batch_size) == 1:
          batch_size = batch_size+1
        if trajectory_len -self.burn_in - self.burn_off < batch_size:
          batch_size = trajectory_len -self.burn_in - self.burn_off
        n_iter = int(math.ceil((trajectory_len-self.burn_in - self.burn_off)/batch_size))
        #print(n_iter)
        #print(batch_size)
        for epoch in range(n_epoches):
          for n_i in range(n_iter):
            c = 0
            for r in range(0, self.lag+1):
              for k in range(1,self.K+1):
                if n_i == n_iter-1  and (r>0 or self.burn_off>0):
                  batch = Q[self.burn_in+n_i*batch_size-r:-self.burn_off-r,:self.state_dim + self.action_dim].detach().clone()
                  b = self.a_net(batch, r, k-1, self.Nets)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:trajectory_len-self.burn_off-r,k-1]).reshape(-1,1).cpu()
                elif n_i == n_iter-1  and (self.burn_off==0 and r==0):
                  batch = Q[self.burn_in+n_i*batch_size:,:self.state_dim + self.action_dim].detach().clone()
                  b = self.a_net(batch, r, k-1, self.Nets)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:trajectory_len-r,k-1]).reshape(-1,1).cpu()
                else:
                  batch = Q[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,:self.state_dim + self.action_dim].detach().clone()
                  b = self.a_net(batch, r, k-1, self.Nets)
                  b_2 = b*torch.tensor(self.H[self.burn_in+n_i*batch_size-r:self.burn_in+(n_i+1)*batch_size-r,k-1]).reshape(-1,1).cpu()
                if k == 1:
                  list_of_cvs[r].append(b_2.detach().cpu().numpy())
                else:
                  list_of_cvs[r][-1] = list_of_cvs[r][-1] + b_2.detach().cpu().numpy()
                if c == 0:
                  c = 1
                  CV_ = b_2
                else:
                  CV_ +=b_2
            #print(CV_.shape)
            if n_i == 0:
              CV1 = CV_
            else:
              CV1 = torch.cat((CV1,CV_)).cpu()
            #print(Q[:,-1].detach().clone().shape)
            #print(CV1.shape)
          #print('aaa')
          Deltas_shifted = Q[:,-1].detach().clone().reshape(-1,1)  - torch.cat([torch.zeros(self.burn_in).reshape(-1,1).cpu(),CV1,torch.zeros(self.burn_off).reshape(-1,1).cpu()])
        
        self.archive = [list_of_cvs, Q[:,-1].cpu().detach().numpy(), Deltas_shifted.reshape((-1,1)).cpu().detach().clone().numpy()]
        return  Deltas_shifted.reshape((-1,1))
    
    
    
    
    def clean_cv(self, status):
        
        self.status = status

        
        self.H = np.zeros((self.max_trajectory_len, self.K))

        self.trajectory_len = 0
        
        self.archive = None
        self.loss_archive = None
    
    
    def Loss(self, R, ref, A):
        # Loss for control variate
        c_1 = self.c_1
        c_2 = self.c_2
        c_3 = self.c_3
        c_4 = self.c_4
        N = R.shape[0]
        
        cv = (ref-R)
        
        expectation = torch.mean(ref)
        
        L =  c_1*torch.mean((R - expectation)**2) + + c_3*A/N #torch.mean(torch.mul(R-expectation,R - expectation))
        #self.loss_archive = L

        return L


