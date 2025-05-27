import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal
from .base_vectorised import BaseVectorisedNetwork, VectorisedLinear

from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.distributions.independent import Independent
from torch.distributions.transforms import AffineTransform

class BaseVectorisedActor(BaseVectorisedNetwork):
    def __init__(self, obs_dims, model_info, ensemble_num=1, **kwargs):

        self.is_actor = True
        self.final_activation = None

        ##doesnt matter we put it in DDPGVectorisedNetwork
       #if kwargs['algo_name'] in ['td3_bc_n','ddpg']:
       #    self.final_activation = getattr(torch.nn,'Tanh',None)

        super().__init__(obs_dims=obs_dims, model_info=model_info,
                            ensemble_num=ensemble_num,**kwargs)




class DiscreteVectorisedActor(BaseVectorisedActor):

    ''' Discrete actor produces a categorical distribution assigns a probability to each action that can be taken '''
    def __init__(self, obs_dims, n_actions, model_info, ensemble_num=1, **kwargs):

        self.final_layer_dim = n_actions
        self.type = 'Discrete'

        super().__init__(obs_dims=obs_dims, model_info=model_info,
                            ensemble_num=ensemble_num,**kwargs)




       ###final layer
        self.policy = self.construct_model(model_info, add_final=True)



    def forward(self, state):

        x = super().forward(state)
        #action_probs = F.softmax(self.logits_layer(x),dim=self.cat_dim)


        ##need to check if i need to add a softmax here
        action_probs = self.policy(x)
        action_dist = Categorical(action_probs)

        return action_dist

class GaussianVectorisedActor(BaseVectorisedActor):

    def __init__(self, obs_dims, action_dims, model_info, ensemble_num=1, 
                    log_std_min=-20, log_std_max=2, transform=True, **kwargs):

        super().__init__(obs_dims=obs_dims, model_info=model_info, ensemble_num=ensemble_num,**kwargs)


        self.type = 'Gaussian'
        self.transform = transform

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_val = torch.tensor(kwargs.get('max_val',np.inf))
        self.min_val = torch.tensor(kwargs.get('min_val',-np.inf))

        self.final_layer_dim = action_dims


        #separate model for mean and std
        self.mean_model = self.construct_model(model_info, add_final=True)
        self.log_std_model = self.construct_model(model_info, add_final=True)

    def forward(self, state):

       #x = super().forward(state)

        mean = self.mean_model(state)
        log_std = self.log_std_model(state)


        log_std = torch.clamp(log_std,self.log_std_min,self.log_std_max)

        std = log_std.exp()

        if self.transform:
            dist = TransformedDistribution(Independent(Normal(mean, std), 1), [
                    TanhTransform(cache_size=1)])
        else:
            dist = Normal(mean, std)
        return dist


    def log_prob(self, state, action, epsilon=1e-6):

        dist = self(state)

        action = action.clamp(self.min_val + epsilon, self.max_val - epsilon)   # to prevent log 0
        log_probs = dist.log_prob(action)

        if not self.transform:
            log_probs = log_probs.sum(dim=-1)

        return log_probs

    def dist_stats(self, dist):

        if self.transform:
            samples = dist.sample((1000,))
            return samples.mean(dim=0), samples.std(dim=0)
        else:
            return dist.mean, dist.stddev

    def sample(self, state, epsilon=1e-6, reparameterise=True, dataset_actions=None, deterministic=False,**kwargs):

        dist = self(state)

        if deterministic:
            ##empty dict as a placeholder
            mean, std = self.dist_stats(dist)
            return {'action':mean,'action_std':std}

        if reparameterise:
            sample_action = dist.rsample()
        else:
            sample_action = dist.sample()


        log_prob = self.log_prob(state, sample_action)
        action_info = {'action':sample_action,'log_prob':log_prob}

        if dataset_actions is not None:
            action_info['bc_log_prob'] = dist.log_prob(dataset_actions) #.sum(self.cat_dim,keepdim=True)

        return action_info

class DDPGVectorisedActor(BaseVectorisedActor):

    ''' DDPG vectorised actor generates a single action given an observation 'deterministic stochastic policy' '''
    def __init__(self, obs_dims, action_dims, model_info, ensemble_num=1, **kwargs):

        self.final_layer_dim = action_dims


        super().__init__(obs_dims=obs_dims, model_info=model_info,
                            ensemble_num=ensemble_num,**kwargs)



        self.final_activation = nn.Tanh
        self.policy = self.construct_model(model_info, add_final=True)
        self.max_val = torch.tensor(kwargs['max_val'])
        self.min_val = torch.tensor(kwargs['min_val'])


    def forward(self, state, **kwargs):

        x = self.policy(state)

        action_info = {}
        action_info['action'] = x*self.max_val
        return action_info


class GoalsDDPGActor(DDPGVectorisedActor):

    def __init__(self, obs_dims, action_dims, model_info, ensemble_num=1, **kwargs):
        self.final_layer_dim = action_dims


        super().__init__(obs_dims=2*obs_dims, action_dims=action_dims,
                        model_info=model_info, ensemble_num=ensemble_num,**kwargs)

        self.policy = self.construct_model(model_info, add_final=True)

    def forward(self, state, goal, **kwargs):
        x = torch.cat([state,goal],1)

        action_info = {}
        action_info['action'] = self.policy(x)
        return action_info
