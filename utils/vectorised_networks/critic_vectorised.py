import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_vectorised import BaseVectorisedNetwork, VectorisedLinear

import time
class BaseCriticNetwork(BaseVectorisedNetwork):


    def __init__(self, obs_dims, model_info, ensemble_num=1, **kwargs):

    
        super().__init__(obs_dims=obs_dims, model_info=model_info,
                            ensemble_num=ensemble_num, **kwargs)

        self.critic_factor = kwargs.get('critic_factor')
        self.is_critic = True
        self.final_activation= getattr(torch.nn,model_info['critic_final_activation'],None)

        ##no activation function

       #for critic_idx in range(self.critic_factor):
       #    action_value_model = self.construct_model(model_info, add_final=True)
       #    self.critic_ensemble.append(action_value_model)

        self.action_value_model = self.construct_model(model_info, add_final=True)

    def forward(self, state, action=None):

        if action is not None: #This will only be true for critics
        ##state and action get broadcasted to the ensemble
           #x = torch.cat([state,action],self.cat_dim)
            x = torch.cat([state,action],-1)
        else:
            x = state
        
        critic_values = self.action_value_model(x)
        return critic_values



class DiscreteVectorisedCritic(BaseCriticNetwork):

    ''' Discrete critic produces a set of action values for all possible actions '''

    def __init__(self, obs_dims, n_actions, model_info, ensemble_num=1, **kwargs):

        self.final_layer_dim = n_actions

        super().__init__(obs_dims=obs_dims, model_info=model_info, ensemble_num=ensemble_num)


        ##no activation function
       #self.action_value_model, self.in_dims = self.construct_model(model_info, base=False, add_final=True) 

       #if self.ensemble_num == 1:
       #    self.value_layer = nn.Linear(fc2_dims, n_actions)
       #else:
       #    self.value_layer = VectorisedLinear(fc2_dims, n_actions, ensemble_num)

    def forward(self, state):

        x = super().forward(state)
        action_values = self.action_value_model(x)
        #action_values = self.value_layer(x)

        return action_values


class ContinuousVectorisedCritic(BaseCriticNetwork):

    ''' Continuous critic produces an output for a given state action pair '''

    def __init__(self, obs_dims, action_dims, model_info, ensemble_num=1, **kwargs):

        self.final_layer_dim = 1

        super().__init__(obs_dims=obs_dims, model_info=model_info,
                            ensemble_num=ensemble_num, action_dims=action_dims, **kwargs)


       
       #    self.action_value_layer = nn.Linear(fc2_dims, 1)
       #else:
       #    self.action_value_layer = VectorisedLinear(fc2_dims, 1, ensemble_num)

    def forward(self, state, action):

        values_list = super().forward(state,action)
       #action_value = self.action_value_model(x)

        return values_list

class VectorisedStateValue(BaseCriticNetwork):
    ''' Value function produces a value for the state regardless of action chosen'''

    
    def __init__(self, obs_dims, model_info, ensemble_num=1, **kwargs):

        self.final_layer_dim = 1

        super().__init__(obs_dims=obs_dims, model_info=model_info,
                            ensemble_num=ensemble_num, **kwargs)

        ##no activation function
       #self.state_value_model,self.in_dims = self.construct_model(model_info, base=False, add_final=True)

       #if self.ensemble_num == 1:
       #    self.state_value_layer = nn.Linear(fc2_dims, 1)
       #else:
       #    self.state_value_layer = VectorisedLinear(fc2_dims, 1, ensemble_num)

    def forward(self, state):

        state_value = super().forward(state)

        return state_value
