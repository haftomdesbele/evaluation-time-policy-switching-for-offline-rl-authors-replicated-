import numpy as np
import torch
import sys


class ReplayBuffer:

    def __init__(self,batch_size,obs_dims=None,mem_size=None,dataset=None,n_envs=1):
        ''' Assume that if a dataset is present we are just creating a buffer
            for the dataset and not comibning dataset data with online data
            which you can create another buffer for'''

        self.batch_size = batch_size

        if dataset:
            self.mem_cntr = self.mem_size = dataset['observations'].shape[0]
            #self.store_offline_data(dataset)
        else:
            self.mean = 0
            self.std = 1
            self.n_envs=n_envs
            self.mem_size = mem_size
            self.mem_cntr = 0
            self.state_memory = np.zeros((self.mem_size,obs_dims),dtype=float)
            self.next_state_memory = np.zeros((self.mem_size,obs_dims),dtype=float)
            self.reward_memory = np.zeros(self.mem_size,dtype=float)
            self.terminal_memory = np.zeros(self.mem_size,dtype=bool)

    def __len__(self):
        return min(self.mem_size,self.mem_cntr)

    def reset_buffer(self,obs_dims):

        self.state_memory = np.zeros((self.mem_size,obs_dims),dtype=float)
        self.next_state_memory = np.zeros((self.mem_size,obs_dims),dtype=float)
        self.reward_memory = np.zeros(self.mem_size,dtype=float)
        self.terminal_memory = np.zeros(self.mem_size,dtype=bool)
        self.mem_cntr = 0

    def to(self, device):
        self.device = device

    def store_transition(self,state,next_state,action,reward,done,mem_idxs=None):

        if getattr(self,'n_envs',None) is None:
            self.n_envs = 1

        ##use mem_idxs directly from multiagent buffer if there is one instead 
        ##of relying on the individual buffers to be in sync makes things simpler
        if mem_idxs==None:
            num_samples = state.shape[0] - 1
            mem_idxs = [i%self.mem_size for i in range(self.mem_cntr,self.mem_cntr+self.n_envs+num_samples)]
            self.mem_cntr += self.n_envs + num_samples

        self.state_memory[mem_idxs] = state
        self.next_state_memory[mem_idxs] = next_state
        self.action_memory[mem_idxs] = action
        self.reward_memory[mem_idxs] = reward
        self.terminal_memory[mem_idxs] = done
    

    def sample(self,sample_range=None,min_idx=None,max_idx=None,batch_size=None,rng=None,entire=False, batch_idx=None,raw_states=False,next_actions=False):

        if batch_idx is None:
            #mem_size = len(self)
            batch_size = self.batch_size if batch_size is None else batch_size

            if sample_range is None:
                min_idx = 0 if min_idx is None else min_idx
                mem_size = min(len(self),max_idx) if max_idx is not None else len(self)
                sample_range = np.arange(min_idx,mem_size)

            if entire:
                batch_idx = sample_range
            else:
                if rng is not None:
                    batch_idx = rng.choice(sample_range,batch_size,replace=False)
                else:
                    batch_idx = np.random.default_rng().choice(sample_range,
                                                               batch_size,
                                                               replace=False)

        states = self.state_memory[batch_idx]
        next_states = self.next_state_memory[batch_idx]
        actions = torch.tensor(self.action_memory[batch_idx],dtype=torch.float).to(self.device)
        rewards = torch.tensor(self.reward_memory[batch_idx],dtype=torch.float).to(self.device).unsqueeze(0)
        done_batch = torch.tensor(self.terminal_memory[batch_idx],dtype=bool).to(self.device).unsqueeze(0)

        if raw_states:
            states, next_states = self.raw_states(states, next_states)

        states = torch.tensor(states,dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states,dtype=torch.float).to(self.device)


        if next_actions:
            next_actions = torch.tensor(self.next_action_memory[batch_idx],dtype=torch.float).to(self.device)
            return states,next_states,actions,next_actions,rewards,done_batch, batch_idx
        else:
            return states,next_states,actions,rewards,done_batch, batch_idx

        
    def store_offline_data(self,dataset,shuffle=False,normalise_state=True, task=''):
        shuffled_idxs = np.arange(len(self))
        if shuffle:
            np.random.shuffle(shuffled_idxs)
        else:
            self.state_memory = dataset['observations'][shuffled_idxs]
            self.next_state_memory = dataset['next_observations'][shuffled_idxs]
            self.reward_memory = dataset['rewards'][shuffled_idxs]
            self.terminal_memory = dataset['terminals'][shuffled_idxs]
            self.action_memory = dataset['actions'][shuffled_idxs]
            self.next_action_memory = dataset['actions'][1:]

        if dataset.get('timeouts',None) is not None:
            self.timeouts = dataset['timeouts'][shuffled_idxs]
        else:
            self.timeouts = [None]*len(self)

        if normalise_state:
            self.normalise_states()

        if 'umaze-diverse' in task:
            ## Get approximate original goal by looking at rewards
            goal_reached = self.state_memory[:, :2][self.reward_memory == 1]
            orig_goal = np.mean(goal_reached, 0).reshape(1, -1)
            orig_goals = np.repeat(orig_goal, len(self), 0)
            threshold = 0.5
            xy = self.state_memory[:, :2]
            distances = np.linalg.norm(xy - orig_goals, axis=-1)
            at_goal = distances < threshold
            self.reward_memory = at_goal
            self.terminal_memory = self.reward_memory



        self.create_state_trajectory(task)

        if 'ant' in task:
            self.reward_memory = (self.reward_memory - 0.5)*4



    def create_state_trajectory(self, task, bc_agent=None):
        return_list = []
        traj_list = []

        
        j = 0 
        traj_gap = []
        dist_covered = 0 

        goal_reached = self.state_memory[:, :2][self.terminal_memory == True]
        orig_goal = np.mean(goal_reached, 0).reshape(1, -1)
        orig_goals = np.repeat(orig_goal, len(self), 0)
        threshold = 0.5
        xy = self.state_memory[:, :2]
        distances = np.linalg.norm(xy - orig_goals, axis=-1)

        for i in range(len(self)):
            j += self.reward_memory[i]

            dist_covered += distances[i]

            try:
                dist_bool = (np.linalg.norm(self.state_memory[i + 1,:2] - self.next_state_memory[i, :2])) > 1e-6
            except IndexError:
                dist_bool = False

            if self.terminal_memory[i] or self.timeouts[i] or dist_bool:
                return_list.append(j)

                traj_gap.append(dist_covered)
                dist_covered = 0

                j = 0


        if 'ant' in task:

            traj_gap_arr = np.array(traj_gap)
            traj_std = traj_gap_arr.std() 

            self.std_scale = (1-np.exp(-3300/traj_std))
            if 'large-play' in task:
                self.std_scale *= 0.5
            
        else:
            return_arr = np.array(return_list)
            return_std = return_arr.std()
            return_max = return_arr.max()
            norm_std = np.abs(return_std/(return_max))

            alpha = 0.075   #0.025,0.15
            self.std_scale = (1-np.exp(-alpha/norm_std))

        print(task)
        print(self.std_scale)


    def normalise_states(self,eps=1e-3):

        self.mean = self.state_memory.mean(0,keepdims=True)
        self.std = self.state_memory.std(0,keepdims=True)+eps
        self.state_memory = (self.state_memory-self.mean)/self.std
        self.next_state_memory = (self.next_state_memory-self.mean)/self.std
        

class ContinuousReplayBuffer(ReplayBuffer):

    def __init__(self,batch_size,obs_dims=None,action_dims=None,mem_size=None,n_envs=1,dataset=None,shuffle=False, normalise_state=True, task=''):

        if dataset:
            super().__init__(batch_size=batch_size,
                             dataset=dataset)
            self.store_offline_data(dataset,shuffle,normalise_state, task=task)
        else:
            super().__init__(batch_size=batch_size,
                            obs_dims=obs_dims,        
                            mem_size=mem_size,
                            n_envs=n_envs)
            self.action_memory = np.zeros((self.mem_size,action_dims),dtype=float)

    def store_offline_data(self,dataset,shuffle, normalise_state, task=''):
        super().store_offline_data(dataset, shuffle, normalise_state, task=task)

    def reset_buffer(self,obs_dims,action_dims):
        self.action_memory = np.zeros((self.mem_size,action_dims),dtype=float)
        super().reset_buffer(obs_dims)


class DiscreteReplayBuffer(ReplayBuffer):

    def __init__(self,batch_size,obs_dims=None,action_dims=None,mem_size=None,n_envs=1,dataset=None):

        if dataset:
            super().__init__(batch_size=batch_size,
                             dataset=dataset)
            self.store_offline_data(dataset,shuffle)
        else:
            super().__init__(obs_dims=obs_dims,
                             mem_size=mem_size,
                             batch_size=batch_size,
                             n_envs=n_envs)
            self.action_memory = np.zeros(self.mem_size,dtype=float)

    def store_offline_data(self,dataset,shuffle):
        super().store_offline_data(dataset, shuffle)
