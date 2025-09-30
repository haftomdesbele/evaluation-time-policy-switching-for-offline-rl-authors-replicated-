import torch
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
import wandb
import numpy as np
import os, time, pickle
import matplotlib.pyplot as plt

from utils.base_agent import BaseAgent
from ..TD3.agent import Agent as TD3Agent
from ..BC.agent import Agent as BCAgent
from ..Gaussian_BC.agent import Agent as GBCAgent

class Agent(BaseAgent):

    def __init__(self, obs_dims, action_dims, algo_name='combined',**kwargs):

        super().__init__(obs_dims=obs_dims,action_dims=action_dims,
                        algo_name=algo_name,**kwargs)

        self.gaussian_bc = kwargs.get('gaussian_bc',True)

        self.agent = TD3Agent(obs_dims=obs_dims, action_dims=action_dims, **kwargs)
        
        kwargs['model_info'] = {'layers':[256,256,256],
                                  'hidden_activation':'ReLU', 
                                  'critic_final_activation':'',
                                  }

        if self.gaussian_bc:
            self.bc_agent = GBCAgent(obs_dims=obs_dims, action_dims=action_dims, **kwargs)
        else:
            self.bc_agent = BCAgent(obs_dims=obs_dims, action_dims=action_dims, **kwargs)

                    
        
        self.total_it = 0

    def __getattr__(self, attr):
        if hasattr(self.agent, attr):
            return getattr(self.agent, attr)

    def choose_action(self, state, **kwargs):
            return self.agent.choose_action(state,**kwargs)

    def choose_bc_action(self, state, **kwargs):
        bc_action_info = self.bc_agent.choose_action(state, **kwargs)

        return bc_action_info


    # def _evaluate_performance(self, env, iteration, config_dict, **kwargs):
    #     obs = env.reset()[0]
    #     if config_dict['normalise_state']:
    #         obs = (obs- self.replay_buffer.mean)/self.replay_buffer.std

    #     dones = False
    #     total_reward = 0
    #     episode_var = 0
    #     step = 0 


    #     bc_steps = 0
    #     total_steps = 0

    #     critic_std_sum = 0 

    #     bc_critic_val = 0
    #     critic_val = 0

    #     while not dones:
    #         obs = obs[np.newaxis,np.newaxis]
            
    #         act = self.choose_action(obs, deterministic=True, transform=True)['action']
    #         a = act.cpu().detach().numpy()

    #         total_steps += 1

    #         np_act = a

    #         if config_dict['policy_stitch']:
    #             bc_act_info = self.choose_bc_action(obs, deterministic=True)
    #             bc_act_std = bc_act_info.get('action_std',None)
    #             bc_act = bc_act_info['action']


    #             with torch.no_grad():

    #                 torch_obs = torch.tensor(obs,dtype=torch.float).to(device=self.device)
    #                 if self.total_reset:
    #                     critic_std_sum += self.critic(torch_obs, act).std()

    #                 bc_value = self.critic(torch_obs,bc_act)
    #                 critic_value = self.critic(torch_obs,act)
    #                 if critic_value.numel() > 1:
    #                     critic_std = critic_value.std()
    #                 else:
    #                     critic_std = torch.tensor(0.0, device=critic_value.device)

    #                 critic_std_sum += critic_std
    #                 #critic_std_sum += critic_value.std()

    #             thresh_val = critic_value.median() - self.replay_buffer.std_scale * critic_std
    #             #thresh_val = critic_value.median()-self.replay_buffer.std_scale*critic_value.std()
    #             bc_thresh_val = bc_value.median()

    #             critic_val += thresh_val
    #             bc_critic_val += bc_thresh_val

    #             if bc_thresh_val > thresh_val:
    #                 bc_steps+=1
    #                 act = bc_act
    #                 np_act = act.cpu().detach().numpy()
    #         else:
    #             with torch.no_grad():
    #                 torch_obs = torch.tensor(obs,dtype=torch.float).to(device=self.device)
    #                 critic_value = self.critic(torch_obs,act)
    #                 critic_std_sum += critic_value.std()

    #         #next_obs, reward, done, trunc, info = env.step(np_act.squeeze())
    #         step_result = env.step(np_act.squeeze())

    #         if len(step_result) == 5:
    #             next_obs, reward, terminated, truncated, info = step_result
    #             done = terminated or truncated
    #         else:
    #             next_obs, reward, done, info = step_result
    #             trunc= False
    #         dones = done | trunc
    #         total_reward += reward

    #         obs = next_obs
    #         if config_dict['normalise_state']:
    #             obs = (obs - self.replay_buffer.mean)/self.replay_buffer.std


    #     mean_critic_std = (critic_std_sum/total_steps).detach().cpu().item()
    #     print('mean critic_std over episode:',critic_std_sum/total_steps)
    #     avg_bc = bc_steps/total_steps*100
    #     return total_reward, avg_bc, mean_critic_std
    def _evaluate_performance(self, env, iteration, config_dict, **kwargs):
        from collections import deque

        obs = env.reset()[0]
        if config_dict['normalise_state']:
            obs = (obs - self.replay_buffer.mean) / self.replay_buffer.std

        dones = False
        total_reward = 0
        step = 0

        bc_steps = 0
        total_steps = 0

        # We'll accumulate recent critic medians in a small window to compute a running std.
        recent_critic_medians = deque(maxlen=20)  # window size: 20 (tunable)
        critic_std_sum = 0.0

        bc_critic_val = 0.0
        critic_val_accum = 0.0

        while not dones:
            obs = obs[np.newaxis, np.newaxis]

            act = self.choose_action(obs, deterministic=True, transform=True)['action']
            a = act.cpu().detach().numpy()

            total_steps += 1

            np_act = a

            if config_dict['policy_stitch']:
                bc_act_info = self.choose_bc_action(obs, deterministic=True)
                bc_act_std = bc_act_info.get('action_std', None)
                bc_act = bc_act_info['action']

                with torch.no_grad():
                    torch_obs = torch.tensor(obs, dtype=torch.float).to(device=self.device)

                    # critic(...) may return a tensor; we compute a scalar summary (median)
                    bc_value = self.critic(torch_obs, bc_act)
                    critic_value = self.critic(torch_obs, act)

                    # produce a scalar summary for this step (use median to be robust)
                    # If critic_value has multiple elements (e.g., ensemble or batch), median is meaningful.
                    try:
                        bc_median = bc_value.median().squeeze()
                    except Exception:
                        # fallback if median not available
                        bc_median = bc_value.view(-1).mean().squeeze()

                    try:
                        critic_median = critic_value.median().squeeze()
                    except Exception:
                        critic_median = critic_value.view(-1).mean().squeeze()

                    # store the scalar medians into the sliding window
                    recent_critic_medians.append(critic_median.detach().cpu())

                    # compute running std over the recent window (if more than 1 sample)
                    if len(recent_critic_medians) > 1:
                        stacked = torch.stack([torch.tensor(x) for x in recent_critic_medians])
                        running_std = stacked.std(unbiased=False).to(critic_median.device)
                    else:
                        running_std = torch.tensor(0.0, device=critic_median.device)

                    # accumulate for episode-mean reporting
                    critic_std_sum += running_std
                    critic_val_accum += critic_median
                    bc_critic_val += bc_median

                    # threshold for switching uses current step's running_std
                    thresh_val = critic_median - self.replay_buffer.std_scale * running_std
                    bc_thresh_val = bc_median

                    # perform switching decision per-step (same logic as before)
                    if bc_thresh_val > thresh_val:
                        #print(f"Switching to BC action at step {total_steps} with threshold {thresh_val:.4f} vs BC median {bc_thresh_val:.4f}")
                        bc_steps += 1
                        act = bc_act
                        np_act = act.cpu().detach().numpy()

            else:
                # policy_stitch disabled: still compute critic median and update running buffer
                with torch.no_grad():
                    torch_obs = torch.tensor(obs, dtype=torch.float).to(device=self.device)
                    critic_value = self.critic(torch_obs, act)
                    try:
                        critic_median = critic_value.median().squeeze()
                    except Exception:
                        critic_median = critic_value.view(-1).mean().squeeze()

                    recent_critic_medians.append(critic_median.detach().cpu())

                    if len(recent_critic_medians) > 1:
                        stacked = torch.stack([torch.tensor(x) for x in recent_critic_medians])
                        running_std = stacked.std(unbiased=False).to(critic_median.device)
                    else:
                        running_std = torch.tensor(0.0, device=critic_median.device)

                    critic_std_sum += running_std
                    critic_val_accum += critic_median

            # step the environment with robust unpacking (works with gym and gymnasium)
            step_result = env.step(np_act.squeeze())
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = bool(terminated or truncated)
                trunc = bool(truncated)
            else:
                next_obs, reward, done, info = step_result
                trunc = False

            dones = bool(done or trunc)
            total_reward += reward

            obs = next_obs
            if config_dict['normalise_state']:
                obs = (obs - self.replay_buffer.mean) / self.replay_buffer.std

        # prevent division by zero
        if total_steps > 0:
            # critic_std_sum is sum of tensors (or floats) — ensure numeric scalar
            try:
                mean_critic_std = (critic_std_sum / total_steps).detach().cpu().item()
            except Exception:
                # if critic_std_sum is a float
                mean_critic_std = float(critic_std_sum / total_steps)
        else:
            mean_critic_std = 0.0

        # print a nice diagnostic
        print('mean critic_std over episode:', mean_critic_std)

        # percent of steps where BC was used (0-100)
        avg_bc = bc_steps / total_steps * 100 if total_steps > 0 else 0.0
        return total_reward, avg_bc, mean_critic_std

    def train_online(self, config_dict, normalise_state=True):

        self.agent.load_model(iter_no=config_dict['num_env_steps']-1)

        if config_dict['policy_stitch']:
            self.bc_agent.load_model(iter_no=config_dict['num_env_steps']-1)

        env = config_dict['env']

        if 'ant' in config_dict['env_id']:
            env.reset()
        else:
            env.reset(seed=self.seed)


        total_steps = 0
        ep_num = 0

        return_list = []
        bc_list = []
        std_list = []
        step_list = []


        avg_return, _, mean_std, avg_bc = self.evaluate_performance(config_dict,total_steps,online=True)
        return_list.append(avg_return)
        bc_list.append(avg_bc)
        std_list.append(mean_std)
        step_list.append(total_steps)

        if wandb.run is not None:
            wandb.log(
                    {'d4rl_unnormalised_score':avg_return,
                    'd4rl_normalised_score':config_dict['env'].get_normalized_score(avg_return)*100,
                    'average_bc_used':avg_bc,
                    'avg_critic_std_across_episode':mean_std},
                    step=total_steps)



        while total_steps <config_dict['online_steps']:

            if 'ant' in config_dict['env_id']:
                obs = env.reset()[0]
            else:
                obs = env.reset(seed=self.seed)[0]

            if config_dict['normalise_state']:
                obs = (obs- self.replay_buffer.mean)/self.replay_buffer.std
            else:
                obs = obs[np.newaxis]

            dones = False
            total_reward = 0
            ep_num += 1
            bc_steps = 0


            while not dones:
                obs = obs[np.newaxis,np.newaxis]

                ###choosing action 
                act = self.choose_action(obs, deterministic=True, transform=True)['action']
                np_act = act.cpu().detach().numpy()

                total_steps += 1

                if config_dict['policy_stitch']:
                    bc_act_info = self.choose_bc_action(obs, deterministic=True)

                    bc_act_std = bc_act_info.get('action_std',None)
                    bc_act = bc_act_info['action']

                    with torch.no_grad():

                        torch_obs = torch.tensor(obs,dtype=torch.float).to(device=self.device)
                        bc_value = self.critic(torch_obs,bc_act)
                        critic_value = self.critic(torch_obs,act)

                    thresh_val = critic_value.median()-self.replay_buffer.std_scale*critic_value.std()
                    bc_thresh_val = bc_value.median()
                    if bc_thresh_val > thresh_val:
                        bc_steps+=1
                        act = bc_act
                        np_act = act.cpu().detach().numpy()

                    np_act += np.random.normal(scale=0.1,size=np_act.shape)
                    np_act = np_act.clip(self.min_action_val.cpu().numpy(),self.max_action_val.cpu().numpy())

                next_obs, reward, done, trunc, info = env.step(np_act.squeeze())

                dones = done | trunc
                total_reward += reward

                if config_dict['normalise_state']:
                    next_obs = (next_obs - self.replay_buffer.mean)/self.replay_buffer.std

                if 'ant' in config_dict['env_id']:
                    reward = 4*(reward - 0.5)

                self.replay_buffer.store_transition(obs, next_obs, np_act, reward, done)

                obs = next_obs
                if not config_dict['normalise_state']:
                    obs = obs[np.newaxis]

                loss = self.learn(online=True)

                if wandb.run is not None and loss[1] is not None:
                    wandb.log({
                                'critic_loss':loss[0],
                                'actor_loss':loss[1],
                                },
                                step=total_steps
                                )

                if total_steps%10000 == 0:
                    print(loss)
                    avg_return, _, mean_std, avg_bc = self.evaluate_performance(config_dict,total_steps,online=True)
                    return_list.append(avg_return)
                    bc_list.append(avg_bc)
                    std_list.append(mean_std)
                    step_list.append(total_steps)

                    if wandb.run is not None:
                        wandb.log(
                                {'d4rl_unnormalised_score':avg_return,
                                'd4rl_normalised_score':config_dict['env'].get_normalized_score(avg_return)*100,
                                'average_bc_used':avg_bc,
                                'avg_critic_std_across_episode':mean_std},
                                step=total_steps)



        if not os.path.exists('online_plot_files'):
            os.makedirs('online_plot_files')

        f_name = 'return'
        tag = ''

        if not config_dict['policy_stitch']:
            tag = '-raw'

        pickle.dump(return_list, open(f'online_plot_files/{f_name+tag}-{config_dict["env_id"]}-seed_{config_dict["seed"]}.pickle','wb'))

        if config_dict['policy_stitch']:
            f_name = 'bc'
            pickle.dump(bc_list, open(f'online_plot_files/{f_name}-{config_dict["env_id"]}-seed_{config_dict["seed"]}.pickle','wb'))


        f_name = 'std'
        pickle.dump(std_list, open(f'online_plot_files/{f_name+tag}-{config_dict["env_id"]}-seed_{config_dict["seed"]}.pickle','wb'))


    def learn(self, sample_range=None, online=False, **kwargs):

        loss = None
        self.total_it += 1
        
        loss = self.agent.learn(online=online)

        return loss

