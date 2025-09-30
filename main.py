import torch
import numpy as np
import gym
import time, uuid
from execution_scripts import td3_n_offline, bc_offline, combined

from utils.plotting_scripts import plot_online_return, plot_online_std


if __name__ == '__main__':
    model_info = {'layers':[256,256,256], ##base layer model spec
                  'hidden_activation':'ReLU', ##activation for hidden laeyrs
                  'critic_final_activation':'',
                  }

    bc_params = {'gaussian_bc':True}


    td3_params = {'policy_noise_std':0.2,
                  'noise_clip':0.5,
                  'policy_update_freq':2,
                  'exploration_noise_std':0.1,
                  'td3_alpha':1}


    lr_info = {'optimiser':'Adam', 
               }

    replay_buffer_params = {'mem_size':2**20,
                            'batch_size':256,
                            'normalise_state':True,
                            }

    ensemble_info = {'ensemble_num':1,
                    'critic_factor':10} ##how many critics per actor



    performance_eval_config = {'num_evals':10,    ## how many times to evaluate when testing performance of policy
                               'eval_counter':10000, ##after how many steps of learning to evaluate offline
                               }

    training_config = {'num_env_steps':1000001,#i romove one zero 
                        'online_steps':250001}

    machine_config = {'n_processors':2, 
                      'device':'cuda:0' if torch.cuda.is_available() else 'cpu',}

    env_id = 'halfcheetah-medium-v2'
    #env_id = 'antmaze-umaze-diverse-v2'

    config_dict = {'env':gym.make(env_id),
                   'hash_id':str(hash(time.time())),
                   'seed':0,
                   'env_id':env_id,
                   'gamma':0.99,
                   'train_model':False,
                   'model_info':model_info,
                   'dep_targ':True,
                    **ensemble_info,
                    **machine_config,
                    **performance_eval_config,
                    **training_config,
                    **lr_info,
                    **replay_buffer_params,
                    **bc_params,
                    **td3_params,
                    'wandb_project':'Policy Stitching',
                    'id':str(uuid.uuid4())[:8],
                    }

    config_dict['task'],*config_dict['data_quality'],_ = env_id.split('-')
    config_dict['data_quality'] = '-'.join(config_dict['data_quality'])

    config_dict['offline'] = True #False
    config_dict['policy_stitch'] = True

    if 'antmaze' in env_id:
        config_dict['gamma'] = 0.999
        config_dict['num_env_steps'] = 500001
       #config_dict['num_evals'] = 100
        config_dict['dep_targ'] = False
        config_dict['normalise_state'] = False

        if config_dict['offline']:
            config_dict['eval_counter'] = 10000

        config_dict['task'] = 'ant'

    config_dict['min_val']=config_dict['env'].action_space.low
    config_dict['max_val']=config_dict['env'].action_space.high

    seed = config_dict['seed']
    config_dict['rng'] = np.random.default_rng(seed)
    config_dict['env'].action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    #td3_n_offline(config_dict)

    #bc_offline(config_dict)

    combined(config_dict)
    
   #plot_online_return(config_dict)
   #plot_online_std(config_dict)