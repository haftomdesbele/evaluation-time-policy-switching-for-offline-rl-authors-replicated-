import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, re
import numpy as np
import pickle
from pathlib import Path

plt.rc('axes', labelsize=16)
plt.rc('axes', titlesize=16)
plt.rc('legend',fontsize=12)

def plot_online_std(config_dict,ma=1):
    plt.figure()

    results_path = 'online_plot_files'
    figures_path = f'online_plots/avg_std_{config_dict["env_id"]}'
    results_dict = {}


    idx = np.arange(0,config_dict['online_steps']+config_dict['eval_counter'],config_dict['eval_counter'])

    result_files = [f.path for f in os.scandir(results_path) if f.is_file()]
    for file in result_files:
        cropped_file = os.path.relpath(file,results_path)
        if config_dict['env_id'] in cropped_file and \
                'std' in cropped_file:
            with open(file,'rb') as f:
                results_dict[cropped_file] = pickle.load(f)



    plot_list = []
    for flag in ['Policy stitching','TD-N']: 
        if flag == 'TD-N':
           #results_list = [results_dict[key] for key in results_dict if 'raw' in key]
           #results_list = [[x.detach().cpu().numpy() for x in results_dict[key]] for key in results_dict if 'raw' in key]
            all_seed_results = np.array(results_list)
            results_list = []
            for key in results_dict:
                if 'raw' in key:
                    try:
                        results_list.append([x.item() for x in results_dict[key]])
                    except AttributeError:
                        results_list.append([x for x in results_dict[key]])
            all_seed_results = np.array(results_list)
            label ='TD3-N std'
        else:
            results_list = []
            for key in results_dict:
                if 'BC' not in key and 'raw' not in key and 'o3f' not in key:

                    try:
                        results_list.append([x.detach().cpu().numpy() for x in results_dict[key]])
                    except AttributeError:
                        results_list.append([x for x in results_dict[key]])
                    
            all_seed_results = np.array(results_list)
            label = 'TD3-N + BC (PS) std'

        mean_return = all_seed_results.mean(axis=0)
        std_return = all_seed_results.std(axis=0)

        y = np.array([sum(mean_return[i:i+ma])/len(mean_return[i:i+ma]) for i in range(ma-1,len(mean_return))])
        y_err = np.array([sum(std_return[i:i+ma])/len(std_return[i:i+ma]) for i in range(ma-1,len(std_return))])

        idx = idx[:len(y)]

        sns.set_style('darkgrid')
        plot_list.append(plt.plot(idx,y,label=label))
        plt.fill_between(idx,y-y_err,y+y_err,alpha=0.2)

    ax = plt.gca()
    ax.set(ylabel='standard deviation across critic ensemble')
    ax.set_ylim(bottom=0)

    twin1 = plt.twinx()

    figures_path = f'online_plots/avg_bc_{config_dict["env_id"]}'
    results_dict = {}

    result_files = [f.path for f in os.scandir(results_path) if f.is_file()]
    for file in result_files:
        cropped_file = os.path.relpath(file,results_path)
        if config_dict['env_id'] in cropped_file and \
                'bc' in cropped_file and \
                'raw' not in cropped_file:
            with open(file,'rb') as f:
                results_dict[cropped_file] = pickle.load(f)


    results_list = [results_dict[key] for key in results_dict]
    all_seed_results = np.array(results_list)


    mean_result = all_seed_results.mean(axis=0)
    std_result = all_seed_results.std(axis=0)

    y = np.array([sum(mean_result[i:i+ma])/len(mean_result[i:i+ma]) for i in range(ma-1,len(mean_result))])
    y_err = np.array([sum(std_result[i:i+ma])/len(std_result[i:i+ma]) for i in range(ma-1,len(std_result))])
    idx = idx[:len(y)]

    
    plot_list.append(twin1.plot(idx,y,'r',label='TD3-N + BC (PS) bc proportion'))
    twin1.set(ylabel='proportion of bc used')

    plt.xlabel('Timestep')
    if 'medium-expert' in config_dict['env_id'] or 'umaze' in config_dict['env_id']:
        plt.legend(handles=[x[0] for x in plot_list], loc='upper right')
    plt.savefig(figures_path+'.pdf')


