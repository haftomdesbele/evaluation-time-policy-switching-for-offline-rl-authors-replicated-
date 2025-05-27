import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, re
import numpy as np
import pickle
from pathlib import Path

from .base_plot import get_result_filenames, get_plot_labels

def plot_online_bc(config_dict,ma=1):
    plt.figure()

    results_path = 'online_plot_files'
    figures_path = f'online_plots/avg_bc_{config_dict["env_id"]}'
    results_dict = {}


    idx = np.arange(0,config_dict['online_steps'],config_dict['eval_counter'])
    
    result_files = [f.path for f in os.scandir(results_path) if f.is_file()]
    for file in result_files:
        cropped_file = os.path.relpath(file,results_path)
        if config_dict['env_id'] in cropped_file and \
                'bc' in cropped_file:
            with open(file,'rb') as f:
                results_dict[cropped_file] = pickle.load(f)


#   if stat == 'std':
#       results_list = [[x.detach().cpu().numpy() for x in results_dict[key]] for key in results_dict]
#   else:
    results_list = [results_dict[key] for key in results_dict]
    all_seed_results = np.array(results_list)

   #all_seed_results = np.array(list(results_dict[key] for key in results_dict))

    mean_result = all_seed_results.mean(axis=0)
    std_result = all_seed_results.std(axis=0)

    y = np.array([sum(mean_result[i:i+ma])/len(mean_result[i:i+ma]) for i in range(ma-1,len(mean_result))])
    y_err = np.array([sum(std_result[i:i+ma])/len(std_result[i:i+ma]) for i in range(ma-1,len(std_result))])

    idx = idx[-len(y):]

    sns.set_style('darkgrid')
    plt.plot(idx,y)
    plt.fill_between(idx,y-y_err,y+y_err,alpha=0.2)


    ax = plt.gca()
    ax.set_ylim(bottom=0)

    plt.title(config_dict['env_id'])
   #plt.legend()
    plt.savefig(figures_path)



