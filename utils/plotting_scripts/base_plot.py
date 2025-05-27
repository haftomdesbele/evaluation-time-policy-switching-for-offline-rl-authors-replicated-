import matplotlib.pyplot as plt
import os, sys
import numpy as np
import pickle
from pathlib import Path

def get_result_filenames(config_dict, results_type):
    '''results_type correspods to the data we want to get from the results directory (includes just returns at the moment)'''

    dist_name = config_dict['dist_method'].__name__

    results_path = 'results'
    figures_path = 'figures'
   #path_key_list = ['optimiser','env_id']
    path_key_list = ['algo_name','env_id']
    for key in path_key_list:
        results_path = os.path.join(results_path,config_dict[key])
        figures_path = os.path.join(figures_path,config_dict[key])

    if config_dict['algo_name'] != 'bc':
        figures_path = os.path.join(figures_path,dist_name)

    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    algo_results_path_list = [f.path for f in os.scandir(results_path) if (f.name == results_type)]

    return algo_results_path_list, figures_path

def get_plot_labels(filename):

    raw_filename = filename.split('.')[0]
    label = ''
    ###last feature is env complexity which we done want in label
    for feature in raw_filename.split('-')[:-1]:
        label += feature + ' '
    return label

