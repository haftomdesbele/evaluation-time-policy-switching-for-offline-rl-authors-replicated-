# Policy Switching

1. Run this command to install the OpenGL and OSMesa dependencies:
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libglfw3-dev patchelf


2.Make sure you have downloaded MuJoCo 2.1.0 and placed it in ~/.mujoco/mujoco210.

mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvzf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz


3. Modified from the authors,   these lines of codes for gym in base_agent(evaluate_performance)

conda env create -f environment.yml
conda activate rlenv

## Running code

To run code, execute `python main.py`


## Details
within `main.py` there is the option to run the following algos:
- `td3_n_offline` - Used to train td3_n offline
- `bc_offline` - Used to train bc agent (either gaussian or vanilla) offline
- `combined` - Used to combine policies for offline or online
- There are also a number of configurations that can be adjusted in `main.py` to control training

## Cite
--- 
Please cite our work if you find it useful

```
@inproceedings{neggatu2025evaluation,
  title={Evaluation-Time Policy Switching for Offline Reinforcement Learning},
  author={Neggatu, Natinael Solomon and Houssineau, Jeremie and Montana, Giovanni},
  booktitle={Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems},
  pages={1520--1528},
  year={2025}
}

```
