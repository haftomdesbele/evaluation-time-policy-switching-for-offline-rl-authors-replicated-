# Policy Switching

run setup.sh to create a virtual environment and install necessary dependancies

- notes.txt provides additional information on modifying libraries to make packages compatible

- we also provide necessary mujoco files, move these to your home directory and rename directory `mujoco` to `.mujoco`


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
@article{neggatu2025evaluation,
  title={Evaluation-Time Policy Switching for Offline Reinforcement Learning},
  author={Neggatu, Natinael Solomon and Houssineau, Jeremie and Montana, Giovanni},
  journal={arXiv e-prints},
  pages={arXiv--2503},
  year={2025}
}

```
