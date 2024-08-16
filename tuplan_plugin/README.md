## Overview

- This code-plugin shows how to re-implement Nuplan results shown in our paper.
- The discription of PDM-open, PDM-projection, PDM-Offset can be found more in our paper.

## Get started

### 1. Installation
To install this code-plugin, please follow these steps:
- setup the nuPlan dataset ([described here](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)) and install the nuPlan devkit ([see here](https://nuplan-devkit.readthedocs.io/en/latest/installation.html))
- download tuPlan Garage and move inside the folder

- make sure the environment you created when installing the nuplan-devkit is activated
```
conda activate nuplan
```
- move inside the folder and install the local tuplan_garage as a pip package
```
pip install -e .
```
- add the following environment variable to your `~/.bashrc`
```
NUPLAN_DEVKIT_ROOT="$HOME/nuplan-devkit/"
```

### 2. Training
Training scripts can be run with the scripts found in `/scripts/training/`.
You can also download the trained models [here](https://drive.google.com/drive/folders/1TAGvivpaOitocRemKwbo2if0qnyhK_o9?usp=sharing).
- Notes: `pdm_open_checkpoint.ckpt` is for PDM-open, and `pdm_offset_checkpoint.ckpt` is for PDM-projection and PDM-Offset

### 3. Open-loop Inference and Closed-loop evaluation
Corresponding scripts can be run with the scripts found in `/scripts/simulation/`.

- Notes1: For code simplication, we use the same config/scripts named as pdm_hybrid_planner for two methods (PDM-projection and PDM-Offset). Therefore, before runing inference, you may need to comment on line 159~184 in `/tuplan_garage/planning/simulation/planner/pdm_planner/pdm_hybrid_planner.py` to choose which model to run. 

- Notes2: When running a training or evaluation, you have to add the hydra.searchpath for the tuplan_garage correctly, as the scripts in `/scripts/training/` and in `/scripts/simulation/`.

### 4. Further Open-loop Evaluation

After runing Open-loop inference.
Modify and Run the following codes to generate predicted trajectory and ground_truth trajectory.
```
python scripts/analysis/open_loop_traj.py
```

Then turn to run our metric codes.