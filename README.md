
# Bridging the Open vs. Closed Loop Gap: New Open-Loop Evaluation Benchmarks for End-to-End Autonomous Driving Planning

## Overview
- [TODO List](#todo-list)
- [Contributions](#contributions)
- [Inference the Trajectory](#inference-the-trajectory)
- [Quick Evaluation based on Provided Trajectory](#quick-evaluation-based-on-provided-trajectory)
- [Benchmark Codes](#benchmark-codes)
- [Related nuPlan Implementation](#related-nuplan-implementation)
- [Acknowledgment](#acknowledgment)
- [License](#license)

## TODO List
- [x] Code and Data for Quick Evaluation Release
- [x] nuPlan Evaluation Release
- [x] Detailed Code and Instructions for Four Benchmark Methods Release
- [ ] Incorprating Other Open-loop Metrics

## Contributions
* We comprehensively reveal the limitation of current L2 open-loop evaluation, demonstrating that current L2 metric cannot account for the dynamics of the real world.

* We introduce novel open-loop evaluation benchmarks including 4 improvements specifically designed for end-to-end autonomous driving planning, better reflecting the dynamic nature of real-world driving scenarios.

* Extensive experiments and reevaluation of existing methods demonstrate that our approach significantly mitigates the issues associated with conventional open-loop evaluations, bridging the gap between open-loop and closed-loop evaluations.

* We explore the relationship between open-loop and closed-loop benchmarks, highlighting that a well-designed open-loop benchmark can serve as an effective rapid test for end-to-end autonomous driving.

## Inference the Trajectory

We show how to reimplement four methods for trajectory inference in [data readme](data/README.md). The methods include:

* **MLP**. We build a MLP with 3 hidden layers, each with 512 hidden units, and output layer with dimension 12 representing 6 waypoints in a trajectory.

* **Llama2 Driver**. We follow gptdriver while we use llama2-7b instead of GPT-3.5. Our Llama2 Driver (ego only) takes only ego status as input and outputs a trajectory. Both inputs and outputs are just text. For our Llama2 Driver(without ego status version), the model takes navigation command and the motion prediction ground truth results, which are converted to text refered to \cite{mao2023gptdriver}, and outputs a trajectory. 

* **UniAD and VAD-Base** are well-known and peer-reviewed open-source end-to-end autonomous driving models.

Notes: We benchmark these methods because they are open-sourced, allowing for transparent validation of results. We are open to including additional methods in the benchmark as long as their predicted trajectories are provided or can be reproduced using their official code repositories.

## Quick Evaluation based on Provided Trajectory

For simplification, we provide the trajectory data to be evaluated in [data folder](data). You can use these data for the metric code below.

## Benchmark Codes

### Traditional L2 error
```shell
python metric_codes/compute_L2.py

# For results of different models:
# change the npy files used in
# pred = np.load('vad_ego_prediction.npy') 
```

### ADE_X and ADE_Y
```shell
python metric_codes/compute_ADE_x.py

# For results of different models:
# change the npy files used in
# pred = np.load('vad_ego_prediction.npy') 
```

```shell
python metric_codes/compute_ADE_y.py

# For results of different models:
# change the npy files used in
# pred = np.load('vad_ego_prediction.npy') 
```

### MAC
```shell
#For 11 centers
python metric_codes/kmeans_11_MAC.py

#For 16 centers
python metric_codes/kmeans_16_MAC.py

#For 21 centers
python metric_codes/kmeans_21_MAC.py
```


### MAC-Hard

```shell
#For 11 centers
python metric_codes/kmeans_11.py

#For 16 centers
python metric_codes/kmeans_16.py

#For 21 centers
python metric_codes/kmeans_21.py
```

For the detailed results, please refer to our paper.

## Related nuPlan Implementation
You can find more in `/tuplan_plugin/`

The `/tuplan_plugin/README.md` shows the training, inference and evaluation for nuplan benchmarks.

## Acknowledgment
- [VAD](https://github.com/hustvl/VAD)
- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [GPT-driver](https://github.com/PointsCoder/GPT-Driver)
- [AD-MLP](https://github.com/E2E-AD/AD-MLP)
- [tuplan-garage](https://github.com/autonomousvision/tuplan_garage)

Sincere appreciation for their great contributions.


## License

Before using the dataset, you should register on the [nuScenes](https://www.nuscenes.org/nuscenes) website and agree to the terms of use of the [nuScenes](https://www.nuscenes.org/nuscenes). 
The code and the generated data are both subject to the [MIT License](./LICENSE).