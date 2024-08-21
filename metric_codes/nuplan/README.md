#NuPlan results


Copy the traj files generated in tuplan_plugin to this repo first.

## The meta-action centers

```shell
python kmeans_center.py

# This will generate the centers needed for meta-action classification
```

## Traditional L2 evaluation and hard cases selection

```shell
python compute_L2.py

# This will evaluate the trajectories using traditional L2 benchmark and filter the easy cases into easy_list.npy, which is needed for MAC-Hard later.
```

## MAC-Hard and traditional L2 evaluation on hard cases

```shell
python compute_L2_hard.py

# This will produce the result of MAC-Hard and evaluate the trajectories of hard cases using traditional L2 benchmark.
```