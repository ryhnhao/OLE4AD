# MLP for AD in nuScenes

## Data Prep

Use make_ADMLP_data.py and make_ADMLP_label.py to generate the following files:
- train.npy
- train_label.npy
- val.npy
- val_label.npy

## Train

```shell
python train_ADMLP.py
```

## Eval

```shell
python eval_ADMLP.py

# This will generate val_mlp_prediction.npy, which contains the trajetory needed for our benchmark.
```

