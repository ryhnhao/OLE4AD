# Data Prep

##Installation
Install the dependent libraries as follows:
```
pip install -r requirements.txt 
```

## nuScenes

a. We use pre-cached information (detections, predictions, trajectories, etc.) from the nuScenes dataset (cached_nuscenes_info.pkl) . The data can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1hUb1dsaDUABbUKnhj63vQBi0n4AZaZyM?usp=sharing).

b. You can put the downloaded data here:
```
GPT-Driver
├── data
│   ├── cached_nuscenes_info.pkl
│   ├── split.json
├── gpt-driver
├── outputs
```

c. You can generate data and labels used in our benchmark following:

```shell
python make_ADMLP_label.py

# This generates val.npy which we have provided in the folder

python make_ADMLP_label.py

# This generates val_label.npy which we have provided in the folder
```



## Trajectories to be evaluated in our benchmark

### MLP

Follow [MLP training](train_MLP) to generate val_mlp_prediction.npy


### UniAD and VAD

Follow the official implementation of [VAD](https://github.com/hustvl/VAD) and [UniAD](https://github.com/OpenDriveLab/UniAD), you will get a pkl file after running the evaluation.

```shell
# Then run eval_uniad.py with the pkl file
# Change the file names in eval_uniad.py if you are using for VAD

python eval_uniad.py
```

### Llama Driver

We will update the implementation Llama Driver soon.

