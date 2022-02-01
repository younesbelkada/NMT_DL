# NMT_DL
Neural Machine Translation - Deep Learning

This code is an implementation of the Neural Machine Translation homework from the *Introduction to Deep Learning* course from the MVA Masters program.

## Requirements

Get the requirements by running 
```
pip install -r requirements.txt
```

## Get the training data

Get the training data by running 
```
bash download-data.sh
```

## Train your translation model

This code supports only the ```eng-fr``` translation.

You have **one file to modify**. Modify the desired variables on the ```hparams.py``` file. The variables names ar explicit for the model parameters, therefore easy to change. Change the ```wandb_project``` variable in the ```hparams.py``` file in order to push your custom run.

### Visualize the results

We highly recommend you to have an account on [wandb](https://www.wandb.ai) before running the traning script. After modifying the ```hparams.py``` file, run ```python3 main.py```.