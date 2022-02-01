import os
import time
import copy
import sys

import torch
import wandb

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial
from tqdm import tqdm as tqdm

import nmt_dataset
import nnet_models
from subword_nmt.apply_bpe import BPE

from utils import load_data, model_dir, reset_seed, train_model

source_lang, target_lang = 'en', 'fr'
use_cuda = True

train_data = load_data(source_lang, target_lang, 'train', max_size=10000)   # set max_size to 10000 for fast debugging
valid_data = load_data(source_lang, target_lang, 'valid')
test_data = load_data(source_lang, target_lang, 'test')

source_dict_path = os.path.join(model_dir, 'dict.{}.txt'.format(source_lang))
target_dict_path = os.path.join(model_dir, 'dict.{}.txt'.format(target_lang))

source_dict = nmt_dataset.load_or_create_dictionary(
    source_dict_path,
    train_data['source_tokenized'],
    minimum_count=10,
    reset=False    # set reset to True if you're changing the data or the preprocessing
)

target_dict = nmt_dataset.load_or_create_dictionary(
    target_dict_path,
    train_data['target_tokenized'],
    minimum_count=10,
    reset=False
)

nmt_dataset.binarize(train_data, source_dict, target_dict, sort=True)
nmt_dataset.binarize(valid_data, source_dict, target_dict, sort=False)
nmt_dataset.binarize(test_data, source_dict, target_dict, sort=False)

max_len = 30       # maximum 30 tokens per sentence (longer sequences will be truncated)
batch_size = 512   # maximum 512 tokens per batch (decrease if you get OOM errors, increase to speed up training)

reset_seed()

# *****START CODE
train_iterator = nmt_dataset.BatchIterator(train_data, source_lang, target_lang, batch_size=batch_size, max_len=max_len, shuffle=True)
valid_iterator = nmt_dataset.BatchIterator(valid_data, source_lang, target_lang, batch_size=batch_size, max_len=max_len, shuffle=False)
test_iterator = nmt_dataset.BatchIterator(test_data, source_lang, target_lang, batch_size=batch_size, max_len=max_len, shuffle=False)
# *****END CODE

shallow_transformer_encoder_preLN = nnet_models.TransformerEncoder(
    input_size=len(source_dict),
    hidden_size=512,
    num_layers=1,
    dropout=0.0,
    heads=4,
    normalize_before = True
)
shallow_transformer_decoder_preLN = nnet_models.TransformerDecoder(
    output_size=len(target_dict),
    hidden_size=512,
    num_layers=1,
    heads=4,
    dropout=0.0,
    normalize_before = True
)

shallow_transformer_model_preLN = nnet_models.EncoderDecoder(
    shallow_transformer_encoder_preLN,
    shallow_transformer_decoder_preLN,
    lr=0.001,
    use_cuda=use_cuda,
    target_dict=target_dict
)

train_model(train_iterator, [valid_iterator], shallow_transformer_model_preLN,epochs=1,checkpoint_path='models')
