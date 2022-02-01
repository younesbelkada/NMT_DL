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


def postprocess(line):
    return line.replace('@@ ', '')

def reset_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_model(model, checkpoint_path):
    dirname = os.path.dirname(checkpoint_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    torch.save(model, os.path.join(checkpoint_path, model.__class__.__name__+'.p'))

def train_model(
        train_iterator,
        valid_iterators,
        model,
        checkpoint_path,
        epochs=1,
        validation_frequency=1,
        config=None
    ):
    """
    train_iterator: instance of nmt_dataset.BatchIterator or nmt_dataset.MultiBatchIterator
    valid_iterators: list of nmt_dataset.BatchIterator
    model: instance of nnet_models.EncoderDecoder
    checkpoint_path: path of the model checkpoint
    epochs: iterate this many times over train_iterator
    validation_frequency: validate the model every N epochs
    """

    reset_seed()

    best_bleu = -1
    model_dir = config.hparams.save_dir
    wandb_name = config.hparams.wandb_project
    wandb_entity = config.hparams.wandb_entity

    wandb.init(project=wandb_name, entity=wandb_entity)
    for epoch in range(1, epochs + 1):

        start = time.time()
        running_loss = 0

        print('Epoch: [{}/{}]'.format(epoch, epochs))

        # Iterate over training batches for one epoch
        for i, batch in tqdm(enumerate(train_iterator), total=len(train_iterator)):
            t = time.time()
            tmp_loss = model.train_step(batch)
            running_loss += tmp_loss
            wandb.log({'loss':tmp_loss})

        # Average training loss for this epoch
        # *****START CODE
        epoch_loss = running_loss / len(train_iterator)
        # *****END CODE

        print("loss={:.3f}, time={:.2f}".format(epoch_loss, time.time() - start))
        sys.stdout.flush()

        # Evaluate and save the model
        if epoch % validation_frequency == 0:
            bleu_scores = []
            
            # Compute BLEU over all validation sets
            for valid_iterator in valid_iterators:
                # *****START CODE
                src, tgt = valid_iterator.source_lang, valid_iterator.target_lang
                translation_output = model.translate(valid_iterator, postprocess)
                bleu_score = translation_output.score
                output = translation_output.output
                # *****END CODE

                with open(os.path.join(model_dir, 'valid.{}-{}.{}.out'.format(src, tgt, epoch)), 'w') as f:
                    f.writelines(line + '\n' for line in output)

                print('{}-{}: BLEU={}'.format(src, tgt, bleu_score))
                sys.stdout.flush()
                bleu_scores.append(bleu_score)

            # Average the validation BLEU scores
            bleu_score = round(sum(bleu_scores) / len(bleu_scores), 2)
            if len(bleu_scores) > 1:
                print('BLEU={}'.format(bleu_score))
            wandb.log({'bleu':bleu_score})
            # Update the model's learning rate based on current performance.
            # This scheduler divides the learning rate by 10 if BLEU does not improve.
            model.scheduler_step(bleu_score)

            # Save a model checkpoint if it has the best validation BLEU so far
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                save_model(model, checkpoint_path)

        print('=' * 50)

    print("Training completed. Best BLEU is {}".format(best_bleu))