import os
import os.path as osp

from dataclasses import dataclass
import simple_parsing


@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    # wandb parameters
    wandb_entity  : str  = "nmlt-deep-learning"
    wandb_project : str = "deep_Translation"
    save_dir      : str  = osp.join(os.getcwd(), 'output_dir')          
    epochs        : int  = 10   
    batch_size    : int  = 512
    use_cuda      : bool = False

@dataclass
class ModelParams:
    """Parameters to use for the model"""
    normalize_before        : bool        = False
    n_layers                : int         = 6
    nheads                  : int         = 4
    hidden_size             : int         = 512
    lr                      : float       = 0.0001
    dropout                 : float       = 0.1  

@dataclass
class DataParams:
    """Parameters to use for the dataset"""
    source_lang                : str         = 'en'
    target_lang                : str         = 'fr'
    data_dir                   : str         = 'data'
    model_dir                  : str         = 'models'


@dataclass
class Parameters:
    """base options."""

    hparams       : Hparams         = Hparams()
    data_param    : DataParams   = DataParams()
    model_param: ModelParams  = ModelParams()
    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance