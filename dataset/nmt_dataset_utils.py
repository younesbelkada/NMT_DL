import click
import os

from utils import reset_seed
import nmt_dataset
from subword_nmt.apply_bpe import BPE

class NMT_Dataset(object):
    """
        Wrapper object for the dataset that is going to be used for training the NMT.
    """
    def __init__(self,
        config
    ):
        self.source_lang = config.source_lang
        self.target_lang = config.target_lang
        self.data_dir = config.data_dir
        self.model_dir = '{}/{}-{}'.format(config.model_dir, config.source_lang, config.target_lang)

        bpe_path = os.path.join(config.data_dir, 'bpecodes.de-en-fr')

        with open(bpe_path) as bpe_codes:
            self.bpe_model = BPE(bpe_codes)
    
    def get_data(self, split='train'):
        assert split in ['train', 'valid', 'test']
        if split=='train':
            data = load_data(self.data_dir, self.source_lang, self.target_lang, 'train', max_size=10000, bpe_model=self.bpe_model)
        else:
            data = load_data(self.data_dir, self.source_lang, self.target_lang, split, bpe_model=self.bpe_model)
        return data

    def get_dict(self, train_data):
        source_dict_path = os.path.join(self.model_dir, 'dict.{}.txt'.format(self.source_lang))
        target_dict_path = os.path.join(self.model_dir, 'dict.{}.txt'.format(self.target_lang))

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
        return source_dict, target_dict

def preprocess(line, is_source=True, source_lang=None, target_lang=None, bpe_model=None):
    return bpe_model.segment(line.lower())


def load_data(data_dir, source_lang, target_lang, split='train', max_size=None, bpe_model=None):
    # max_size: max number of sentence pairs in the training corpus (None = all)
    path = os.path.join(data_dir, '{}.{}-{}'.format(split, *sorted([source_lang, target_lang])))
    return nmt_dataset.load_dataset(path, source_lang, target_lang, preprocess=preprocess, max_size=max_size, bpe_model=bpe_model)   # set max_size to 10000 for fast debugging

