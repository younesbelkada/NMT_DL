import nmt_dataset
from dataset.nmt_dataset_utils import NMT_Dataset
from agents.model_agent import ModelAgent

from utils import reset_seed, train_model


class MainAgent(object):
    def __init__(self, config):
        self.config = config
        self.NMT_data = NMT_Dataset(config.data_param)
        self.model_agent = ModelAgent(config.model_param)
    
    def run(self):
        train_data = self.NMT_data.get_data('train')
        valid_data = self.NMT_data.get_data('valid')
        test_data = self.NMT_data.get_data('test')

        source_dict, target_dict = self.NMT_data.get_dict(train_data)

        nmt_dataset.binarize(train_data, source_dict, target_dict, sort=True)
        nmt_dataset.binarize(valid_data, source_dict, target_dict, sort=False)
        nmt_dataset.binarize(test_data, source_dict, target_dict, sort=False)

        max_len = 30       # maximum 30 tokens per sentence (longer sequences will be truncated)
        batch_size = self.config.hparams.batch_size   # maximum 512 tokens per batch (decrease if you get OOM errors, increase to speed up training)
        epochs = self.config.hparams.epochs
        reset_seed()

        # *****START CODE
        train_iterator = nmt_dataset.BatchIterator(train_data, self.config.data_param.source_lang, self.config.data_param.target_lang, batch_size=batch_size, max_len=max_len, shuffle=True)
        valid_iterator = nmt_dataset.BatchIterator(valid_data, self.config.data_param.source_lang, self.config.data_param.target_lang, batch_size=batch_size, max_len=max_len, shuffle=False)
        test_iterator = nmt_dataset.BatchIterator(test_data, self.config.data_param.source_lang, self.config.data_param.target_lang, batch_size=batch_size, max_len=max_len, shuffle=False)

        model = self.model_agent.get_model(source_dict, target_dict, self.config.hparams.use_cuda)
        epochs = self.config.hparams.epochs

        train_model(train_iterator, [valid_iterator], model, epochs=epochs, checkpoint_path='models', config=self.config)