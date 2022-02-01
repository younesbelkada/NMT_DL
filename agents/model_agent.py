import nnet_models

class ModelAgent(object):
    def __init__(self, config):
        self.config = config
    def get_model(self, source_dict, target_dict, use_cuda=False):

        nlayers = self.config.n_layers
        normalize_before = self.config.normalize_before
        nheads = self.config.nheads
        lr = self.config.lr
        dropout = self.config.dropout
        hidden_size = self.config.hidden_size


        encoder = nnet_models.TransformerEncoder(
            input_size=len(source_dict),
            hidden_size=hidden_size,
            num_layers=nlayers,
            dropout=dropout,
            heads=nheads,
            normalize_before = normalize_before
        )
        decoder = nnet_models.TransformerDecoder(
            output_size=len(target_dict),
            hidden_size=hidden_size,
            num_layers=nlayers,
            heads=nheads,
            dropout=dropout,
            normalize_before = normalize_before
        )

        model = nnet_models.EncoderDecoder(
            encoder,
            decoder,
            lr=lr,
            use_cuda=use_cuda,
            target_dict=target_dict
        )

        return model