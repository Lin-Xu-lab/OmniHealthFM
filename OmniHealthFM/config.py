import torch

class Config(object):
    """para"""
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
        self.dropout = 0.1                                                 
        self.batch_size = 32
        self.lr = 1e-4                  
        self.encoder_layer = 4
        self.encoder_head = 2
        self.decoder_layer = 2
        self.decoder_head = 2
        self.mask_ratio = 0.3
        self.RNA_tokens = 2646
        self.emb_dim = 128       
        self.total_epoch = 100
        self.warmup_epoch = 10

config = Config()