import sys
sys.path.insert(0, '/root/workspace/Long-Seq-Model')
sys.path.insert(0,'/root/Long-Seq-Model')
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Mamba_Family import Mamba_Layer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
from models.mamba_ssm.Mamba1 import Mamba
class Model(nn.Module):
    """
    Mamba
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

        if configs.embed_type == 0:
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 3:
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        self.mamba_layers = nn.ModuleList(
            [
                Mamba_Layer(
                    Mamba(configs.d_model, d_state=configs.d_state, d_conv=configs.d_conv),
                    configs.d_model
                ) 
                for i in range(configs.d_layers)
            ]
        )
        self.out_proj=nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_dec, x_mark_dec):
        x = self.dec_embedding(x_dec, x_mark_dec)
        for i in range(self.configs.d_layers):
            x = self.mamba_layers[i](x)
        out = self.out_proj(x)

        return out

if __name__ == '__main__' :
    class Configs:
        def __init__(self):
            self.embed_type = 2  # Example embed type for testing
            self.dec_in = 10     # Example input size
            self.d_model = 512   # Model dimension
            self.embed = 'fixed' # Example embedding type
            self.freq = 'h'      # Example frequency
            self.dropout = 0.1   # Dropout rate
            self.d_state = 128   # State dimension for Mamba2
            self.d_conv = 64     # Conv dimension for Mamba2
            self.d_layers = 3    # Number of Mamba layers
            self.c_out = 1       # Output dimension

    configs = Configs()
    main=Model(configs).to("cuda")
    print(main)