import math
import torch
import torch.nn as nn

from torchvision.models import resnet18

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.log(torch.tensor(10000.0)) / d_model)

        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TimeSeriesModel(nn.Module):
    def __init__(self, nhead, num_layers, dim_feedforward, dropout, num_in_features, out_channels, k, stride, seq_len, num_classes, d_model=256):
        super().__init__()
        max_len = math.floor((seq_len - k) / stride) + 2 # max_len post-conv1d PLUS prepended cls token
        
        self.conv_layer = nn.Conv1d(num_in_features, out_channels, k, stride) # padding=0 by default, (batch_size, num_in_features, max_len - 1) -> (batch_size, out_channels, max_len - 1)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(out_channels)
        
        self.patch_layer = nn.Linear(out_channels, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc = PositionalEncoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.conv_layer(src) # (batch_size, out_channel, max_len - 1)
        src = self.gelu(src)
        src = self.norm(src.permute(0, 2, 1)) # (batch_size, max_len - 1, out_channel)

        src = self.patch_layer(src) # (batch_size, max_len - 1, d_model)
        cls_exp = self.cls.expand(src.size(0), 1, -1)
        src = torch.cat([cls_exp, src], dim=1) # (batch_size, max_len, d_model)
        src = self.pos_enc(src) # (batch_size, max_len, d_model)
        src = self.dropout(src)

        memory = self.transformer_encoder(src) # (batch_size, max_len, d_model)
        cls_token = memory[:, 0, :] # (batch_size, d_model), only CLS token
        output = self.head(cls_token) # (batch_size, num_classes)

        return output

class MLP(nn.Module):
    def __init__(self, num_in_features, num_layers, num_nodes, num_classes, dropout):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(num_in_features)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_in_features if i == 0 else num_nodes, num_nodes), 
                nn.ReLU(),
                nn.BatchNorm1d(num_nodes),
                nn.Dropout(dropout)
            ) for i in range(num_layers)])
        self.head = nn.Linear(num_nodes, num_classes)

    def forward(self, x):
        x = self.input_bn(x)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.head(x) # (batch_size, num_classes)

        return output

class MultiModalModel(nn.Module):
    def __init__(self, num_layers, num_nodes, num_classes, dropout, resnet_state_dict_path, mlp_state_dict_path, mlp_best_params):
        super().__init__()
        
        mlp_model = MLP(
                    9, 
                    mlp_best_params['num_layers'], 
                    mlp_best_params['num_nodes'], 
                    2, 
                    mlp_best_params['dropout'])
        mlp_model.load_state_dict(torch.load(mlp_state_dict_path, weights_only=True))
        mlp_model.head = nn.Identity() # (batch_size, num_nodes)
        
        mlp_model.eval()
        for param in mlp_model.parameters():
            param.requires_grad = False

        self.mlp_model = mlp_model
        num_in_features = mlp_model.hidden_layers[-1][0].out_features # output feature size of last hidden layer becomes input feature size for fusion head
        
        resnet_model = resnet18(weights=None)  # architecture only, weights will be overwritten
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)  # match final FC layer shape from trained model
        resnet_model.load_state_dict(torch.load(resnet_state_dict_path, weights_only=True))
        resnet_model.fc = nn.Identity() # (batch_size, 512)
        
        resnet_model.eval()
        for param in resnet_model.parameters():
            param.requires_grad = False

        self.resnet_model = resnet_model

        self.fusion_head = MLP(num_in_features + 512, num_layers, num_nodes, num_classes, dropout) # num_in_features is fixed! MLP num_in_features + 512

    def train(self, mode=True): # prevent generic model.train call from changing mlp_model and resnet_model mode away from .eval
        super().train(mode)
        self.mlp_model.eval()
        self.resnet_model.eval()
        return self

    def forward(self, x_mlp, x_resnet):
        with torch.no_grad():
            mlp_out = self.mlp_model(x_mlp)
            resnet_out = self.resnet_model(x_resnet)
        
        fused_input = torch.cat([mlp_out, resnet_out], dim=-1)
        output = self.fusion_head(fused_input)

        return output