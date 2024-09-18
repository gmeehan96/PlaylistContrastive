"""
This script contains utility functions and classes for defining models. Some are adapted
from the following script in the contrastive-mir-learning repository (corresponding to 
reference [18] in the paper):
https://github.com/andrebola/contrastive-mir-learning/blob/master/models.py
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
import yaml
from pathlib import Path


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv_2d_block(nn.Module):
    """
    Convolutional block used in SC-CNN model.
    """

    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d_block, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, shape, stride=stride, padding=shape // 2
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class AudioBackbone(nn.Module):
    """
    SC-CNN encoder model with seven convolutional blocks.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.audio_backbone = nn.Sequential(
            nn.BatchNorm2d(1),  # 256x48
            Conv_2d_block(1, 128, pooling=2),  # 128x24
            Conv_2d_block(128, 128, pooling=2),  # 64x12
            Conv_2d_block(128, 256, pooling=2),  # 32x6
            Conv_2d_block(256, 256, pooling=2),  # 16x3
            Conv_2d_block(256, 256, pooling=(1, 2)),  # 8x3
            Conv_2d_block(256, 256, pooling=(1, 2)),  # 4x3
            Conv_2d_block(256, 512, pooling=2),  # 2x1
            Flatten(),
        )
        self.backbone_output_dim = 1024

    def forward(self, x):
        z = self.audio_backbone(torch.unsqueeze(x, 1))
        return z


class AudioEncoder(nn.Module):
    """
    Full SC-CNN model with encoder and projection head.
    """

    def __init__(
        self,
        fc_hidden_dim=512,
        fc_output_dim=128,
        fc_dropout=0.5,
        fc_hidden_batch_norm=True,
        fc_output_layer_norm=True,
        **kwargs
    ):
        """
        Args:
            fc_hidden_dim: size of projection head's hidden layer
            fc_output_dim: size of projection head's output layer
            fc_dropout: projection head dropout rate
            fc_hidden_batch_norm: bool indicating whether to apply batch norm in
                projection head
            fc_output_layer_norm: bool indicating whether to apply layer norm to
                projection head's output layer
        """
        super().__init__()

        self.audio_encoder = AudioBackbone()

        self.fc_audio = DenseEncoder(
            input_dim=self.audio_encoder.backbone_output_dim,
            hidden_dim=fc_hidden_dim,
            output_dim=fc_output_dim,
            dropout=fc_dropout,
            hidden_batch_norm=fc_hidden_batch_norm,
            output_layer_norm=fc_output_layer_norm,
        )

    def forward(self, x):
        z = self.audio_encoder(x)
        z_d = self.fc_audio(z)
        return z, z_d


class DenseEncoder(nn.Module):
    """
    Flexible dense encoder utility class used for defining projection heads and
    downstream training models.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout=0.1,
        hidden_batch_norm=False,
        output_layer_norm=True,
        hidden_2_dim=None,
    ):
        """
        Args:
            input_dim: input size
            hidden_dim: size of model's first hidden layer
            output_dim: size of model's output layer
            dropout: dropout rate
            hidden_batch_norm: bool indicating whether to apply batch norm in
                model (to all hidden layers)
            fc_output_layer_norm: bool indicating whether to apply layer norm to
                model's output layer
            hidden_2_dim: size of model's second hidden layer; if None, no second
                layer is included
        """
        super(DenseEncoder, self).__init__()
        modules = [nn.Linear(input_dim, hidden_dim)]

        if hidden_batch_norm:
            modules.append(nn.BatchNorm1d(hidden_dim))

        if hidden_2_dim is None:
            modules += [
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            ]
        else:
            modules += [
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_2_dim),
            ]
            if hidden_batch_norm:
                modules.append(nn.BatchNorm1d(hidden_2_dim))
            modules += [
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_2_dim, output_dim),
            ]

        if output_layer_norm:
            modules.append(nn.LayerNorm(output_dim, eps=1e-6))

        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        return self.fc(x)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention used in tag encoder"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class TagSelfAttentionEncoder(nn.Module):
    """
    Self-attention based genre tag encoder (from Ferraro et al.)
    """

    def __init__(self, n_head, d_model, d_k, d_v, emb_file, dropout=0.1):
        super(TagSelfAttentionEncoder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            torch.Tensor(np.load(emb_file)), freeze=False
        )

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tags, mask=None):
        # in self attention q comes from the same source
        tag_embeddings = self.embeddings(tags)
        k = tag_embeddings
        v = tag_embeddings
        q = tag_embeddings

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # No residual for now
        # residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # q += residual

        # we sum all the values that were multiplied with attention weights
        q = q.sum(1)
        q = self.layer_norm(q)
        q = q.view(-1, q.shape[-1])

        return q, attn


class ResnetBackbone(torch.nn.Module):
    """
    Resnet18 audio encoder, with feed-forward components removed and concatenation
    of max and avg pooling on final layer.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.backbone_output_dim = 1024

        resnet_model = models.resnet18(pretrained=False)
        resnet_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        resnet_model = nn.Sequential(
            OrderedDict([*(list(resnet_model.named_children())[:-2])])
        )
        self.resnet_model = resnet_model

    def forward(self, x):
        z = self.resnet_model(torch.unsqueeze(x, 1))
        z_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))(z)
        z_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))(z)
        z_concat = torch.squeeze(torch.cat([z_avg_pool, z_max_pool], 1))
        return z_concat


class Resnet(torch.nn.Module):
    """
    Full Resnet18 model with projection head added.
    """

    def __init__(
        self,
        fc_hidden_dim=128,
        fc_output_dim=128,
        fc_dropout=0.0,
        fc_hidden_batch_norm=False,
        fc_output_layer_norm=False,
        **kwargs
    ):
        """
        Args:
            fc_hidden_dim: size of projection head's first hidden layer
            fc_output_dim: size of projection head's output layer
            fc_dropout: projection head dropout rate
            fc_hidden_batch_norm: bool indicating whether to apply batch norm in
                projection head
            fc_output_layer_norm: bool indicating whether to apply layer norm to
                projection head's output layer

        """
        super().__init__()

        self.audio_encoder = ResnetBackbone()
        self.fc_audio = DenseEncoder(
            input_dim=self.audio_encoder.backbone_output_dim,
            hidden_dim=fc_hidden_dim,
            output_dim=fc_output_dim,
            dropout=fc_dropout,
            hidden_batch_norm=fc_hidden_batch_norm,
            output_layer_norm=fc_output_layer_norm,
        )

    def forward(self, x):
        z_concat = self.audio_encoder(x)
        z_d = self.fc_audio(z_concat)
        return z_concat, z_d


def get_audio_model(model_params):
    """
    Utility function for retrieving model based on config parameters.
    """
    if model_params["backbone_type"] == "resnet":
        return Resnet(**model_params)
    elif model_params["backbone_type"] == "sc_cnn":
        return AudioEncoder(**model_params)
    else:
        raise ValueError("Backbone type must be either resnet or sc_cnn")


class DownstreamModel(torch.nn.Module):
    """
    MLP model used for downstream music tagging tasks.
    """

    def __init__(
        self,
        run_name,
        save_model_loc,
        backbone_type="resnet",
        scratch=False,
        fc_hidden_dim=160,
        fc_hidden_2_dim=None,
        num_classes=50,
        fc_dropout=0.0,
        fc_hidden_batch_norm=False,
        fc_output_layer_norm=False,
        model_name="best",
        **kwargs
    ):
        """
        Args:
            run_name: name of run which will be used for downstream tagging
            save_model_loc: folder where data for all runs is saved (./data)
            backbone_type: type of encoder for selected run (resnet or sc_cnn)
            scratch: bool indicating whether to intialise the encoder randomly or
                load named run
            fc_hidden_dim: size of MLP's first hidden layer
            fc_hidden_2_dim: size of MLP's second hidden layer
            num_classes: size of MLP's output layer (number of classes in tagging task)
            fc_dropout: MLP dropout rate
            fc_hidden_batch_norm: bool indicating whether to apply batch norm in
                MLP
            fc_output_layer_norm: bool indicating whether to apply layer norm to
                MLP's output layer
            model_name: suffix in model filename

        """
        super().__init__()

        if not scratch:
            self.audio_encoder = get_backbone(run_name, save_model_loc, model_name)
        else:
            if backbone_type == "resnet":
                self.audio_encoder = ResnetBackbone()
            elif backbone_type == "sc_cnn":
                self.audio_encoder = AudioEncoder().audio_encoder
            else:
                raise ValueError("Backbone type must be either resnet or sc_cnn")

        self.fc_audio = DenseEncoder(
            input_dim=self.audio_encoder.backbone_output_dim,
            hidden_dim=fc_hidden_dim,
            hidden_2_dim=fc_hidden_2_dim,
            output_dim=num_classes,
            dropout=fc_dropout,
            hidden_batch_norm=fc_hidden_batch_norm,
            output_layer_norm=fc_output_layer_norm,
        )

    def forward(self, x):
        z_concat = self.audio_encoder(x)
        z_d = self.fc_audio(z_concat)
        return z_d


def load_config(run_name, save_model_loc):
    """
    Utility function for loading the config file of a specific file based on
    run_name.
    """
    config_dir = str(Path(save_model_loc, run_name, "config.yaml"))

    with open(config_dir, "r") as in_f:
        config = yaml.safe_load(in_f)
    return config


def get_backbone(run_name, save_model_loc, model_name="best"):
    """
    Utility function for retrieving encoder of trained model based on run_name.
    """
    config = load_config(run_name, save_model_loc)

    model_dir = str(
        Path(save_model_loc, run_name, "audio_encoder_epoch_%s.pt" % model_name)
    )
    audio_model = get_audio_model(config["model_params"]["audio"])

    audio_model.load_state_dict(torch.load(model_dir))
    audio_backbone = audio_model.audio_encoder
    return audio_backbone


def get_model(run_name, save_model_loc, model_name="best"):
    """
    Utility function for retrieving full trained model based on run_name.
    """

    config = load_config(run_name, save_model_loc)

    model_dir = str(
        Path(save_model_loc, run_name, "audio_encoder_epoch_%s.pt" % model_name)
    )
    audio_model = get_audio_model(config["model_params"]["audio"])

    audio_model.load_state_dict(torch.load(model_dir))
    return audio_model


# Dictionary used for retrieving appropriate model for each data mode
model_functions = {
    "audio": lambda model_params: get_audio_model(model_params),
    "audio_self": lambda model_params: get_audio_model(model_params),
    "genre_w2v": lambda model_params: TagSelfAttentionEncoder(**model_params),
    "cf": lambda model_params: DenseEncoder(**model_params),
}
