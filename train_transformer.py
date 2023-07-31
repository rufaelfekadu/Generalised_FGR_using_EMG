import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops import einsum
from einops.layers.torch import Rearrange
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

from utils import preprocess_data, arg_parse, get_logger
from trainner import train

import sys
sys.path.append('/home/rufael.marew/Documents/projects/tau/Fingers-Gesture-Recognition')
from pathlib import Path
import Source.fgr.models as models

from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager

# define a ViT model

# attention class
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        #self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        #self.to_out = nn.Linear(dim, dim)
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        #batch_size, num_tokens, _ = x.shape
        #qkv = self.to_qkv(x).chunk(3, dim=-1)
        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        #similarity = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        #attention = similarity.softmax(dim=-1)
        #out = einsum('b h i j, b h j d -> b h i d', attention, v)
        #out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        #return self.to_out(out)
        batch_size, num_tokens, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv[:, :, :self.dim], qkv[:, :, self.dim:2 * self.dim], qkv[:, :, -self.dim:]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        similarity = einsum(q, k, 'b h i d, b h j d -> b h i j' ) * self.scale
        attention = similarity.softmax(dim=-1)
        out = einsum(attention, v, 'b h i j, b h j d -> b h i d' )
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# mlp class
class MLP(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim_out))

    def forward(self, x):
        return self.net(x)
    
# transformerlayer class
class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.mlp = MLP(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x
# patch embeding class
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        #self.image_size = image_size
        self.patch_size = patch_size
        #self.patches = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(in_chans * patch_size ** 2, embed_dim)

    def forward(self, x):
        #batch_size, _, height, width = x.shape
        #patch_size = self.patch_size
        #x = self.patches(x).flatten(2).transpose(1, 2)
        #x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        #return x
        #batch_size, _, height, width = x.shape
        #patch_size = self.patch_size
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.proj(x)
        return x
    
#postional encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, dim, num_patches):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.shape[1]]
    
#vision trnasformer  class
class VisionTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=1, dropout=0., emb_dropout=0.):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        #self.patch_size = patch_size
        #self.num_patches = num_patches
        #self.to_patch_embedding = nn.Linear(channels * patch_size ** 2, dim)
        self.patch_embedding = PatchEmbedding(image_size=image_size, patch_size=patch_size, in_chans=channels, embed_dim=dim)
        self.pos_embedding = PositionalEncoding(dim, num_patches)
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(TransformerLayer(dim, heads, dim // heads, mlp_dim, dropout))
        self.norm = nn.LayerNorm(dim)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x):
        #batch_size = x.shape[0]
        #x = self.to_patch_embedding(x)
        #cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch_size)
        #x = torch.cat((cls_tokens, x), dim=1)
        #x += self.pos_embedding[:, :(self.num_patches + 1)]
        #x = self.dropout(x)
        #for transformer in self.transformer:
        #    x = transformer(x)
        #x = self.norm(x)
        #x = self.to_cls_token(x[:, 0])
        #return self.mlp_head(x)
        x = self.patch_embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for transformer in self.transformer:
            x = transformer(x)
        x = self.norm(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
    
def get_model():
    return VisionTransformer(
        image_size=4,
        patch_size=1,
        num_classes=10,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    )
if __name__ == '__main__':

    #setup device
    

    # parse arguments
    args = arg_parse()

    logger = get_logger(os.path.join(args.logdir, 'train.log'))
    logger.info(args)

    # load data
    data_path = Path('../data/doi_10')
    pipeline = Data_Pipeline(base_data_files_path=data_path)  # configure the data pipeline you would like to use (check pipelines module for more info)
    subject = 1
    dm = Data_Manager([subject], pipeline)
    print(dm.data_info())

    dataset = dm.get_dataset(experiments=[f'{subject:03d}_*_*'])
    # target_datset = dm.get_dataset(experiments=[f'{subject:03d}_1_2'])

    train_loader, test_loader, class_info = preprocess_data(dataset)
    # target_train_loader, target_test_loader, _ = preprocess_data(target_datset)


    # load model
    model = get_model()
    model = model.to(args.device)


    #count number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'number of parameters: {pytorch_total_params}')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.classInfo = class_info

    # train
    train(model, train_loader, test_loader, criterion, optimizer, args = args, logger=logger)


