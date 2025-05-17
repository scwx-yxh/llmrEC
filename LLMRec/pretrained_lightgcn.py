"""

"""

import torch
import torch.nn as nn
import numpy as np

class LightGCN(nn.Module):
    def __init__(self, dataset, user_num, item_num, graph, latent_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.graph = graph.cuda()
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        if dataset == "movielens":
            self.user_emb = nn.Embedding.from_pretrained(
                torch.load("/home/share/yangxuanhui/dataset/movielens/movielens_emb/movielens_lightgcn_user"), 
                freeze=False,
            )
            self.item_emb = nn.Embedding.from_pretrained(
                torch.load("/home/share/yangxuanhui/dataset/movielens/movielens_emb/movielens_lightgcn_item"), 
                freeze=False,
            )
        elif dataset == "tiktok":
            self.user_emb = nn.Embedding.from_pretrained(
                torch.load("/home/share/yangxuanhui/dataset/tiktok/tiktok_emb/tiktok_lightgcn_user"), 
                freeze=False,
            )
            self.item_emb = nn.Embedding.from_pretrained(
                torch.load("/home/share/yangxuanhui/dataset/tiktok/tiktok_emb/tiktok_lightgcn_item"), 
                freeze=False,
            )
        elif dataset == "kwai":
            self.user_emb = nn.Embedding.from_pretrained(
                torch.load("/home/share/yangxuanhui/dataset/kwai/kwai_emb/kwai_lightgcn_user"), 
                freeze=False,
            )
            self.item_emb = nn.Embedding.from_pretrained(
                torch.load("/home/share/yangxuanhui/dataset/kwai/kwai_emb/kwai_lightgcn_item"), 
                freeze=False,
            )
        else:
            self.user_emb = nn.Embedding(user_num, latent_dim)
            nn.init.xavier_normal_(self.user_emb.weight)
            self.item_emb = nn.Embedding(item_num, latent_dim)
            nn.init.xavier_normal_(self.item_emb.weight)
            
    def forward(self):
        all_user_emb = self.user_emb.weight
        all_item_emb = self.item_emb.weight
        all_emb = torch.cat([all_user_emb, all_item_emb]).cuda()

        embs = [all_emb]

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        user_emb, item_emb = torch.split(light_out, [self.user_num, self.item_num])

        return user_emb, item_emb
    
