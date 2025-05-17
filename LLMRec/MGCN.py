import math 
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
import os

class MGCN(torch.nn.Module):
    def __init__(self, v_feat, a_feat, t_feat, n_ui_layers, embedding_dim, knn_k, n_layers, n_users, n_items, user_embedding, item_embedding, graph, dataset):
        super(MGCN,self).__init__()
        self.sparse = True
        #self.cl_loss = cl_loss
        self.n_ui_layers = n_ui_layers
        self.embedding_dim = embedding_dim
        self.knn_k = knn_k
        self.n_layers = n_layers
        #self.reg_weight = reg_weight
        self.n_user = n_users
        self.n_item = n_items

        # self.interaction_matrix = user_item_dict
        
        # self.user_embedding = user_embedding
        # self.item_id_embedding = item_embedding
        self.user_embedding = nn.Embedding(self.n_user, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_item, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        self.dir_str = dataset
        pre_adj_mat = sp.load_npz('/home/share/yangxuanhui/dataset/' + self.dir_str + '/s_pre_adj_mat.npz')
        self.norm_adj = pre_adj_mat
        self.adj = graph

        # self.norm_adj = self.get_adj_mat()
        # self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().cuda()
        # self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().cuda()

        dataset_path = os.path.abspath('/home/share/yangxuanhui/dataset/' + self.dir_str)
        image_adj_file = os.path.join(dataset_path,'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        audio_adj_file = os.path.join(dataset_path, 'audio_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        
        self.v_feat = v_feat
        self.a_feat = a_feat
        self.t_feat = t_feat

        if v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(image_adj, image_adj_file) 
            self.image_original_adj = image_adj.cuda()
        
        if a_feat is not None:
            self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat, freeze=False)
            if os.path.exists(audio_adj_file):
                audio_adj = torch.load(audio_adj_file)
            else:
                audio_adj = build_sim(self.audio_embedding.weight.detach())
                audio_adj = build_knn_normalized_graph(audio_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(audio_adj, audio_adj_file)
            self.audio_original_adj = audio_adj.cuda()
        
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        if v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if a_feat is not None:
            self.audio_trs = nn.Linear(self.a_feat.shape[1], self.embedding_dim)
        if t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        
        self.soft_max = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim,self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim,1,bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim,self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_a = nn.Sequential(
            nn.Linear(self.embedding_dim,self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim,self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim,self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_audio_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim,self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim,self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5

    # def  pre_epoch_processing(self):
    #     pass
        
    # def get_adj_mat(self):
    #     adj_mat = sp.dok_matrix((self.n_user + self.n_item, self.n_user+self.n_item), dtype=np.float16)
    #     adj_mat = adj_mat.tolil()
    #     R = self.interaction_matrix.tolil()

    #     adj_mat[:self.n_user, self.n_user:] = R 
    #     adj_mat[self.n_user:, :self.n_user] = R.T 
    #     adj_mat = adj_mat.todok()

    #     def normalized_adj_single(adj):
    #         rowsum = np.array(adj.sum(1))

    #         d_inv = np.power(rowsum,-0.5).flatten()
    #         d_inv[np.isinf(d_inv)] = 0. 
    #         d_mat_inv = sp.diags(d_inv)

    #         norm_adj = d_mat_inv.dot(adj_mat)
    #         norm_adj = norm_adj.dot(a_mat_inv)
    #         return norm_adj.tocoo() 
             
    #     norm_adj_mat = normalized_adj_single(adj_mat)
    #     norm_adj_mat = norm_adj_mat.tolil()
    #     self.R = norm_adj_mat[:self.n_user, self.n_user:]
    #     return norm_adj_mat.tocsr()
        
    # def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
    #     sparse_mx = sparse_mx.tocoo().astype(np.float16)
    #     indices = torch.from_numpy(np.vstack((saprse_mx.row, sparse_mx.col)).astype(np.int32))
    #     values = torch.from_numpy(sparse_mx.data)
    #     shape = torch.Size(sparse_mx.shape)
    #     return torcha.sparse.FloatTensor(indices, values, shape)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def forward(self,train=False):
        adj = self.adj
        R = self.norm_adj[:self.n_user, self.n_user:]
        self.R = self._convert_sp_mat_to_sp_tensor(R).cuda()
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.a_feat is not None:
            audio_feats = self.audio_trs(self.audio_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            
        #behavior-guided purifier
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_feats))
        if self.a_feat is not None:
            audio_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_a(audio_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_feats))

        #user-item view
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        #item-item view
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        if self.a_feat is not None:
            if self.sparse:
                for i in range(self.n_layers):
                    audio_item_embeds = torch.sparse.mm(self.audio_original_adj, audio_item_embeds)
            else:
                for i in range(self.n_layers):
                    audio_item_embeds = torch.mm(self.audio_original_adj, self.audio_item_embeds)
            audio_user_embeds = torch.sparse.mm(self.R, audio_item_embeds)
            audio_embeds = torch.cat([audio_user_embeds, audio_item_embeds], dim=0)
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        #behaviour-aware fuser
        if self.a_feat is not None:
            att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds), self.query_common(audio_embeds)], dim=-1)
        else:
            att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)],dim=-1)
        weight_common = self.soft_max(att_common)
        if self.a_feat is not None:
            common_embeds = weight_common[:,0].unsqueeze(dim=1) * image_embeds + weight_common[:,1].unsqueeze(dim=1) * text_embeds + weight_common[:,2].unsqueeze(dim=1) * audio_embeds
        else:
            common_embeds = weight_common[:,0].unsqueeze(dim=1) * image_embeds + weight_common[:,1].unsqueeze(dim=1) * text_embeds
        sep_image_embeds = image_embeds - common_embeds
        if self.a_feat is not None:
            sep_audio_embeds = audio_embeds - common_embeds
        sep_text_embeds = text_embeds - common_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        if self.a_feat is not None:
            audio_prefer = self.gate_audio_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)

        sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        if self.a_feat is not None:
            sep_audio_embeds = torch.multiply(audio_prefer, sep_image_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
        if self.a_feat is not None:
            side_embeds = (sep_image_embeds + sep_audio_embeds + sep_text_embeds)/3 + common_embeds
        else:
            side_embeds = (sep_image_embeds + sep_text_embeds)/2 + common_embeds

        all_embeds = content_embeds + side_embeds
        all_embeddings_user, all_embeddings_item = torch.split(all_embeds, [self.n_user, self.n_item])
        if train :
            return all_embeddings_user, all_embeddings_item, side_embeds, content_embeds
        return all_embeddings_user, all_embeddings_item 
    


            

        




