import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import random
import torch
import time
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

class BipartiteGraphDataset(Dataset):
    def __init__(self, dataset):
        super(BipartiteGraphDataset, self).__init__()
        self.dataset = dataset

        self.n_user, self.m_item = 0, 0
        self.trainData, self.allPos, self.testData = [], {}, {}

        if self.dataset == "movielens":
            self.n_user, self.m_item = 55485, 5986
            self.dir_str = "/home/share/yangxuanhui/dataset/movielens"
        elif dataset == "tiktok":
            self.n_user, self.m_item = 36656, 76085
            self.dir_str = "/home/share/yangxuanhui/dataset/tiktok"
        elif dataset == "kwai":
            self.n_user, self.m_item = 7010, 86483
            self.dir_str = "/home/share/yangxuanhui/dataset/kwai"
        else:
            raise KeyError("Dataset not found.")
        
        self.allPos = np.load(self.dir_str+'/user_item_dict.npy', allow_pickle=True).item()
        self.trainData = np.load(self.dir_str+'/train.npy', allow_pickle=True)
        
        test_data = np.load(self.dir_str+'/test.npy', allow_pickle=True)
        self.testData = {pair[0]: pair[1:] for pair in test_data}
        
        self.get_sparse_graph()


    def __getitem__(self, idx):
        user, pos_item = self.trainData[idx]
        random.shuffle(self.allPos[user])

        while True:
            neg_item = random.randint(0, self.m_item-1)
            if neg_item not in self.allPos[user]:
                break

        return user, self.allPos[user], pos_item, neg_item

    def __len__(self):
        return len(self.trainData)
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        print('loading adjacency matrix')
        try:
            pre_adj_mat = sp.load_npz(self.dir_str + '/s_pre_adj_mat.npz')
            print('successfully loaded...')
            norm_adj = pre_adj_mat
        except:
            print('generating adjacency matrix')
            adj_mat = sp.dok_matrix((self.n_user+self.m_item, self.n_user+self.m_item), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            train_user = self.trainData[:, 0]
            train_item = self.trainData[:, 1]
            R = csr_matrix((np.ones(len(train_user)), (train_user, train_item)), shape=(self.n_user, self.m_item)).tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T
            adj_mat = adj_mat.todok()
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum + 1e-5, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            
            sp.save_npz(self.dir_str + '/s_pre_adj_mat.npz', norm_adj)
        
        self.graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.graph = self.graph.coalesce().cuda()
    


class Val_Dataset(Dataset):
    def __init__(self, dataset, type="val", max_len=50):
        super(Val_Dataset, self).__init__()
        self.dataset = dataset
        self.type = type
        self.max_len = max_len

        self.n_user, self.m_item = 0, 0
        self.allPos, self.valData = {}, {}

        if self.dataset == "movielens":
            self.n_user, self.m_item = 55485, 5986
            dir_str = "/home/share/yangxuanhui/dataset/movielens"
        elif self.dataset == "tiktok":
            self.n_user, self.m_item = 36656, 76085
            dir_str = "/home/share/yangxuanhui/dataset/tiktok"
        elif self.dataset == "kwai":
            self.n_user, self.m_item = 7010, 86483
            dir_str = "/home/share/yangxuanhui/dataset/kwai"
        
        self.allPos = np.load(dir_str+'/user_item_dict.npy', allow_pickle=True).item()

        if self.type == "val":
            self.data = np.load(dir_str+'/val.npy', allow_pickle=True)    # todo
        elif self.type == "test":
            self.data = np.load(dir_str+'/test.npy', allow_pickle=True)
            self.testData = {pair[0]: pair[1:] for pair in self.data}
        else:
            raise KeyError("Dataset not found.")


    def __getitem__(self, idx):
        user, labels = self.data[idx][0], random.choice(self.data[idx][1:])
        random.shuffle(self.allPos[user])

        while True:
            neg_item = random.randint(0, self.m_item-1)
            if neg_item not in self.allPos[user]:
                break
        
        if self.type == "val":
            return user, self.allPos[user], labels, neg_item
        elif self.type == "test":
            items = self.allPos[user]
            item_len = min(self.max_len, len(items))
            input = [user] + items[:self.max_len] + [0] * (self.max_len - item_len)
            # if self.modal == "true":
            #     input_mask = [1] + [1] * item_len * (1+self.modal_num) + [0] * (self.max_len - item_len) * (1+self.modal_num)
            # else:
            input_mask = [1] + [1] * item_len + [0] * (self.max_len - item_len)
            # print(len(input), len(input_mask), labels, neg_item)
            return torch.LongTensor(input), torch.LongTensor(input_mask), torch.LongTensor([labels]), torch.LongTensor([neg_item]), torch.LongTensor([item_len])
            

    def __len__(self):
        return len(self.data)
        # return 1000


@dataclass
class BipartiteGraphCollator:
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, batch) -> dict:
        user, items, labels, neg_item = zip(*batch)
        bs = len(user)
        # max_len = max([len(item) for item in items])
        inputs = [[user[i]] + items[i][:self.max_len] + [0] * (self.max_len - min(self.max_len, len(items[i]))) for i in range(bs)]
        # if self.modal == "true":
        #     inputs_mask = [[1] + [1] * min(self.max_len, len(items[i])) * (1+self.modal_num) + [0] * (self.max_len - min(self.max_len, len(items[i]))) * (1+self.modal_num) for i in range(bs)]
        # else:
        inputs_mask = [[1] + [1] * min(self.max_len, len(items[i])) + [0] * (self.max_len - min(self.max_len, len(items[i]))) for i in range(bs)]
        inputs, inputs_mask, labels, neg_item = torch.LongTensor(inputs), torch.LongTensor(inputs_mask), torch.LongTensor(labels), torch.LongTensor(neg_item)

        return {
            "inputs": inputs,
            "inputs_mask": inputs_mask,
            "labels": labels,
            "neg_item": neg_item,
        }