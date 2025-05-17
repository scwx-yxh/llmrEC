import pandas as pd
import torch
import numpy as np
import math
from sklearn.metrics import roc_auc_score


# ====================Metrics==============================
def RecallPrecision_atK(test, r, k):
    tp = r[:, :k].sum(1)
    precision = np.sum(tp) / k
    recall_n = np.array([len(test[i]) for i in range(len(test))])
    recall = np.sum(tp / recall_n)
    return precision, recall


def MRR_atK(test, r, k):
    pred = r[:, :k]
    weight = np.arange(1, k+1)
    MRR = np.sum(pred / weight, axis=1) / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    MRR = np.sum(MRR)
    return MRR


def MAP_atK(test, r, k):
    pred = r[:, :k]
    rank = pred.copy()
    for i in range(k):
        rank[:, k - i - 1] = np.sum(rank[:, :k - i], axis=1)
    weight = np.arange(1, k+1)
    AP = np.sum(pred * rank / weight, axis=1)
    AP = AP / np.array([len(test[i]) if len(test[i]) <= k else k for i in range(len(test))])
    MAP = np.sum(AP)
    return MAP


def NDCG_atK(test, r, k):
    pred = r[:, :k]
    test_mat = np.zeros((len(pred), k))
    for i, items in enumerate(test):
        length = k if k <= len(items) else len(items)
        test_mat[i, :length] = 1

    idcg = np.sum(test_mat * (1. / np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.] = 1.
    dcg = pred * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    ndcg = np.sum(ndcg)
    return ndcg


def AUC(all_item_scores, dataset, test):
    r_all = np.zeros((dataset.m_item, ))
    r_all[test] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test, pred):
    r = []
    for i in range(len(test)):
        groundTruth, predTopK = test[i], pred[i]
        hits = list(map(lambda x: x in groundTruth, predTopK))
        hits = np.array(hits).astype("float")
        r.append(hits)
    return np.array(r).astype('float')
# ====================end Metrics=============================

class Trainer_Eval:
    def __init__(self, dataset, topk, max_len):
        self.topk = topk
        self.max_len = max_len
        if dataset == "movielens":
            dir_str = "/home/share/yangxuanhui/dataset/movielens"
        elif dataset == "tiktok":
            dir_str = "/home/share/yangxuanhui/dataset/tiktok"
        elif dataset == "kwai":
            dir_str = "/home/share/yangxuanhui/dataset/kwai"
        
        self.allPos = np.load(dir_str+'/user_item_dict.npy', allow_pickle=True).item()
        val_data = np.load(dir_str+'/val.npy', allow_pickle=True)   # todo
        self.valData = {pair[0]: pair[1:] for pair in val_data}

    def compute_metrics(self, pred):
        pre, labels = pred

        rank_list_10, rank_list_20, user = torch.tensor(pre[0]), torch.tensor(pre[1]), pre[2]

        precision_10, recall_10, ndcg_10, hit_rate_10 = full_accuracy(rank_list_10, user, self.valData, 10)
        precision_20, recall_20, ndcg_20, hit_rate_20 = full_accuracy(rank_list_20, user, self.valData, 20)

        return {
            "precision@10": precision_10,
            "recall@10": recall_10,
            "ndcg@10": ndcg_10,
            "hit_rate@10": hit_rate_10,
            "precision@20": precision_20,
            "recall@20": recall_20,
            "ndcg@20": ndcg_20,
            "hit_rate@20": hit_rate_20,
        }

    def preprocess_logits_for_metrics(self, logits, labels):
        inputs, score_matrix = logits

        user, items = inputs[:, 0], inputs[:, 1:]

        all_index_of_rank_list_10, all_index_of_rank_list_20 = torch.LongTensor([]), torch.LongTensor([])

        for row in range(score_matrix.size(0)):
            item_len = min(self.max_len, len(self.allPos[int(user[row])]))
            for col in range(item_len):
                score_matrix[row][items[row][col]] = 1e-5
            
        _, index_of_rank_list_10 = torch.topk(score_matrix, 10)
        all_index_of_rank_list_10 = torch.cat((all_index_of_rank_list_10, index_of_rank_list_10.cpu()), dim=0)

        all_index_of_rank_list_10 = all_index_of_rank_list_10.view(-1, 10)

        _, index_of_rank_list_20 = torch.topk(score_matrix, 20)
        all_index_of_rank_list_20 = torch.cat((all_index_of_rank_list_20, index_of_rank_list_20.cpu()), dim=0)

        all_index_of_rank_list_20 = all_index_of_rank_list_20.view(-1, 20)

        return all_index_of_rank_list_10, all_index_of_rank_list_20, user


def full_accuracy(rank_list, user, pos_list, topk):
    length = 0
    precision = recall = ndcg = 0.0
    total_hit = total_pos_item = 0

    for i in range(rank_list.size(0)):
        pos_items = set(pos_list[user[i]] if user[i] in pos_list.keys() else [])
        num_pos = len(pos_items)
        if num_pos == 0:
            continue
        length += 1
        items_list = rank_list[i].tolist()

        items = set(items_list)

        num_hit = len(pos_items.intersection(items))
        total_hit += num_hit
        total_pos_item += num_pos

        precision += float(num_hit / topk)
        recall += float(num_hit / num_pos)
        
        ndcg_score = 0.0
        max_ndcg_score = 0.0

        for i in range(min(num_pos, topk)):
            max_ndcg_score += 1 / math.log2(i+2)
        if max_ndcg_score == 0:
            continue

        for i, temp_item in enumerate(items_list):
            if temp_item in pos_items:
                ndcg_score += 1 / math.log2(i+2)

        ndcg += ndcg_score / max_ndcg_score

    return precision / length, recall / length, ndcg / length, total_hit / total_pos_item