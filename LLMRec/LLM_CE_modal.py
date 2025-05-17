import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import (
    LlamaModel,
    LlamaForCausalLM,
    AutoTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)
import os

from pretrained_lightgcn import LightGCN

from MMGCN import Net 
from torch.nn.functional import cosine_similarity
from MGCN import MGCN

import numpy as np


class LLM4Rec(nn.Module):
    def __init__(self, **args):
        super(LLM4Rec, self).__init__()
        self.args = args
        self.input_dim, self.output_dim = args['input_dim'], args['output_dim']
        self.hidden_dim = 64    # todo test other parameters
        self.dataset = args['dataset']

        self.user_num, self.item_num, self.graph = self.args['lightgcn_parm']

        self.weight = torch.tensor([[1.], [-1.]]).cuda()
        
        self.dropout = 0.3
        self.reg_weight = 0.1
        self.cl_weight = 1.0
        self.reg_loss = EmbLoss()

        self.train_edge, self.user_item_dict = self.args['mmgcn_parm']


        print(f'Initializing language decoder ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type='FEATURE_EXTRACTION',
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=self.args['lora_target_modules'],
            bias='none',
        )

        self.llama_model = LlamaModel.from_pretrained(self.args['base_model'],
        #load_in_8bit=True, 
        torch_dtype=torch.float16,
                                                      local_files_only=True, cache_dir=args['cache_dir'],
                                                      device_map=self.args['device_map'])
        self.llama_model = prepare_model_for_kbit_training(self.llama_model)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()
        self.llama_model.config.use_cache = False

        self.llama_tokenizer = AutoTokenizer.from_pretrained(self.args['base_model'], use_fast=False, local_files_only=True, cache_dir=args['cache_dir'])
        self.llama_tokenizer.pad_token = "0"
        self.llama_tokenizer.padding_side = "right"
        self.instruct_ids, self.instruct_mask = self.llama_tokenizer(self.args['instruction_text'][0],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        self.response_ids, self.response_mask = self.llama_tokenizer(self.args['instruction_text'][1],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        print('Language decoder initialized.')

        self.task_type = args['task_type']

        self.lightgcn = LightGCN(self.dataset, self.user_num, self.item_num, self.graph, latent_dim=self.input_dim, n_layers=3)
        self.score_weight = 0.7
        # self.user_embeds = nn.Parameter(self.args['user_embeds'])
        # self.user_embeds = nn.Embedding.from_pretrained(self.args['user_embeds'], freeze=True)
        # self.user_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size)
        self.user_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
        )
        # self.input_embeds = nn.Parameter(self.args['input_embeds'])
        # self.input_embeds = nn.Embedding.from_pretrained(self.args['input_embeds'], freeze=True)
        # self.input_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
        )
        # self.score = nn.Linear(self.llama_model.config.hidden_size, self.output_dim, bias=False)
        
        user_embedding_lgcn, item_embedding_lgcn = self.lightgcn()
        user_embedding_lgcn = user_embedding_lgcn.cpu()
        user_embedding_lgcn = user_embedding_lgcn.detach().numpy()
        item_embedding_lgcn = item_embedding_lgcn.cpu()
        item_embedding_lgcn = item_embedding_lgcn.detach().numpy()

        np.save('/home/share/yangxuanhui/embedding/lgcn_user.npy', user_embedding_lgcn)
        np.save('/home/share/yangxuanhui/embedding/lgcn_item.npy', item_embedding_lgcn)

        v_feat = self.args["modal_feat"][0] 
        a_feat = self.args["modal_feat"][1] if self.dataset != "kwai" else None
        t_feat = self.args["modal_feat"][2]
        self.MMGCN = Net(v_feat,a_feat,t_feat,None,self.train_edge,self.user_num,self.item_num,'mean','false',3,True,self.user_item_dict)
        self.MGCN = MGCN(v_feat,a_feat,t_feat,2,64,10,1,self.user_num,self.item_num,user_embedding_lgcn, item_embedding_lgcn, self.graph, self.dataset)
        ##### new
        self.v_feat = nn.Embedding.from_pretrained(self.args["modal_feat"][0], freeze=True)
        self.a_feat = nn.Embedding.from_pretrained(self.args["modal_feat"][1], freeze=True) if self.dataset != "kwai" else None
        self.t_feat = nn.Embedding.from_pretrained(self.args["modal_feat"][2], freeze=True)

        self.v_proj = nn.Sequential(
            nn.Linear(self.v_feat.weight.size(1), self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
        )
        # self.a_proj = nn.Linear(self.a_feat.weight.size(1), self.llama_model.config.hidden_size)
        if self.dataset != "kwai":
            self.a_proj = nn.Sequential(
                nn.Linear(self.a_feat.weight.size(1), self.llama_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
            )
        # self.t_proj = nn.Linear(self.t_feat.weight.size(1), self.llama_model.config.hidden_size)
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_feat.weight.size(1), self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
        )
        self.pred = nn.Sequential(
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.hidden_dim),
        )
        self.item_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.hidden_dim),
        )
        self.modal_proj = nn.Sequential(
            nn.Linear(self.llama_model.config.hidden_size*3, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size)
        )
        ##### new #####

    def predict(self, inputs, inputs_mask):
        bs = inputs.shape[0]
        instruct_embeds = self.llama_model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self.llama_model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        self.user_embeds, self.input_embeds = self.lightgcn()
        #print(self.user_embeds.shape)
        
        ### new embedding
        self.modal_embeds = self.MMGCN()
        user_embedding = self.modal_embeds[:self.user_num]
        item_embedding = self.modal_embeds[self.user_num:]
        user_embedding_s = user_embedding[inputs[:,0]]
        #print(user_embedding_s.shape)
        #print(item_embedding.shape)
        score_matrix_mmgcn = torch.matmul(user_embedding_s, item_embedding.T)
        user_embedding = user_embedding.cpu()
        user_embedding = user_embedding.detach().numpy()
        item_embedding = item_embedding.cpu()
        item_embedding = item_embedding.detach().numpy()

        #np.save('/home/share/yangxuanhui/embedding/user.npy', user_embedding)
        #np.save('/home/share/yangxuanhui/embedding/item.npy', item_embedding)
        ###
        ###MGCN embed
        self.mgcn_user, self.mgcn_item = self.MGCN()
        mgcn_user_embedding = self.mgcn_user
        mgcn_item_embedding = self.mgcn_item
        scores = torch.matmul(mgcn_user_embedding, mgcn_item_embedding.T)

        mgcn_user_embedding = mgcn_user_embedding.cpu()
        mgcn_user_embedding = mgcn_user_embedding.detach().numpy()
        mgcn_item_embedding = mgcn_item_embedding.cpu()
        mgcn_item_embedding = mgcn_item_embedding.detach().numpy()

        np.save('/home/share/yangxuanhui/embedding/mgcn_user.npy', mgcn_user_embedding)
        np.save('/home/share/yangxuanhui/embedding/mgcn_item.npy', mgcn_item_embedding)

        

        ##BM3
        u_online_ori, i_online_ori = self.user_embeds[inputs[:,0]], self.input_embeds[inputs[:,1:]]
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.t_feat(inputs[:,1:])
        if self.v_feat is not None:
            v_feat_online = self.v_feat(inputs[:,1:])

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)
            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)
        u_online, i_online = self.user_proj(u_online_ori), self.input_proj(i_online_ori)
        u_target, i_target = self.user_proj(u_target), self.input_proj(i_target)

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = self.t_proj(t_feat_online)
            t_feat_target = self.t_proj(t_feat_target)
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if self.v_feat is not None:
            v_feat_online = self.v_proj(v_feat_online)
            v_feat_target = self.v_proj(v_feat_target)
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()
    

        loss_ui = 1 - cosine_similarity(u_online.unsqueeze(1), i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.unsqueeze(1).detach(), dim=-1).mean()

        #print(loss_iu, loss_ui, loss_t, loss_tv, loss_v, loss_vt)
        #print(self.reg_loss(u_online_ori, i_online_ori))

        loss_BM3 = (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()


        ###


        users = self.user_proj(self.user_embeds[inputs[:, 0].unsqueeze(1)])
        items = self.input_proj(self.input_embeds[inputs[:, 1:]])

        ###
        score_matrix_ori = torch.matmul(self.user_embeds[inputs[:,0]], self.input_embeds.T)
        ###

        user_emb = self.user_embeds[inputs[:, 0]]
        #print(user_emb.shape)

        # user_res = self.user_res(self.user_embeds[inputs[:, 0]])

        ##### new #####
        v = self.v_proj(self.v_feat(inputs[:, 1:]))
        a = self.a_proj(self.a_feat(inputs[:, 1:])) if self.dataset != "kwai" else None
        t = self.t_proj(self.t_feat(inputs[:, 1:]))
        modal_feat = v+a+t if self.dataset != "kwai" else v+t
        # modal_feat = self.modal_proj(torch.cat((v, a, t), dim=2))
        ##### new #####

        items_mask = inputs_mask[:, 1:]

        inputs = torch.cat([users, items], dim=1)
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.llama_model(inputs_embeds=inputs, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.last_hidden_state[:, -1]
        # pooled_logits = self.score(pooled_output)
        # pred_emb = self.pred(pooled_output)
        self.pred_emb = self.pred(pooled_output)
        #print(self.pred_emb.shape)
        self.item_emb = self.item_proj(self.input_embeds)
        #print(self.item_emb.shape)
        #print(self.input_embeds.shape)
        #score_matrix = torch.matmul(self.pred_emb, self.item_emb.T)
        self.fin_user = self.score_weight*user_emb+(1-self.score_weight)*self.pred_emb
        self.fin_item = self.score_weight*self.input_embeds+(1-self.score_weight)*self.item_emb
        #score_matrix = torch.matmul(self.fin_user, self.fin_item.T)
        score_matrix = self.score_weight * torch.matmul(self.fin_user, self.fin_item.T) + (1-self.score_weight) * torch.matmul(self.pred_emb, self.item_emb.T)

        # return outputs, pooled_logits.view(-1, self.output_dim)
        return outputs, score_matrix, loss_BM3, score_matrix_mmgcn

    def forward(self, inputs, inputs_mask, labels, neg_item):
        # outputs, pooled_logits = self.predict(inputs, inputs_mask)
        outputs, score_matrix, loss_BM3, _, = self.predict(inputs, inputs_mask)
        #loss_bm3 = loss_BM3.item()

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(score_matrix, labels)
            loss += 0.01*((self.user_embeds**2).mean() + (self.input_embeds**2).mean())
            loss += 1*((self.pred_emb**2).mean() + (self.item_emb**2).mean())
            loss = 0.5*loss_BM3 + 0.5*loss

            # idx = torch.arange(0, len(labels))
            # pos_score, neg_score = score_matrix[idx, labels].view(-1, 1), score_matrix[idx, neg_item].view(-1, 1)
            # score = torch.cat((pos_score, neg_score), dim=1)
            # loss += -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight)))).cuda()
            # loss += 0.01*((self.user_embeds**2).mean() + (self.input_embeds**2).mean())
            
        return {
            "loss":loss,
            "inputs": inputs,
            "score_matrix": score_matrix,
        }

        # return SequenceClassifierOutputWithPast(
        #     loss=loss,
        #     logits=pooled_logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss