import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import (
    LlamaModel,
    LlamaForCausalLM,
    AutoTokenizer,
)
# from new_llama_model import LlamaModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from modal_encoder import ModalEncoder, ModalEncoderLayer

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)
import os

from pretrained_lightgcn import LightGCN
from MMGCN import Net 


class LLM4Rec(nn.Module):
    def __init__(self, **args):
        super(LLM4Rec, self).__init__()
        self.args = args
        self.input_dim, self.output_dim = args['input_dim'], args['output_dim']
        self.hidden_dim = 64    # todo test other parameters
        self.dataset = args['dataset']

        self.user_num, self.item_num, self.graph = self.args['lightgcn_parm']

        self.weight = torch.tensor([[1.], [-1.]]).cuda()

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

        self.llama_model = LlamaModel.from_pretrained(self.args['base_model'], load_in_8bit=True, torch_dtype=torch.float16,
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
        self.score_weight = 0.9
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

        v_feat = self.args["modal_feat"][0] 
        a_feat = self.args["modal_feat"][1] if self.dataset != "kwai" else None
        t_feat = self.args["modal_feat"][2]
        self.MMGCN = Net(v_feat,a_feat,t_feat,None,self.train_edge,self.user_num,self.item_num,'mean','False',2,True,self.user_item_dict)

        ##### new add
        self.v_feat = nn.Embedding.from_pretrained(F.normalize(self.args["modal_feat"][0]), freeze=True)
        self.a_feat = nn.Embedding.from_pretrained(F.normalize(self.args["modal_feat"][1]), freeze=True) if self.dataset != "kwai" else None
        self.t_feat = nn.Embedding.from_pretrained(F.normalize(self.args["modal_feat"][2]), freeze=True)

        encoder_dim = 512
        
        self.v_proj = nn.Sequential(
            nn.Linear(self.v_feat.weight.size(1), self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, encoder_dim),
        )
        # self.a_proj = nn.Linear(self.a_feat.weight.size(1), self.llama_model.config.hidden_size)
        if self.dataset != "kwai":
            self.a_proj = nn.Sequential(
                nn.Linear(self.a_feat.weight.size(1), self.llama_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.llama_model.config.hidden_size, encoder_dim),
            )
        # self.t_proj = nn.Linear(self.t_feat.weight.size(1), self.llama_model.config.hidden_size)
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_feat.weight.size(1), self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, encoder_dim),
        )
        self.item_out_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, encoder_dim),   
        )
        self.v_out_proj = nn.Sequential(
            nn.Linear(self.v_feat.weight.size(1), self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, encoder_dim),
        )
        if self.dataset != "kwai":
            self.a_out_proj = nn.Sequential(
                nn.Linear(self.a_feat.weight.size(1), self.llama_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.llama_model.config.hidden_size, encoder_dim),
            )
        self.t_out_proj = nn.Sequential(
            nn.Linear(self.t_feat.weight.size(1), self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, encoder_dim),
        )
        self.item_proj = nn.Sequential(
            nn.Linear(encoder_dim*2, self.llama_model.config.hidden_size),
            # nn.Linear(self.input_dim, self.llama_model.config.hidden_size),
            # nn.Linear(self.input_dim+self.v_feat.weight.size(1)+self.a_feat.weight.size(1)+self.t_feat.weight.size(1), self.llama_model.config.hidden_size),
            # nn.Linear(self.input_dim+encoder_dim, self.llama_model.config.hidden_size),
            # nn.Linear(encoder_dim, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.hidden_dim),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, encoder_dim),
        )
        self.modal_proj = nn. Sequential(
            nn.Linear(encoder_dim*2, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.llama_model.config.hidden_size, self.hidden_dim),
        )
        self.modal_encoder_layer = ModalEncoderLayer(word_emb_dim=encoder_dim, nhead=2)
        self.modal_encoder = ModalEncoder(self.modal_encoder_layer, num_layers=3)
        self.modal_dense = nn.Linear(encoder_dim, self.hidden_dim)
        ##### new #####

    def predict(self, inputs, inputs_mask):
        bs = inputs.shape[0]
        instruct_embeds = self.llama_model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self.llama_model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        self.user_embeds, self.input_embeds = self.lightgcn()

        self.modal_embeds = self.MMGCN()
        print(self.modal_embeds.shape)
        
        users = self.user_proj(self.user_embeds[inputs[:, 0].unsqueeze(1)])
        items = self.input_proj(self.input_embeds[inputs[:, 1:]])
        # modal = self.modal_proj(self.modal_embeds[inputs[:,0]])
        origin_score = torch.matmul(self.user_embeds,self.input_embeds.T)

        user_emb = self.user_embeds[inputs[:, 0]]

        # user_res = self.user_res(self.user_embeds[inputs[:, 0]])

        ##### new ##### 
        v = self.v_proj(self.v_feat(inputs[:, 1:]))

        if self.dataset != "kwai":
            a = self.a_proj(self.a_feat(inputs[:, 1:]))
        
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
        # pooled_output = outputs.last_hidden_state[:, -1]

        ls_len = attention_mask.size(1)
        encoder_inputs = torch.cat((self.output_proj(outputs.last_hidden_state), modal_feat), dim=1).transpose(0, 1)
        encoder_inputs_mask = (torch.cat((torch.ones_like(attention_mask), items_mask), dim=1) == 0)
        self.pred_emb = self.modal_dense(self.modal_encoder(src=encoder_inputs, ls_len=ls_len, src_key_padding_mask=encoder_inputs_mask)[0])

        # pooled_logits = self.score(pooled_output)
        # pred_emb = self.pred(pooled_output)
        # pred_emb = self.pred(pooled_output)
        # self.item_emb = self.item_proj(self.input_embeds)
        # temp_modal_feat = self.v_proj(self.v_feat.weight)+self.a_proj(self.a_feat.weight)+self.t_proj(self.t_feat.weight)
        # self.item_emb = self.item_proj(torch.cat((self.input_embeds, self.v_feat.weight, self.a_feat.weight, self.t_feat.weight), dim=1))
        # self.item_emb = self.item_proj(torch.cat((self.input_embeds, temp_modal_feat), dim=1))
        # self.item_emb = self.item_proj(temp_modal_feat)
        temp_modal_feat = self.v_out_proj(self.v_feat.weight)+self.a_out_proj(self.a_feat.weight)+self.t_out_proj(self.t_feat.weight) if self.dataset != "kwai" else self.v_out_proj(self.v_feat.weight)+self.t_out_proj(self.t_feat.weight)
        self.item_emb = self.item_proj(torch.cat((self.item_out_proj(self.input_embeds), temp_modal_feat), dim=1))
        # score_matrix = torch.matmul(self.pred_emb, self.item_emb.T)
        # score_matrix = self.score_weight*torch.matmul(user_emb, self.input_embeds.T) + (1-self.score_weight)*torch.matmul(self.pred_emb, self.item_emb.T)
        self.fin_user = self.score_weight*user_emb+(1-self.score_weight)*self.pred_emb
        self.fin_item = self.score_weight*self.input_embeds+(1-self.score_weight)*self.item_emb
        score_matrix = torch.matmul(self.fin_user, self.fin_item.T)

        # return outputs, pooled_logits.view(-1, self.output_dim)
        return outputs, score_matrix, self.fin_user, self.fin_item

    def forward(self, inputs, inputs_mask, labels, neg_item):
        # outputs, pooled_logits = self.predict(inputs, inputs_mask)
        outputs, score_matrix, _, _ = self.predict(inputs, inputs_mask)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(score_matrix, labels)
            loss += 0.01*((self.user_embeds**2).mean() + (self.input_embeds**2).mean())
            loss += 1*((self.pred_emb**2).mean() + (self.item_emb**2).mean())

            # idx = torch.arange(0, len(labels))
            # pos_score, neg_score = score_matrix[idx, labels].view(-1, 1), score_matrix[idx, neg_item].view(-1, 1)
            # score = torch.cat((pos_score, neg_score), dim=1)
            # loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight)))).cuda()
            # # loss += 0.01*((self.user_embeds**2).mean() + (self.input_embeds**2).mean())
            # loss += 0.001*((self.pred_emb**2).mean() + (self.item_emb**2).mean())
            
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