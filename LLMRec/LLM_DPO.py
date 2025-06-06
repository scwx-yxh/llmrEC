import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import (
    LlamaModel,
    LlamaForCausalLM,
    LlamaTokenizer,
)
# from llama_test.llama_model import LlamaModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)
import os

from LightGCN_pretrained import LightGCN


class LLM4Rec(nn.Module):
    def __init__(self, **args):
        super(LLM4Rec, self).__init__()
        self.args = args
        self.dataset = args["dataset"]
        self.input_dim, self.output_dim = args['input_dim'], args['output_dim']
        self.hidden_dim = 64    # todo

        self.user_num, self.item_num, self.graph = self.args['lightgcn_parm']

        self.weight = torch.tensor([[1.], [-1.]]).cuda()

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
        self.llama_model = prepare_model_for_int8_training(self.llama_model)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()
        self.llama_model.config.use_cache = False

        # ========================================ref_model=======================================================

        self.llama_model_ref = LlamaModel.from_pretrained(self.args['base_model'], load_in_8bit=True, torch_dtype=torch.float16,
                                                      local_files_only=True, cache_dir=args['cache_dir'],
                                                      device_map=self.args['device_map'])
        self.llama_model_ref = prepare_model_for_int8_training(self.llama_model_ref)
        self.llama_model_ref.config.use_cache = False
        parameter_names = [n for n, _ in self.llama_model_ref.named_parameters()]
        for param_name in parameter_names:
            param = self.llama_model_ref.get_parameter(param_name)
            param.requires_grad = False
        self.llama_model_ref.eval()

        # ========================================ref_model=======================================================

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.args['base_model'], use_fast=False, local_files_only=True, cache_dir=args['cache_dir'])
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

    def predict(self, inputs, inputs_mask):
        bs = inputs.shape[0]
        instruct_embeds = self.llama_model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self.llama_model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        self.user_embeds, self.input_embeds = self.lightgcn()

        users = self.user_proj(self.user_embeds[inputs[:, 0].unsqueeze(1)])
        items = self.input_proj(self.input_embeds[inputs[:, 1:]])

        user_emb = self.user_embeds[inputs[:, 0]]

        inputs = torch.cat([users, items], dim=1)
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs_theta = self.llama_model(inputs_embeds=inputs, attention_mask=attention_mask, return_dict=True)
        pooled_output_theta = outputs_theta.last_hidden_state[:, -1]
        outputs_ref = self.llama_model_ref(inputs_embeds=inputs, attention_mask=attention_mask, return_dict=True)
        pooled_output_ref = outputs_ref.last_hidden_state[:, -1]

        # pooled_logits = self.score(pooled_output)
        # pred_emb = self.pred(pooled_output)
        self.pred_emb_theta = self.pred(pooled_output_theta) # (batch_size, hs)
        self.pred_emb_ref = self.pred(pooled_output_ref)
        self.item_emb = self.item_proj(self.input_embeds) # (item_num, hs)
        # score_matrix = torch.matmul(pred_emb, item_emb.T)
        # score_matrix = self.score_weight*torch.matmul(user_emb, self.input_embeds.T) + (1-self.score_weight)*torch.matmul(self.pred_emb, self.item_emb.T)
        self.fin_user_theta = self.score_weight*user_emb+(1-self.score_weight)*self.pred_emb_theta
        self.fin_user_ref = self.score_weight*user_emb+(1-self.score_weight)*self.pred_emb_ref
        self.fin_item = self.score_weight*self.input_embeds+(1-self.score_weight)*self.item_emb

        # self.fin_user_theta = self.pred(pooled_output_theta) # (batch_size, hs)
        # self.fin_user_ref = self.pred(pooled_output_ref)
        # self.fin_item = self.item_proj(self.input_embeds) # (item_num, hs)

        score_matrix_theta = torch.matmul(self.fin_user_theta, self.fin_item.T) # (batch_size, item_num)
        score_matrix_ref = torch.matmul(self.fin_user_ref, self.fin_item.T) # (batch_size, item_num)
        # return outputs, pooled_logits.view(-1, self.output_dim)
        return score_matrix_theta, score_matrix_ref, self.fin_user_theta, self.fin_user_ref, self.fin_item

    def forward(self, inputs, inputs_mask, labels, neg_item):
        # outputs, pooled_logits = self.predict(inputs, inputs_mask)
        score_matrix_theta, score_matrix_ref, user_theta, user_ref, item = self.predict(inputs, inputs_mask)

        loss = None
        if labels is not None:
            # # CE Loss
            # loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(score_matrix, labels)
            # loss += 0.01*((self.user_embeds**2).mean() + (self.input_embeds**2).mean())
            # loss += 1*((self.pred_emb**2).mean() + (self.item_emb**2).mean())
            
            # # BPR Loss
            # idx = torch.arange(0, len(labels))
            # pos_score, neg_score = score_matrix[idx, labels].view(-1, 1), score_matrix[idx, neg_item].view(-1, 1)
            # score = torch.cat((pos_score, neg_score), dim=1)
            # loss += -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight)))).cuda()
            # loss += 0.01*((self.user_embeds**2).mean() + (self.input_embeds**2).mean())

            # DPO Loss
            idx = torch.arange(0, len(labels))
            pos_score_theta, neg_score_theta = score_matrix_theta[idx, labels].view(-1, 1), score_matrix_theta[idx, neg_item].view(-1, 1)
            pos_score_ref, neg_score_ref = score_matrix_ref[idx, labels].view(-1, 1), score_matrix_ref[idx, neg_item].view(-1, 1)
            beta = 0.05
            chosen_reward = beta*(torch.log(pos_score_theta) - torch.log(pos_score_ref))
            rejected_reward = beta*(torch.log(neg_score_theta) - torch.log(neg_score_ref))
            batch_loss = (-1)*torch.log(torch.sigmoid(chosen_reward - rejected_reward))
            loss = torch.nanmean(batch_loss)
            
        return {
            "loss":loss,
            "inputs": inputs,
            "score_matrix": score_matrix_theta,
        }

        # return SequenceClassifierOutputWithPast(
        #     loss=loss,
        #     logits=pooled_logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
