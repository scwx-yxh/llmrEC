"""
无模态信息 纯ID信息的程序入口
"""

import os
import sys
from typing import List

from tqdm import tqdm

import fire
import torch
import pickle
import random
import numpy as np
import transformers
from utils.prompter import Prompter
from LLM_CE import LLM4Rec
from utils.data_utils import BipartiteGraphDataset, BipartiteGraphCollator, Val_Dataset, set_seed
from utils.eval_utils import RecallPrecision_atK, MRR_atK, MAP_atK, NDCG_atK, AUC, getLabel, Trainer_Eval, full_accuracy
from torch.utils.data import DataLoader

def train(
    # model/data params
    base_model: str = "", 
    data_path: str = "",
    cache_dir: str = "",
    output_dir: str = "",
    task_type: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    seed: int = 2024,
    learning_rate: float = 3e-4,
    max_len: int = 50,
    cutoff_len: int = 4096,
    val_set_size: int = 0,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 100, 
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # from peft docs: ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out", "wte", "gate_proj", "down_proj", "up_proj"]
    lora_target_modules: List[str] = ["gate_proj", "down_proj", "up_proj"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca"
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"cache_dir: {cache_dir}\n"
            f"output_dir: {output_dir}\n"
            f"task_type: {task_type}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"seed: {seed}\n"
            f"learning_rate: {learning_rate}\n"
            f"max_len: {max_len}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"warmup_steps: {warmup_steps}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )

    set_seed(seed=seed)

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size


    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if task_type == 'general':
        train_dataset = BipartiteGraphDataset(data_path)
        val_dataset = Val_Dataset(data_path, type="val")
        data_collator = BipartiteGraphCollator(max_len=max_len)

    prompter = Prompter(prompt_template_name)

    model = LLM4Rec(
        base_model=base_model,
        task_type=task_type,
        cache_dir=cache_dir,
        input_dim=64,
        dataset=data_path,
        output_dim=train_dataset.m_item,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        device_map=device_map,
        instruction_text=prompter.generate_prompt(task_type),
        lightgcn_parm=[train_dataset.n_user, train_dataset.m_item, train_dataset.graph],
    )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    eval = Trainer_Eval(data_path, topk=100, max_len=max_len)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            # max_steps=10000,
            learning_rate=learning_rate,
            # dataloader_num_workers=16,
            # fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=1000 if val_set_size > 0 else None,
            save_steps=1000,
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="none",
            run_name=None,
            metric_for_best_model="recall@10",
        ),
        data_collator=data_collator,
        compute_metrics=eval.compute_metrics,
        preprocess_logits_for_metrics=eval.preprocess_logits_for_metrics,
        callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.eval()

    topk, dataloader_bs = 10, 4 if model == "true" else 10
    test_dataset = Val_Dataset(data_path, type="test", max_len=max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=dataloader_bs, shuffle=False, num_workers=8)

    all_index_of_rank_list_10, all_index_of_rank_list_20, users = torch.LongTensor([]), torch.LongTensor([]), []
    user_feature, item_feature = None, None

    pbar = tqdm(total=len(test_dataset))
    for inputs, inputs_mask, labels, neg_item, item_len in test_dataloader:
        with torch.no_grad():
            _, score_matrix, fin_user, fin_item = model.predict(inputs.cuda(), inputs_mask.cuda())

        if user_feature == None:
            user_feature= fin_user.cpu()
        else:
            user_feature = torch.cat((user_feature.cpu(), fin_user.cpu()), dim=0)
        item_feature = fin_item.cpu()
        # print(user_feature.size()) 
        
        user, items = inputs[:, 0], inputs[:, 1:]
        users += user.tolist()

        for row in range(score_matrix.size(0)):
            for col in range(item_len[row]):
                score_matrix[row][items[row][col]] = 1e-5

        _, index_of_rank_list_10 = torch.topk(score_matrix, 10)
        all_index_of_rank_list_10 = torch.cat((all_index_of_rank_list_10, index_of_rank_list_10.cpu()), dim=0)

        all_index_of_rank_list_10 = all_index_of_rank_list_10.view(-1, 10)

        _, index_of_rank_list_20 = torch.topk(score_matrix, 20)
        all_index_of_rank_list_20 = torch.cat((all_index_of_rank_list_20, index_of_rank_list_20.cpu()), dim=0)

        all_index_of_rank_list_20 = all_index_of_rank_list_20.view(-1, 20)

        pbar.update(dataloader_bs)

    pbar.close()

    torch.save(user_feature, output_dir+"user")
    torch.save(item_feature, output_dir+"item")

    precision_10, recall_10, ndcg_10, hit_rate_10 = full_accuracy(all_index_of_rank_list_10, users, test_dataset.testData, 10)
    precision_20, recall_20, ndcg_20, hit_rate_20 = full_accuracy(all_index_of_rank_list_20, users, test_dataset.testData, 20)
    
    print(f'Precision@10: {precision_10} \n '
            f'Recall@10: {recall_10} \n '
            f'NDCG@10: {ndcg_10} \n'
            f'HIT_RATE@10: {hit_rate_10} \n '
            f'Precision@20: {precision_20} \n '
            f'Recall@20: {recall_20} \n '
            f'NDCG@20: {ndcg_20} \n'
            f'HIT_RATE@20: {hit_rate_20} \n ')


    model.llama_model.save_pretrained(output_dir)
    model_path = os.path.join(output_dir, "adapter.pth")
    user_proj, input_proj, pred, item_proj = model.user_proj.state_dict(), model.input_proj.state_dict(), model.pred.state_dict(), model.item_proj.state_dict()
    torch.save({'user_proj': user_proj, 'input_proj': input_proj, 'pred': pred, "item_proj": item_proj}, model_path)


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)
