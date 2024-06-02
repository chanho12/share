# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
import numpy as np
import os
from itertools import chain
from . import raw_datasets
from tqdm import tqdm
import json

DATASET_CACHE_DIR = "/xllm3-ft/data/cache/"

def get_raw_dataset(dataset_name, output_path, seed, local_rank, train_data_path=""):
    if dataset_name == "Dahoas/rm-static":
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank)
    elif dataset_name == "Dahoas/full-hh-rlhf":
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank)
    elif dataset_name == "Dahoas/synthetic-instruct-gptj-pairwise":
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank)
    elif dataset_name == "yitingxie/rlhf-reward-datasets":
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank)
    elif dataset_name == "openai/webgpt_comparisons":
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank)
    elif dataset_name == "stanfordnlp/SHP":
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank)
    elif dataset_name == "wangrui6/Zhihu-KOL":
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank)
    elif dataset_name == "Cohere/miracl-zh-queries-22-12":
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank)
    elif dataset_name == "Hello-SimpleAI/HC3-Chinese":
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank)
    elif dataset_name == "mkqa-Chinese":
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank)
    elif dataset_name == "mkqa-Japanese":
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank)
    elif dataset_name == "Cohere/miracl-ja-queries-22-12":
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank)
    elif dataset_name == "lmqg/qg_jaquad":
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank)
    elif dataset_name == "lmqg/qag_jaquad":
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank)
    elif dataset_name == 'xiaoice_label_datasets':
        return raw_datasets.XiaoiceLabelDataset(output_path, seed, local_rank, train_data_path)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, train_data_path=""):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank, train_data_path)
    print(len(raw_dataset))
    raise Exception('debug!!!')
    train_dataset = raw_dataset.get_train_data()
    eval_dataset = raw_dataset.get_eval_data()
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                            raw_dataset.dataset_name_clean,
                                            seed, "train", data_split,
                                            train_phase - 1,
                                            len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                        train_phase, tokenizer,
                                        end_of_conversation_token,
                                        max_seq_len)

    
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                            raw_dataset.dataset_name_clean,
                                            seed, "eval",
                                            data_split, train_phase - 1,
                                            len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          train_data_path=""):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = '_'.join(data_path)
    tokenizer_name = tokenizer.init_kwargs['name_or_path'].replace('/', '_')
    fname = '_'.join(fname.split('/'))
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    # Skip creating cache if we found it on all the nodes.
    if buf_create_cache.item() == 0:
        return torch.load(train_fname), torch.load(eval_fname)
    else:
        train_dataset, eval_dataset = create_dataset(
            local_rank, data_path[0], data_split, output_path, train_phase,
            seed, tokenizer, end_of_conversation_token, max_seq_len, train_data_path=train_data_path)
        
        if local_rank <= 0:
            torch.save(train_dataset, train_fname)
            torch.save(eval_dataset, eval_fname)
        return train_dataset, eval_dataset


def infer_dataset_columns(datapath):
    with open(datapath, "r", encoding="utf-8") as file:
        line = file.readline()
        return list(json.loads(line).keys())
    

def get_unsupervised_data(args, tokenizer, data_files, train_phase=1, streaming=False):
    data_columns = infer_dataset_columns(data_files)
    unsupervised_raw_datasets = load_dataset("json", 
                                            data_files=data_files, 
                                            streaming=streaming,
                                            split="train", 
                                            name="unsupdata")
    
    if train_phase==1:
        max_len = args.max_seq_len
    else:
        max_len = args.max_prompt_seq_len + args.max_answer_seq_len

    def _single_label_masking_phase1(label, bos_id):
        """
        find all bos position
        """
        bos_indices = [i for i,bi in enumerate(label) if bi==bos_id]
        if len(bos_indices) > 1:
            for i,(_start,_end) in enumerate(zip(bos_indices[:-1],bos_indices[1:])):
                if i%2==0:
                    for j in range(_start, _end):
                        label[j] = -100
        return label

    def tokenize_function(examples):
        # for instruction data, single data process

        if train_phase==1:
            text = examples['prompt']
            _single_label_masking = _single_label_masking_phase1
        
        text = examples['prompt'] + examples['last_speaker'] + ' : ' +examples['answer'] 
        text = text.rstrip()

        ret = {}
        text = text + tokenizer.eos_token

        inputs = tokenizer(text, add_special_tokens=True, padding='max_length', 
                           max_length=max_len, truncation=True)
        
        ret["input_ids"] = inputs["input_ids"]
        ret["attention_mask"] = inputs["attention_mask"]
        ret["labels"] = _single_label_masking(ret["input_ids"].copy(), tokenizer.bos_token_id)

        if ret["input_ids"][-1] in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
            ret["valid"] = 1
        else:
            ret["valid"] = 0
        
        return ret
    
    # do tokenize
    if streaming:
         # streaming 没有num_proc
        unsupervised_raw_datasets = unsupervised_raw_datasets.map(tokenize_function, remove_columns=data_columns)
        unsupervised_raw_datasets = unsupervised_raw_datasets.filter(lambda x: True if x["valid"] == 1 else False)
    else:
        unsupervised_raw_datasets = unsupervised_raw_datasets.map(tokenize_function, remove_columns=data_columns, num_proc=10)
        unsupervised_raw_datasets = unsupervised_raw_datasets.filter(lambda x: True if x["valid"] == 1 else False, num_proc=10)
    unsupervised_raw_datasets = unsupervised_raw_datasets.remove_columns(["valid"])
    # unsupervised_raw_datasets = unsupervised_raw_datasets.shuffle(seed=42) # 反正在外面意见shuffle好，加了这个速度会变慢
    return unsupervised_raw_datasets