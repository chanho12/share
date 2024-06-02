# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
)
from transformers.deepspeed import HfDeepSpeedConfig

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig, prepare_model_for_kbit_training

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig
)

# from transformers.models.gptj.configuration_gptj import GPTJConfig
# from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig

### For training model code


def create_model(model_name_or_path):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_config.dropout = 0.0

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=model_config,
        trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer = True)


    return model, tokenizer


def get_lora_model(model_name, model_path_, ds_config):
    peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.1,
            bias = 'none'
            )
    
    if model_name == 'llama':
        print(f'This model is {model_name}')

        model_path = 'meta-llama/Llama-2-7b-chat-hf'

        model,tokenizer = create_model(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    elif model_name == 'quan_gemma':
        print(f'This model is {model_name}')
        model_path = 'google/gemma-7b'
        
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                   fast_tokenizer=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_path ,quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    elif model_name == 'gemma':
        print(f'This model is {model_name}')
        model_path = 'google/gemma-2b'

        model,tokenizer = create_model(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        

    elif model_name == 'quan_llama':
        print(f'This model is {model_name}')
        model_path = 'meta-llama/Llama-2-7b-chat-hf'
        
        tokenizer = LlamaTokenizer.from_pretrained(model_path,
                                                   fast_tokenizer=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = LlamaForCausalLM.from_pretrained(model_path ,quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    else:
        print('bring a checkpoint')
        config = PeftConfig.from_pretrained(model_path_)

        model,tokenizer = create_model(config.base_model_name_or_path)

        model = PeftModel.from_pretrained(model, model_path_,is_trainable=True)
        model.print_trainable_parameters()

    return model, tokenizer

### Infernece model. 

def get_peft_checkpoint(path, device):

    config = PeftConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,fast_tokenizer=True)
    model= AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,load_in_8bit=True)
    model = PeftModel.from_pretrained(model, path)

    model.to(device)
    model.eval()
    
    return model, tokenizer


def get_peft_checkpoint_(path, device):

    config = PeftConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,fast_tokenizer=True)
    model= AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, path)

    model.to(device)
    model.eval()
    
    return model, tokenizer



def get_eval_model(config, model_path, tokenizer):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config)
    model.resize_token_embeddings(len(tokenizer))

    # prepare the tokenizer and model config
    tokenizer.pad_token = tokenizer.eos_token
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    return model

def eval_model(path, device):

    tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    config = AutoConfig.from_pretrained(path)
    model = get_eval_model(config, path, tokenizer)

    model.eval()
    model.to(device)

    return model, tokenizer



def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=True,
             num_return_sequences=1,
             max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]
    
    return result