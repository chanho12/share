# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])

    args = parser.parse_args()

    return args


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
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
                                    clean_up_tokenization_spaces=False)
    return result

def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


def prompt_eval(args, baseline_model, baseline_tokenizer,finetuned_model,finetuned_tokenizer, device,
                prompts):
        prompt = prompts['prompt']
        dialogue = prompts['dialogue']
        

        b_inputs = baseline_tokenizer(prompt + dialogue, return_tensors="pt").to(device)
        f_inputs = finetuned_tokenizer(prompt + dialogue, return_tensors="pt").to(device) 

        print(dialogue)

        print("==========Baseline: Greedy=========")
        b_output = generate(baseline_model,
                          baseline_tokenizer,
                          b_inputs,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)
        print_utils(b_output)
        print("==========finetune: Greedy=========")
        f_output = generate(finetuned_model,
                                finetuned_tokenizer,
                                f_inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        print_utils(f_output)
        # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # We provide examples as below for users to try.

        # print("==========finetune: Multinomial sampling=========")
        # r_finetune_m = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=1,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_m)
        # print("==========finetune: Beam Search=========")
        # r_finetune_b = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_b)
        # print("==========finetune: Beam-search multinomial sampling=========")
        # r_finetune_s = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_s)
        # print("==========finetune: Diverse Beam Search=========")
        # r_finetune_d = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_beam_groups=args.num_beam_groups,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_d)
        # print("==========finetune: Constrastive Search=========")
        # r_finetune_c = generate_constrastive_search(model_fintuned, tokenizer, inputs,
        #                                             top_k=args.top_k,
        #                                             penalty_alpha=args.penalty_alpha,
        #                                             num_return_sequences=args.num_return_sequences,
        #                                             max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_c)
        print("====================prompt end=============================")

def make_hf_model(path, device):

    tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    config = AutoConfig.from_pretrained(path)

    model = AutoModelForCausalLM.from_pretrained(
    path,
    from_tf=bool(".ckpt" in path),
    config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

def make_peft_model(path, device):

    config = PeftConfig.from_pretrained(path)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,fast_tokenizer=True)

    model_finetuned = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

    model = PeftModel.from_pretrained(model_finetuned, path)

    return model, tokenizer


def main():
    args = parse_args()

    device = torch.device("cuda:0")

    path = 'google/gemma-2b' 

    baseline_model, baseline_tokenizer = make_hf_model(path,device)

    peft_model, peft_tokenizer = peft_model(args.peft_model_name_or_path, device)

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    
    prompts = {'prompt' : '' , 'dialogue': ''}

    prompt_eval(args, baseline_model, baseline_tokenizer,peft_model, peft_tokenizer, device,
                prompts)

if __name__ == "__main__":
    main()
