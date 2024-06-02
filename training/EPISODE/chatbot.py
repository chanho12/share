import torch

from peft import PeftModel, PeftConfig

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from utils.model_utils import get_model

def make_hf_model(path, device):

    tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    config = AutoConfig.from_pretrained(path)
    model = get_model(config, path, tokenizer)

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


def peft_model(path, device):

    config = PeftConfig.from_pretrained(path)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,fast_tokenizer=True)

    model= AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,load_in_8bit=True)

    model = PeftModel.from_pretrained(model, path)

    model.eval()
    
    return model, tokenizer















