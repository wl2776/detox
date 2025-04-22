#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from tqdm import trange


def get_args(argv):
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-i', '--input', type=Path, help='input tsv', required=True)
    parser.add_argument('-o', '--output', type=Path, help='output tsv')
    parser.add_argument('-m', '--model', type=Path, help='path to model')
    return parser.parse_args(argv)


LANG_PROMPTS = {
    'zh': '排毒：',
    'es': 'Desintoxicar: ',
    'ru': 'Детоксифицируй: ',
    'ar': 'إزالة السموم: ',
    'hi': 'विषहरण: ',
    'hin': 'विषहरण: ',
    'uk': 'Детоксифікуй: ',
    'de': 'Entgiften: ',
    'am': 'መርዝ መርዝ: ',
    'en': 'Detoxify: ',
    'it': 'Disintossicare: ',
    'fr': 'Désintoxiquer: ',
    'he': 'לְסַלֵק רַעַל: ',
    'tt': 'Детоксификация: ',
    'ja': '解毒'
}


def preprocess_function(examples, tokenizer, max_length=256):
    inputs = [LANG_PROMPTS[lang] + text for lang, text in zip(examples['lang'], examples['toxic_sentence'])]
    
    model_inputs = tokenizer(inputs, max_length=max_length,
                             truncation=True, padding='max_length')

    return model_inputs


def generate_predictions(model, tokenizer, dataset, batch_size=32, device='cpu'):
    predictions = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for start_idx in trange(0, len(dataset), batch_size):
            end_idx = min(start_idx + batch_size, len(dataset))
            
            batch = dataset[start_idx:end_idx]
            input_ids = torch.tensor(batch['input_ids'], device=device)
            attention_mask = torch.tensor(batch['attention_mask'], device=device)
            
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
            
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(decoded_outputs)
    
    return predictions

def main():
    args = get_args(sys.argv[1:])

    df = pd.read_csv(args.input, sep='\t')
    dataset = Dataset.from_pandas(df.drop('neutral_sentence', axis=1))

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True, 
                                  fn_kwargs={'tokenizer': tokenizer})  #, 'max_length': config.tokenizer.max_length})
    
    df['neutral_sentence'] = generate_predictions(model, tokenizer, encoded_dataset, device='cuda')
    df['neutral_sentence'].fillna('')
    df.rename(columns={'toxic_sentence': 'toxic_text', 'neutral_sentence': 'neutral_text'}, inplace=True)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(args.output, sep='\t', header=True, index=False)


if __name__ == '__main__':
    main()
