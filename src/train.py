#!/usr/bin/env python3
import sys
import os
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import datasets
from datasets import load_dataset
from omegaconf import OmegaConf
import mlflow


def get_args(argv):
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Config file path')
    return parser.parse_args(argv)


LANG_PROMPTS = {
    'zh': '排毒：',
    'es': 'Desintoxicar: ',
    'ru': 'Детоксифицируй: ',
    'ar': 'إزالة السموم: ',
    'hi': 'विषहरण: ',
    'uk': 'Детоксифікуй: ',
    'de': 'Entgiften: ',
    'am': 'መርዝ መርዝ: ',
    'en': 'Detoxify: ',
}


def preprocess_function(examples, tokenizer, max_length=256):
    inputs = [LANG_PROMPTS[lang] + text for lang, text in zip(examples['language'], examples['toxic_sentence'])]
    targets = examples['neutral_sentence']
    
    model_inputs = tokenizer(inputs, max_length=max_length,
                             truncation=True, padding='max_length')

    labels = tokenizer(targets, max_length=max_length,
                       truncation=True, padding='max_length').input_ids
    
    model_inputs['labels'] = [[label if label != tokenizer.pad_token_id else -100 for label in l] for l in labels]
    return model_inputs


def  add_language_column(dataset):
    preprocessed_dataset = []
    for lang_code, sub_dataset in dataset.items():
        def add_language_column(example):
            example['language'] = lang_code
            return example        
        processed_sub_dataset = sub_dataset.map(add_language_column)
        preprocessed_dataset.append(processed_sub_dataset)
    merged_dataset = datasets.concatenate_datasets(preprocessed_dataset)
    return merged_dataset
     

def main():
    args = get_args(sys.argv[1:])
    config = OmegaConf.load(args.config)

    dataset = load_dataset('textdetox/multilingual_paradetox')
    dataset = add_language_column(dataset)

    tokenizer = T5Tokenizer.from_pretrained(config.model_type)
    model = T5ForConditionalGeneration.from_pretrained(config.model_type)

    encoded_dataset = dataset.map(preprocess_function, batched=True, 
                                  fn_kwargs={'tokenizer': tokenizer, 'max_length': config.tokenizer.max_length})
    train_data = encoded_dataset.train_test_split(test_size=0.2)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", config.mlflow.uri))
    mlflow.set_experiment(config.mlflow.experiment_name)

    training_args = TrainingArguments(**config.train)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data['train'],
        eval_dataset=train_data['test'],
    )

    with mlflow.start_run() as run:
        trainer.train()

    model.save_pretrained('./final_model')
    tokenizer.save_pretrained('./final_model')


if __name__ == '__main__':
    main()
