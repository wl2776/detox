#!/usr/bin/env python3
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import datasets
from datasets import load_dataset


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


def preprocess_function(examples, tokenizer):
    inputs = [LANG_PROMPTS[lang] + text for lang, text in zip(examples['language'], examples['toxic_sentence'])]
    targets = examples['neutral_sentence']
    
    model_inputs = tokenizer(inputs, max_length=max(len(s) for s in inputs),
                             truncation=True, padding='max_length')

    labels = tokenizer(targets, max_length=max(len(s) for s in targets),
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
    dataset = load_dataset('textdetox/multilingual_paradetox')
    dataset = add_language_column(dataset)

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    encoded_dataset = dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    train_data = encoded_dataset.train_test_split(test_size=0.2)


    # Аргументы тренировки
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        save_total_limit=2,
        learning_rate=1e-4,
        gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data['train'],
        eval_dataset=train_data['test'],
    )

    trainer.train()

    model.save_pretrained('./final_model')
    tokenizer.save_pretrained('./final_model')


if __name__ == '__main__':
    main()