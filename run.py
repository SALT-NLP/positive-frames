#!pip install sentence_transformers
#!pip install sacrebleu
#!pip install --upgrade bleu
#!pip install rouge
#!pip install datasets
#!pip install rouge_score
#%tensorflow_version 1.x (not required any more)
#!pip install -q gpt-2-simple

import argparse
import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer, util
import os
from transformers import Trainer, pipeline, set_seed, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
nltk.download('punkt')
import csv
from datasets import load_dataset, load_metric



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='bart', choices=['random', 'sbert', 'bart', 't5', 'gpt', 'gpt2', 'seq2seqlstm'])
    parser.add_argument('-s', '--setting', default='unconstrained', choices=['unconstrained', 'controlled', 'predict'])
    parser.add_argument('--train', default='data/wholetrain.csv') #default is for bart/t5; data format will be different for GPT
    parser.add_argument('--dev', default='data/wholedev.csv')
    parser.add_argument('--test', default='data/wholetest.csv')
    parser.add_argument('--output_dir', type=str, default='output/')
    args = parser.parse_args()
    return args

#Retrieval
def run_random():
    train = pd.read_csv(train_path)
    original_text = train['original_with_label'].tolist()
    hyp = []
    for i in range(835):
        hyp.append(random.choice(original_text))
    with open(os.path.join(path,'random.txt'), 'w') as f:
        for item in hyp:
            f.write("%s\n" % item)

def run_sbert():
    train = pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    original_text = train['original_with_label'].tolist()
    train_embedding=[]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for i in range(len(original_text)):
        match = [original_text[i]]
        embeddings2 = model.encode(match, convert_to_tensor=True)
        train_embedding.append(embeddings2)
    hyp2=[]
    original_text2 = test['original_with_label'].tolist()
    reframed_text = train['reframed_text'].tolist()
    for prompt in original_text2:
        prompt=[prompt]
        embeddings1 = model.encode(prompt, convert_to_tensor=True)
        index = 0
        max = 0
        for i in range(len(original_text)):
            #match = [original_text[i]]
            embeddings2 = train_embedding[i]
            value=util.pytorch_cos_sim(embeddings1, embeddings2)[0][0].item()
            if (value>max):
                index=i
                max=value
        hyp2.append(reframed_text[index])
    print('here')
    with open(os.path.join(path,'sbert.txt'), 'w') as f:
        for item in hyp2:
            f.write("%s\n" % item)

#BART
def run_bart_unconstrained():
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    def preprocess_function(examples):
        inputs = examples["original_text"]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["reframed_text"], max_length=1024, truncation=True) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

    from transformers import BartForConditionalGeneration, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        #result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {key: value.mid.fmeasure for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        # Add bleu scores
        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model("output/reframer")
    # Load trained model
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'bart_unconstrained.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)
            
def run_bart_controlled():
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    def preprocess_function(examples):
        inputs = examples["original_with_label"]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["reframed_text"], max_length=1024, truncation=True) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

    from transformers import BartForConditionalGeneration, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        #result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {key: value.mid.fmeasure for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        # Add bleu scores
        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model("output/reframer")
    # Load trained model
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")

    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_with_label'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'bart_controlled.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)
    
def run_bart_predict():
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    def preprocess_function(examples):
        inputs = examples["original_text"]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["strategy_reframe"])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

    from transformers import BartForConditionalGeneration, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        #result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {key: value.mid.fmeasure for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        # Add bleu scores
        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model("output/reframer")
    # Load trained model
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'bart_predict.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)

#T5
def run_t5_unconstrained(): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #print(tokenizer("Hello, this one sentence!")) #a test
    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["original_text"]]
        model_inputs = tokenizer(inputs) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["reframed_text"]) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)

        # print(result2)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    # save model
    trainer.save_model("output/reframer") #TODO

    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_unconstrained.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)

def run_t5_controlled(): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #print(tokenizer("Hello, this one sentence!")) #a test
    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["original_with_label"]]
        model_inputs = tokenizer(inputs) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["reframed_text"]) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)

        # print(result2)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    # save model
    trainer.save_model("output/reframer") #TODO

    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_with_label'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_controlled.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)

def run_t5_predict(): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #print(tokenizer("Hello, this one sentence!")) #a test
    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["original_text"]]
        model_inputs = tokenizer(inputs) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["strategy_reframe"]) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)

        # print(result2)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_dev_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    # save model
    trainer.save_model("output/reframer") #TODO

    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_predict.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)

#GPT
def run_gpt_unconstrained():
    from transformers import TextDataset,DataCollatorForLanguageModeling
    def load_dataset(train_path,test_path,tokenizer):
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=train_path,
            block_size=50)
        test_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=test_path,
            block_size=50)   
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )
        return train_dataset,test_dataset,data_collator
    tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
    train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)

    from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
    model = AutoModelWithLMHead.from_pretrained("openai-gpt")
    #model.config.max_length=100
    n_epoch = 5
    training_args = TrainingArguments(
        output_dir="./gpt", #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=n_epoch, # number of training epochs
        per_device_train_batch_size=6, # batch size for training
        per_device_eval_batch_size=6,  # batch size for evaluation
        eval_steps = 200, # Number of update steps between two evaluations.
        save_steps=400, # after # steps model is saved 
        warmup_steps=300,# number of warmup steps for learning rate scheduler
        prediction_loss_only=True,
        learning_rate = 3e-5
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    trainer.save_model("gptreframer")

    #predict
    model = AutoModelWithLMHead.from_pretrained("gptreframer")
    reframe = pipeline('text-generation',model=model, tokenizer='openai-gpt')
    import csv
    with open(test_path, newline='') as data:
        annotations = csv.DictReader(data, delimiter=',', quotechar='"')
        annotations_list = list(annotations)
        for i in range(0,835):
            prefix = "<|startoftext|> " + annotations_list[i]['original_text'] + "\nreframed:"
            #print(prefix)
            gen_text = reframe(prefix, max_length=100)[0]['generated_text']
            text_file = open('gpt_unconstrained/gpt_gentext_{i}.txt'.format(i=i), "w")
            text_file.write(gen_text)
            text_file.close()

#GPT2
def run_gpt2_unconstrained():
    import gpt_2_simple as gpt2
    gpt2.download_gpt2(model_name="124M")
    import tensorflow as tf
    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                dataset=file_name,
                model_name='124M',
                steps=1000,
                restore_from='fresh',
                run_name='reframer',
                learning_rate=0.00001,
                print_every=10,
                sample_every=250,
                save_every=1000
                )
    #gpt2.copy_checkpoint_to_gdrive(run_name='reframer')
    #gpt2.copy_checkpoint_from_gdrive(run_name='reframer')
    #predict
    import tensorflow as tf
    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name='reframer')

    with open(test_path, newline='') as data:
        annotations = csv.DictReader(data, delimiter=',', quotechar='"')
        annotations_list = list(annotations)
        for i in range(0,835):
            prefix = "<|startoftext|> " + annotations_list[i]['original_text'] + "\nreframed:"
            #reframed_text = annotations_list[i]['reframed_text']
            gen_file = 'gpt2_unconstrained/gpt2_gentext_{i}.txt'.format(i=i)
            gpt2.generate_to_file(sess,
                        run_name="reframer",
                        destination_path=gen_file,
                        length=50,
                        truncate="<|endoftext|>",
                        prefix=prefix,
                        include_prefix=False,
                        nsamples=1,
                        batch_size=1
                        )

#Seq2SeqLSTM
def run_seq2seqlstm_unconstrained():
    return None


def main():
    #run models
    if args.model=='random':
        run_random()
    elif args.model=='sbert':
        run_sbert()
    elif args.model=='bart' and args.setting=='unconstrained':
        run_bart_unconstrained()
    elif args.model=='bart' and args.setting=='controlled':
        run_bart_controlled()
    elif args.model=='bart' and args.setting=='predict':
        run_bart_predict()
    elif args.model=='t5' and args.setting=='unconstrained':
        run_t5_unconstrained()
    elif args.model=='t5' and args.setting=='controlled':
        run_t5_controlled()
    elif args.model=='t5' and args.setting=='predict':
        run_t5_predict()
    elif args.model=='gpt' and args.setting=='unconstrained':
        run_gpt_unconstrained()
    elif args.model=='gpt2' and args.setting=='unconstrained':
        run_gpt2_unconstrained()
    elif args.model=='seq2seqlstm' and args.setting=='unconstrained':
        run_seq2seqlstm_unconstrained()

if __name__=='__main__':
    args = parse_args()
    model = args.model

    #load datasets
    if model in ['random', 'sbert', 'bart', 't5']:
        train_path = args.train
        train_dataset = load_dataset('csv', data_files=train_path)
        dev_path = args.dev
        dev_dataset = load_dataset('csv', data_files=train_path)
        test_path = args.test
        test_dataset = load_dataset('csv', data_files=test_path)
    elif model in ['gpt', 'gpt2']:
        train_path = args.train
        test_path = args.test


    else:
        raise Exception("Sorry, this model is currently not included.")

    path=args.output_dir
    main()