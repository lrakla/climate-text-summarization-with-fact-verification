
import os
import pandas as pd
from datasets import load_dataset, Dataset
from summarizer.sbert import SBertSummarizer

import transformers
from evaluate import load
from huggingface_hub import notebook_login
from transformers import AutoTokenizer,pipeline
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import torch
import os
import pandas as pd
import nltk
import numpy as np
import nltk
nltk.download('punkt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
    def __init__(self,dataset,model_name, tokenizer_name, args, final_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.args = args
        self.dataset = dataset
        self.final_model_name = final_model_name
        self.prefix = "summarize: "
        self.metric = load(self.args['metric'])

    def preprocess_function(self,samples):
        inputs = [self.prefix + doc for doc in samples["Body"]]
        model_inputs = self.tokenizer(inputs, max_length=self.args['max_input_length'], truncation=True)
        labels = self.tokenizer(text_target=samples["Saved"], max_length=self.args['max_target_length'], truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

   
    def training_args(self):
        self.training_args = Seq2SeqTrainingArguments(
        self.final_model_name,
        evaluation_strategy = self.args['eval_strategy'],
        logging_strategy=self.args['eval_strategy'],
        learning_rate= self.args['lr'],
        per_device_train_batch_size=self.args['batch_size'],
        per_device_eval_batch_size=self.args['batch_size'],
        gradient_accumulation_steps = self.args[ 'gradient_accumulation_steps'],
        weight_decay=self.args['decay'],
        save_total_limit=self.args['save_total_limit'],
        save_strategy =  self.args['save_strategy'],
        num_train_epochs=self.args['epochs'],
        predict_with_generate=True,
        load_best_model_at_end=self.args['load_best_model'],
        fp16=self.args['fp16']
    #     push_to_hub=True,
        )
    def compute_metrics(self,eval_pred):
        predictions, labels = eval_pred
        self.decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        self.decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in self.decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in self.decoded_labels]

        # Note that other metrics may not have a `use_aggregator` parameter
        # and thus will return a list, computing a metric for each sentence.
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
        # Extract a few results
        result = {key: value for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    def run_trainer(self,save_model = False):
        tokenized_dataset = self.dataset.map(self.preprocess_function, batched=True)    
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        self.training_args()
        trainer = Seq2SeqTrainer(
                    self.model,
                    self.training_args,
                    train_dataset=tokenized_dataset["train"],
                    eval_dataset=tokenized_dataset["test"],
                    data_collator=data_collator,
                    tokenizer=self.tokenizer,
                    compute_metrics=self.compute_metrics
                )
        trainer.train()
        if save_model:
            self.model.save_pretrained(self.final_model_name, from_pt=True) 
            self.tokenizer.save_pretrained(self.final_model_name)
        return self.decoded_preds, self.decoded_labels