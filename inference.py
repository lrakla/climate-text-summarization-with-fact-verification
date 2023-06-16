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
import tqdm
import torch
import os
import pandas as pd
import nltk
import numpy as np
nltk.download('punkt')
#https://pypi.org/project/bert-extractive-summarizer/#use-sbert

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GetSummary:
    def __init__(self, dataset,summary_type, model_name = None, tokenizer_name = None, test = False, add_to_dataset = False,finetuned =False):  
        self.dataset = dataset['train'] if not test else dataset['test']
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.summary_type = summary_type
        self.test = test
        self.add_to_dataset = add_to_dataset
        self.prefix = 'summarize :'
        self.finetuned = finetuned

    
    def get_summary(self, args):
        self.args = args
        if self.summary_type == 'extractive':
            extractive = self._get_extractive_summary(self.args['min_length'])
            if self.add_to_dataset:
                self.dataset = self.add_column(extractive,'Extractive')
                return self.dataset, extractive
            return extractive
        else:    
            if self.finetuned:
                abstractive =  self._get_abstractive_from_finetuned()   
            else:
                abstractive = self._get_abstractive_summary()
            return abstractive
        
    def  _get_abstractive_from_finetuned(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
        summarizer = pipeline("summarization", model=model, tokenizer = tokenizer, batch_size = self.args['batch_size'])
        body = []
        for i in tqdm(range(len(self.dataset))):
            text = self.prefix+ self.dataset[i]['Body']
            body.append(text)
        return [sample['summary_text'] for sample in summarizer(body)]
        
    def _get_extractive_summary(self, min_length):
        model = SBertSummarizer(self.model_name)
        extractive = []
        for sample in self.dataset:
            body = sample['Body']
            result = model(body, min_length=min_length)
            extractive.append(result)
        return extractive
    
    def _get_abstractive_summary(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
        abstractive = []
        for sample in self.dataset:
            text =  self.prefix + sample['Body']
            inputTokens = tokenizer(text, return_tensors="pt",max_length=self.args['max_new_length'], truncation=True).to(device)
            outputs = model.generate(inputTokens['input_ids'], attention_mask=inputTokens['attention_mask'],\
                                     max_new_tokens = self.args['max_new_length'])
            abstractive.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        return abstractive
    
    def add_column(self,col_vals, col_name, dataset = 'NA'):
        if dataset != 'NA':
            dataset = dataset.add_column(col_name, col_vals)
            return dataset
        self.dataset = self.dataset.add_column(col_name, col_vals)
        return self.dataset
            

        