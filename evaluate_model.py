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

class Evaluate:
    def __init__(self, metric, tokenizer_name):
        self.metric = metric
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def compute_metrics(self,preds, labels):
        metric = load(self.metric)
        sent_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
        sent_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
        scores = metric.compute(predictions=sent_preds, references=sent_labels)
        result = {key: value for key, value in scores.items()}
        # Add mean generated length
        prediction_len = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in sent_preds]
        result["gen_len"] = np.mean(prediction_len)
        return {k: round(v, 4) for k, v in result.items()}