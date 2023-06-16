import os
import pandas as pd
from datasets import load_dataset, Dataset

class processData:
    def __init__(self,path):
        self.path = path
    
    def read_data_pandas(self,file_name, encoding = 'utf-8'):
        dataset = pd.read_csv(os.path.join(self.path,file_name),encoding=encoding)
        return dataset
    
    def read_data_huggingface(self,train_name='NA', test_name='NA'):
        if test_name == 'NA':
            dataset = load_dataset("csv", data_files={"train" : os.path.join(self.path, train_name)})
        elif train_name== 'NA':
            dataset = load_dataset("csv", data_files={"test" : os.path.join(self.path, test_name)})
        else:
            dataset = load_dataset("csv", data_files={"train" :os.path.join(self.path, train_name),\
                                                          "test": os.path.join(self.path, test_name)})
        return dataset
    def save_dataset(self,dataset,file_name):
        dataset = pd.DataFrame(dataset)
        dataset.to_csv(os.path.join(self.path, file_name), index = False)
    def train_test_split(self,dataset, no_of_train=649, no_of_test = 50, seed = 42):
        dataset = dataset['train'].train_test_split(train_size = no_of_train, test_size = no_of_test)
        return dataset 
    def _process_row(self,row):
        evidence = ''
        for i in range(5):
            if row[f"evidences/{i}/evidence_label"] == row['claim_label']:
                evidence += row[f"evidences/{i}/evidence"]
        return evidence
    
    def process_climfever(self):
        dataset = self.read_data_pandas('climate-fever.csv')
        dataset = dataset.drop(dataset[dataset['claim_label'] == 'NOT_ENOUGH_INFO'].index)
        dataset = dataset.drop(dataset[dataset['claim_label'] == 'DISPUTED'].index)
        dataset['evidence'] = dataset.apply(self._process_row, axis = 1)
        dataset['label'] = dataset.apply(lambda x : 0 if x['claim_label'] == 'SUPPORTS' else 1 , axis=1)
        dataset = dataset[['label', 'claim', 'evidence']]
        dataset.to_csv(os.path.join(self.path, 'climate-fever-processed.csv'), index = False)
        print("Successfully created 'climate-fever' dataset!")
        
        