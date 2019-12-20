import re
import os
import logging
import random
import pandas as pd
import numpy as np
from scipy import stats
from difflib import SequenceMatcher
from transformers_sklearn import BERTologyClassifier
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(model_type,model_path,output_dir):
    train_df = pd.read_csv('pairs_data/stage_3/train.csv')
    X_train = pd.concat([train_df['question1'],train_df['question2']],axis=1)
    y_train = train_df['label']

    cls = BERTologyClassifier(model_type=model_type,model_name_or_path=model_path,
                              gradient_accumulation_steps=8,logging_steps=1000,
                              save_steps=1000,output_dir=output_dir,
                              per_gpu_train_batch_size=2,per_gpu_eval_batch_size=2
                              )
    cls.fit(X_train,y_train)

    # train_df = pd.read_csv('pairs_data/stage_3/dev.csv')
    # X_dev = pd.concat([train_df['question1'], train_df['question2']], axis=1)
    # y_dev = train_df['label']
    # logger.info("dev results:\n{}".format(cls.score(X_dev,y_dev)))

def run_albert():
    path_list = ['albert_tiny', 'albert_base', 'albert_large', 'albert_xlarge']
    for name in path_list:
        model_path = os.path.join('albert_pretrained_models', name)
        output_path = os.path.join('albert_results', name)
        train(model_type='albert',model_path=model_path, output_dir=output_path)

def run_bertology():
    name_list = ['chinese_wwm_ext','ERNIE','RoBERTa','bert-base-chinese']
    for name in name_list:
        model_path = os.path.join('bert_pretrained_models/', name)
        output_path = os.path.join('bert_results', name)
        if name == 'bert-base-chinese':
            train(model_type='bert', model_path=name, output_dir=output_path)
        else:
            train(model_type='bert', model_path=model_path, output_dir=output_path)

class Papaer_Approach:

    def __init__(self,finetune_model_path,model_type,mode='dev',strategy=None):
        self.mode = mode
        self.strategy = strategy
        self.finetune_model_path = finetune_model_path
        self.model = BERTologyClassifier(output_dir=finetune_model_path,model_type=model_type)
    def albert_answer(self,X):
        y_pred = self.model.predict(X)
        logger.info("\nalbert y_pred:\n{}".format(y_pred[:10]))
        return y_pred
    def text_similarity_answer(self,X,no_topic=False):
        X['label'] = X.apply(lambda df: self.row_answer(df,no_topic),axis=1)
        y_pred = X['label'].tolist()
        logger.info("\ntext similarity y_pred:\n{}".format(y_pred[:10]))
        return y_pred

    def row_answer(self,row,no_topic):
        score = SequenceMatcher(a=row['question1'],
                                b=row['question2']).ratio()
        score = round(score,4)
        if no_topic:
            if score < 0.2:
                return 0
            if score > 0.4242:
                return 1
        else:
            if score < 0.333:
                return 0
            if score > 0.5263:
                return 1
        ## if above not return, then randomly return the label
        if random.random() > 0.5:
            return 1
        else:
            return 0

    def delete_topic_words(self,X):
        colums = X.columns
        temp_df = X.copy()
        temp_df[colums[0]] = temp_df[colums[0]].apply(self.delete_row)
        temp_df[colums[1]] = temp_df[colums[1]].apply(self.delete_row)
        return temp_df

    def delete_row(self,text):
        topic_word_patten = "糖尿病|艾滋病|aids|艾滋|HIV|hiv|乳腺癌|乳腺增生|高血压|乙肝|乙肝表面抗体"
        return re.sub(topic_word_patten,'',text)

    def voting(self,X):
        y_pred = []
        if not self.strategy:
            ## original
            y_pred_0 = self.albert_answer(X)

            colums = X.columns
            ## exchanging the order
            exchanging_df = pd.concat([X[colums[1]], X[colums[0]]], axis=1)
            y_pred_1 = self.albert_answer(exchanging_df)

            ##
            topic_out_df = self.delete_topic_words(X)
            y_pred_2 = self.albert_answer(topic_out_df)
            ## similarity
            y_pred_0_similarity = self.text_similarity_answer(X,no_topic=False)
            y_pred_2_similarity = self.text_similarity_answer(topic_out_df, no_topic=True)
            y_pred_multiple = np.array([y_pred_0,y_pred_0_similarity,y_pred_1,y_pred_2,y_pred_2_similarity])
            y_pred = stats.mode(y_pred_multiple)[0][0]

        if self.strategy == 'albert':
            ## original
            y_pred_0 = self.albert_answer(X)

            colums = X.columns
            ## exchanging the order
            exchanging_df = pd.concat([X[colums[1]], X[colums[0]]], axis=1)
            y_pred_1 = self.albert_answer(exchanging_df)

            ##
            topic_out_df = self.delete_topic_words(X)
            y_pred_2 = self.albert_answer(topic_out_df)
            y_pred_multiple = np.array([y_pred_0, y_pred_1, y_pred_2])
            y_pred = stats.mode(y_pred_multiple)[0][0]

        if self.strategy == 'text':
            ## similarity
            topic_out_df = self.delete_topic_words(X)
            y_pred_0_similarity = self.text_similarity_answer(X, no_topic=False)
            y_pred_2_similarity = self.text_similarity_answer(topic_out_df, no_topic=True)
            y_pred_multiple = np.array([y_pred_0_similarity,y_pred_2_similarity])
            y_pred = stats.mode(y_pred_multiple)[0][0]

        logger.info("\nvoting y_pred:\n{}".format(y_pred[:10]))
        return y_pred

    def score(self,X,y):
        y_pred = self.voting(X)
        result = classification_report(y,y_pred,digits=4)
        logger.info(result)
        name = self.finetune_model_path.split('/')[1]
        if not self.strategy:
            output = '{}_{}_results.txt'.format(self.mode, name)
        else:
            output = '{}_{}_results_{}.txt'.format(self.mode, name, self.strategy)

        with open(output,'w',encoding='utf8') as f:
            f.write(result)

def train_and_score_albert():
    # 1. fine tune the albert model
    run_albert()
    # 2. evalute the paper approach
    dev_df = pd.read_csv('pairs_data/stage_3/dev.csv')
    X_dev = pd.concat([dev_df['question1'],dev_df['question2']],axis=1)
    y_dev = dev_df['label'].to_numpy()
    # ## albert
    name_list = ['albert_tiny', 'albert_base', 'albert_large']
    for name in name_list:
        paper_approach = Papaer_Approach(finetune_model_path='albert_results/{}'.format(name),
                                 model_type='albert',mode='dev')
        paper_approach.score(X_dev,y_dev)

def train_and_score_bert():
    # 1. fine tune the albert model
    run_bertology()
    # 2. evalute the paper approach
    dev_df = pd.read_csv('pairs_data/stage_3/dev.csv')
    X_dev = pd.concat([dev_df['question1'],dev_df['question2']],axis=1)
    y_dev = dev_df['label'].to_numpy()
    # ## bert
    name_list = ['chinese_wwm_ext','ERNIE','RoBERTa','bert-base-chinese']
    for name in name_list:
        paper_approach = Papaer_Approach(finetune_model_path='bert_results/{}'.format(name),
                                 model_type='bert',mode='dev')
        paper_approach.score(X_dev,y_dev)


def test_albert():
    dev_df = pd.read_csv('pairs_data/stage_3/test.csv')
    X_dev = pd.concat([dev_df['question1'], dev_df['question2']], axis=1)
    y_dev = dev_df['label'].to_numpy()
    # ## albert
    name_list = ['albert_tiny', 'albert_base', 'albert_large']
    for name in name_list:
        paper_approach = Papaer_Approach(finetune_model_path='albert_results/{}'.format(name),
                                         model_type='albert',mode='test')
        paper_approach.score(X_dev, y_dev)


def test_bert():
    dev_df = pd.read_csv('pairs_data/stage_3/test.csv')
    X_dev = pd.concat([dev_df['question1'], dev_df['question2']], axis=1)
    y_dev = dev_df['label'].to_numpy()
    # ## bert
    name_list = ['chinese_wwm_ext', 'ERNIE', 'RoBERTa', 'bert-base-chinese']
    for name in name_list:
        paper_approach = Papaer_Approach(finetune_model_path='bert_results/{}'.format(name),
                                         model_type='bert', mode='test')
        paper_approach.score(X_dev, y_dev)


def ablation_albert_text(strategy):
    dev_df = pd.read_csv('pairs_data/stage_3/test.csv')
    X_dev = pd.concat([dev_df['question1'], dev_df['question2']], axis=1)
    y_dev = dev_df['label'].to_numpy()
    # ## albert
    paper_approach = Papaer_Approach(finetune_model_path='albert_results/albert_large',
                                     model_type='albert', mode='test', strategy=strategy)
    paper_approach.score(X_dev, y_dev)


def ablation_data_augmentation():
    train_df = pd.read_csv('pairs_data/ablation/train.csv')
    X_train = pd.concat([train_df['question1'], train_df['question2']], axis=1)
    y_train = train_df['label']

    cls = BERTologyClassifier(model_type='albert', model_name_or_path='albert_pretrained_models/albert_large',
                              gradient_accumulation_steps=8, logging_steps=1000,
                              save_steps=1000, output_dir='albert_results/albert_large_ablation',
                              per_gpu_train_batch_size=2, per_gpu_eval_batch_size=2
                              )
    cls.fit(X_train, y_train)

    dev_df = pd.read_csv('pairs_data/ablation/test.csv')
    X_dev = pd.concat([dev_df['question1'], dev_df['question2']], axis=1)
    y_dev = dev_df['label'].to_numpy()
    # ## albert
    paper_approach = Papaer_Approach(finetune_model_path='albert_results/albert_large_ablation',
                                     model_type='albert', mode='test', strategy=None)
    paper_approach.score(X_dev, y_dev)

if __name__ == '__main__':
   # train_and_score_albert()
   # train_and_score_bert()
   # test_albert()
   # test_bert()
   # ablation_albert("text")
   # ablation_albert("albert")
   ablation_data_augmentation()



