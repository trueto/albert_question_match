import os
import logging
import pandas as pd
from transformers_sklearn import BERTologyClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(model_path,output_dir):
    train_df = pd.read_csv('pairs_data/stage_3/train.csv')
    X_train = pd.concat([train_df['question1'],train_df['question2']],axis=1)
    y_train = train_df['label']

    cls = BERTologyClassifier(model_type='albert',model_name_or_path=model_path,
                              gradient_accumulation_steps=8,logging_steps=1000,
                              save_steps=1000,output_dir=output_dir,
                              per_gpu_train_batch_size=2,per_gpu_eval_batch_size=2
                              )
    cls.fit(X_train,y_train)

    train_df = pd.read_csv('pairs_data/stage_3/dev.csv')
    X_dev = pd.concat([train_df['question1'], train_df['question2']], axis=1)
    y_dev = train_df['label']
    logger.info("dev results:\n{}".format(cls.score(X_dev,y_dev)))

if __name__ == '__main__':
    path_list = ['albert_tiny','albert_base','albert_large','albert_xlarge']
    for name in path_list:
        model_path = os.path.join('albert_pretrained_models',name)
        output_path = os.path.join('albert_results',name)
        run(model_path,output_dir=output_path)