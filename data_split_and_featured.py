import os
import logging
import pandas as pd
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataProcess:

    def __init__(self,
                 data_path: str,
                 paper_dataset_path: str):
        self.data_df = pd.read_csv(data_path)
        self.paper_dataset_path = paper_dataset_path
        if not os.path.exists(paper_dataset_path):
            os.mkdir(paper_dataset_path)


    def split_data(self):
        train_dev_df, test_df = train_test_split(self.data_df,test_size=0.1,
                                             random_state=520,shuffle=True)

        train_df, dev_df = train_test_split(train_dev_df,test_size=2000,
                                            random_state=520,shuffle=True)
        # save the split data
        if len(os.listdir(self.paper_dataset_path)) > 0:
            self.train_df = self.load_df('train.csv')
            self.dev_df = self.load_df('dev.csv')
            self.test_df = self.load_df('test.csv')
            logger.info("load train/dev/test data from {}".format(self.paper_dataset_path))
        else:
            self.train_df,self.dev_df,self.test_df = train_df,dev_df,test_df
            self.save_df(train_df,'train.csv')
            self.save_df(dev_df,'dev.csv')
            self.save_df(test_df,'test.csv')
            logger.info("dataset was splitted as 8:1:1")

    def exchange_a_b(self,
                     df:pd.DataFrame,
                     name:str):
        df_path = os.path.join(self.paper_dataset_path,name)
        if os.path.exists(df_path):
            self.train_df_b_a = self.load_df(name)
            logger.info("load data from {}".format(df_path))
            return
        temp_df = df.copy(deep=True)
        temp_df['question1'] = df['question2']
        temp_df['question2'] = df['question1']
        new_df = pd.concat([df,temp_df],ignore_index=True)
        self.train_df_b_a = new_df
        self.save_df(new_df,name)
        logger.info("save data as {}".format(df_path))

    def similar_unsimilar(self,
                          data_df: pd.DataFrame,
                          neg_rate:float,
                          pos_rate:float,
                          name:str):

        df_name = "{}_{}".format(neg_rate,pos_rate) + name
        df_path = os.path.join(self.paper_dataset_path, df_name)
        if os.path.exists(df_path):
            self.sample_with_score = self.load_df(df_name)
            logger.info("load data from {}".format(df_path))
            return
        # print(data_df.head())
        data_df['score'] = data_df.apply(lambda df:
                                         self.similarity(df['question1'],df['question2']),axis=1)
        print(data_df[(data_df['label']==0)]['score'].mean())
        ## get the sample whose sequence similarity is more than neg_rate but label is 0
        negative_df = data_df[(data_df['score']>=neg_rate) & (data_df['label'] == 0)]
        ## get the sample whose sequence similarity is less than pos_rate but label is 1
        positive_df = data_df[(data_df['score']<=pos_rate) & (data_df['label'] == 1)]

        temp_df = pd.concat([negative_df,positive_df],ignore_index=True)
        self.sample_with_score = temp_df
        self.save_df(temp_df,df_name)
        logger.info("save data at {}".format(df_path))

    def similarity(self,text_a,text_b):
        score = SequenceMatcher(a=text_a,b=text_b).ratio()
        return round(score,4)

    def load_df(self,
                name:str):
        df_path = os.path.join(self.paper_dataset_path,name)
        return pd.read_csv(df_path)

    def save_df(self,
                df:pd.DataFrame,
                name:str):
        df_path = os.path.join(self.paper_dataset_path,name)
        df.to_csv(df_path,index=False)

if __name__ == '__main__':
    data_process = DataProcess(data_path='data/train.csv',
                               paper_dataset_path='paper_dataset')

    ## split the original dataset: train : dev : test = 8 : 1 : 1
    data_process.split_data()

    ## trick 1: exchange the order of text pairs
    data_process.exchange_a_b(data_process.train_df,'train_b_a.csv')

    ## get the special sample
    data_process.similar_unsimilar(data_process.train_df,
                                   neg_rate=0.8,
                                   pos_rate=0.2,
                                   name='sample_with_score.csv')
