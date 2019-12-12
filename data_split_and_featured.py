import re
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
        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)

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

        df_name = "{}_{}_".format(neg_rate,pos_rate) + name
        df_path = os.path.join(self.paper_dataset_path, df_name)
        if os.path.exists(df_path):
            self.sample_with_score = self.load_df(df_name)
            logger.info("load data from {}".format(df_path))
            return
        # print(data_df.head())
        data_df['score'] = data_df.apply(lambda df:
                                         self.similarity(df['question1'],df['question2']),axis=1)
        data_df['longest_match'] = data_df.apply(lambda df:
                                         self.longest_match(df['question1'],df['question2']),axis=1)
        self.save_df(data_df,'train_score_longest_text.csv')
        ## get the sample whose sequence similarity is more than neg_rate but label is 0
        negative_df = data_df[(data_df['score']>=neg_rate) & (data_df['label'] == 0)]
        ## get the sample whose sequence similarity is less than pos_rate but label is 1
        positive_df = data_df[(data_df['score']<=pos_rate) & (data_df['label'] == 1)]

        temp_df = pd.concat([negative_df,positive_df],ignore_index=True)
        self.sample_with_score = temp_df
        self.save_df(temp_df,df_name)
        logger.info("save data at {}".format(df_path))

    def sample_distribution(self,df:pd.DataFrame,name):
        label_name = name + '_label.csv'
        category_name = name + '_category.csv'
        label_path = os.path.join(self.paper_dataset_path,label_name)
        category_path = os.path.join(self.paper_dataset_path, category_name)
        if os.path.exists(label_path) or os.path.exists(category_path):
            return
        label_dis = df['label'].value_counts()
        category_dis = df['category'].value_counts()
        label_dis.to_csv(label_path)
        category_dis.to_csv(category_path)
        logger.info('file saved as {} and {}'.format(label_path,category_path))

    def sample_for_classification(self,
                                  df:pd.DataFrame,
                                  name):
        df_name = name
        df_path = os.path.join(self.paper_dataset_path, df_name)
        dis_path = os.path.join(self.paper_dataset_path,'dis_'+name)
        if os.path.exists(df_path):
            return
        train_df = self.get_classification(self.train_df)
        new_df = self.get_classification(df)
        cls_df = pd.concat([train_df,new_df],ignore_index=True)
        cls_df.drop_duplicates(inplace=True)
        label_df = cls_df['label'].value_counts()
        label_df.to_csv(dis_path)
        cls_df.to_csv(df_path,index=False)
        logger.info('data for classification saved as {}'.format(df_path))

    def delete_seed_words(self,df:pd.DataFrame,name):
        df_name = name
        df_path = os.path.join(self.paper_dataset_path, df_name)
        if os.path.exists(df_path):
            self.train_df_and_noseeds = self.load_df(name)
            return

        temp_df = df.copy()

        data = []
        for name, batch_df in temp_df.groupby(by='category',sort=False):
            patten = en2zh[name]
            for ques1, ques2,label in zip(batch_df['question1'].tolist(),
                                          batch_df['question2'].tolist(),
                                          batch_df['label'].tolist()):
                ques_1 = re.sub(patten,'',ques1)
                ques_2 = re.sub(patten, '', ques2)
                data.append([ques_1,ques_2,label,name])
        new_temp_df = pd.DataFrame(data,columns=df.columns)
        saved_df = pd.concat([df,new_temp_df],ignore_index=True)
        self.train_df_and_noseeds = saved_df
        saved_df.to_csv(df_path,index=False)
        logger.info('data for delete seeds saved as {}'.format(df_path))


    def get_classification(self,df: pd.DataFrame):
        ques1_list = df['question1'].tolist()
        ques2_list = df['question2'].tolist()
        ques_list = ques1_list + ques2_list
        category_list = df['category'].tolist()
        label_list = category_list + category_list
        temp_df = pd.DataFrame(columns=['text','label'])
        temp_df['text'] = ques_list
        temp_df['label'] = label_list
        return temp_df

    def longest_match(self,text_a,text_b):
        matcher = SequenceMatcher(a=text_a,b=text_b)
        i,_,size = matcher.find_longest_match(alo=0,ahi=len(text_a),
                                           blo=0,bhi=len(text_b))
        return text_a[i:i+size]

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
    # label 0
    # 0.3708017028254289 mean
    # 0.3333 median
    # label 1
    # 0.5324387016848364 mean
    # 0.5263 median
    data_process.similar_unsimilar(data_process.train_df,
                                   neg_rate=0.4,
                                   pos_rate=0.5,
                                   name='sample_with_score.csv')

    ## label distribution
    data_process.sample_distribution(data_process.train_df,name='train')
    data_process.sample_distribution(data_process.dev_df, name='dev')
    data_process.sample_distribution(data_process.test_df, name='test')

    ## sample for classfication
    dev_id_df = pd.read_csv('data/dev_id.csv')
    test_final_df = pd.read_csv('data/test_final.csv')
    unlabel_df = pd.concat([dev_id_df,test_final_df])
    data_process.sample_for_classification(unlabel_df,name='classification_data.csv')

    ## delete category words
    en2zh = {
        "diabetes": "糖尿病",
        "aids": "艾滋病|aids|艾滋|HIV|hiv",
        "breast_cancer": "乳腺癌|乳腺增生",
        "hypertension": "高血压",
        "hepatitis" : "乙肝"
    }
    data_process.delete_seed_words(data_process.train_df,'train_and_noseeds.csv')
    data_process.exchange_a_b(data_process.train_df_and_noseeds,'train_and_noseeds_b_a.csv')


