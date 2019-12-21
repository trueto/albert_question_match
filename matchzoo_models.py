import re
import os
import torch
import random
import logging
import numpy as np
import matchzoo as mz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.metrics import classification_report

from difflib import SequenceMatcher
from scipy import stats
# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

MATCHZOO_TASKS = {
    'ranking': mz.tasks.Ranking,
    'classification': mz.tasks.Classification
}

MATCHZOO_MODELS = {
    'dense_baseline': mz.models.DenseBaseline,
    'dssm': mz.models.DSSM,
    'cdssm': mz.models.CDSSM,
    'drmm': mz.models.DRMM,
    'drmmtks': mz.models.DRMMTKS,
    'esim': mz.models.ESIM,
    'knrm': mz.models.KNRM,
    'conv_knrm': mz.models.ConvKNRM,
    'bimpm': mz.models.BiMPM,
    'matchlstm': mz.models.MatchLSTM,
    'arci': mz.models.ArcI,
    'arcii': mz.models.ArcII,
    'mvlstm': mz.models.MVLSTM,
    'match_pyramid': mz.models.MatchPyramid,
    'anmm': mz.models.aNMM,
    'hbmp': mz.models.HBMP,
    'duet': mz.models.DUET,
    'diin': mz.models.DIIN,
    'match_srnn': mz.models.MatchSRNN
}

class MacthZooClassifer(BaseEstimator,ClassifierMixin):

    def __init__(self,
                 task='classification',
                 model_type='esim',
                 language='zh',
                 train_batch_size=32,
                 eval_bacth_size=32,
                 lr=1e-5,
                 epochs=5,
                 model_path=None):

        self.task = task.lower()
        self.model_type = model_type.lower()
        self.language = language
        self.train_batch_size = train_batch_size
        self.eval_bacth_size = eval_bacth_size
        self.lr = lr
        self.epochs = epochs
        self.model_path = 'model/' + model_type if model_path is None else model_path

        # if not Path(self.model_path).exists():
        #     Path(self.model_path).mkdir(parents=True)

        logger.info('matchzoo version %s' % mz.__version__)

    def fit(self,X,y):
        task = MATCHZOO_TASKS[self.task]()
        if self.task == 'classification':
            task.metrics = ['acc']
        else:
            task.metrics = [
                mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
                mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
                mz.metrics.MeanAveragePrecision()
            ]

        logger.info("{} task initialized with metrics: {}".format(
            self.task,task.metrics
        ))

        logger.info('data loading ...')
        X_train, X_dev, y_train, y_dev = train_test_split(X,y, test_size=0.1)
        train_pack_raw = self._data_pack(X_train,y_train,stage='train')
        dev_pack_raw = self._data_pack(X_dev,y_dev, stage='dev')
        logger.info('data loaded as `train_pack_raw` `dev_pack_raw`')

        self.preprocessor = MATCHZOO_MODELS[self.model_type].get_default_preprocessor()
        train_pack_processed = self.preprocessor.fit_transform(train_pack_raw)
        dev_pack_processed = self.preprocessor.transform(dev_pack_raw)
        logger.info("\n preprocessor.context:\n{}".format(self.preprocessor.context))

        fasttext_embedding = mz.datasets.embeddings.load_fasttext_embedding(language=self.language)
        term_index = self.preprocessor.context['vocab_unit'].state['term_index']
        embedding_matrix = fasttext_embedding.build_matrix(term_index)
        l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
        embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
        logger.info("embedding_matrix shape:{}".format(embedding_matrix.shape))

        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode='point',
            batch_size=self.train_batch_size
        )

        devset = mz.dataloader.Dataset(
            data_pack=dev_pack_processed,
            mode='point',
            batch_size=self.eval_bacth_size
        )
        logger.info("trainset len:{}".format(len(trainset)))
        logger.info("devset len:{}".format(len(devset)))

        padding_callback = MATCHZOO_MODELS[self.model_type].get_default_padding_callback()
        trainloader = mz.dataloader.DataLoader(
            dataset=trainset,
            stage='train',
            callback=padding_callback
        )
        devloader = mz.dataloader.DataLoader(
            dataset=devset,
            stage='dev',
            callback=padding_callback
        )

        model = MATCHZOO_MODELS[self.model_type]()
        model.params['task'] = task
        model.params['embedding'] = embedding_matrix
        if self.model_type == 'esim':
            model.params['mask_value'] = 0
            model.params['dropout'] = 0.2
            model.params['hidden_size'] = 200
            model.params['lstm_layer'] = 1
        elif self.model_type == 'anmmm':
            model.params['dropout_rate'] = 0.1
        elif self.model_type == 'conv_knrm':
            model.params['filters'] = 128
            model.params['conv_activation_func'] = 'tanh'
            model.params['max_ngram'] = 3
            model.params['use_crossmatch'] = True
            model.params['kernel_num'] = 11
            model.params['sigma'] = 0.1
            model.params['exact_sigma'] = 0.001
        elif self.model_type == 'arcii':
            model.params['left_length'] = 10
            model.params['right_length'] = 100
            model.params['kernel_1d_count'] = 32
            model.params['kernel_1d_size'] = 3
            model.params['kernel_2d_count'] = [64, 64]
            model.params['kernel_2d_size'] = [(3, 3), (3, 3)]
            model.params['pool_2d_size'] = [(3, 3), (3, 3)]
            model.params['dropout_rate'] = 0.3
        elif self.model_type == 'match_pyramid':
            model.params['kernel_count'] = [16, 32]
            model.params['kernel_size'] = [[3, 3], [3, 3]]
            model.params['dpool_size'] = [3, 10]
            model.params['dropout_rate'] = 0.1
        model.build()

        logger.info("\n model:\n{}".format(model))
        logger.info('Trainable params: %d' % sum(p.numel() for p in model.parameters() if p.requires_grad))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)

        trainer = mz.trainers.Trainer(
            model=model,
            optimizer=optimizer,
            trainloader=trainloader,
            validloader=devloader,
            validate_interval=None,
            epochs=self.epochs,
            save_dir=self.model_path,
            save_all=True,
            scheduler=scheduler if self.model_type=='conv_knrm' else None,
            clip_norm=10 if self.model_type=='conv_knrm' else None
        )
        trainer.run()
        # save_dict = {
        #     "model":model,
        #     "preprocessor": preprocessor,
        # }
        # torch.save(save_dict, os.path.join(self.model_path,'{}.pt'.format(self.model_type)))
        self.trainer = trainer
        return self

    def predict(self,X):

        test_pack_raw = self._data_pack(X, None, stage='test')

        # load_dict = torch.load(os.path.join(self.model_path,'{}.pt'.format(self.model_type)))
        # preprocessor = load_dict['preprocessor']

        test_pack_processed = self.preprocessor.transform(test_pack_raw)

        testset = mz.dataloader.Dataset(
            data_pack=test_pack_processed,
            mode='point',
            batch_size=self.eval_bacth_size,
            shuffle=True,
            sort=False
        )
        padding_callback = MATCHZOO_MODELS[self.model_type].get_default_padding_callback()
        testloader = mz.dataloader.DataLoader(
            dataset=testset,
            stage='test',
            callback=padding_callback
        )
        return self.trainer.predict(testloader)
        # model = load_dict['model']
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # state_dict = torch.load(os.path.join(self.model_path,'model.pt'),map_location=device)
        # model.load_state_dict(state_dict=state_dict)
        # model.to(device)
        #
        # with torch.no_grad():
        #     model.eval()
        #     predictions = []
        #     for batch in testloader:
        #         inputs = batch[0]
        #         outputs = model(inputs).detach().cpu()
        #         predictions.append(outputs)
        #
        #     return torch.cat(predictions, dim=0).numpy()

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        return classification_report(y,y_pred,digits=4)

    def _data_pack(self,X,y=None,stage='train'):
        data = pd.concat([X, y], axis=1, ignore_index=True, sort=False)
        columns = data.columns
        if stage in ['train', 'dev']:
            df = pd.DataFrame({
                'text_left': data[columns[0]],
                'text_right': data[columns[1]],
                'label': data[columns[2]].astype(int)
            })
        else:
            df = pd.DataFrame({
                'text_left': data[columns[0]],
                'text_right': data[columns[1]],
            })
        return mz.pack(df, self.task)


class Papaer_Approach:

    def __init__(self,model_name,model):
        self.model = model
        self.model_name = model_name

    def matchzoo_answer(self,X):
        y_pred = self.model.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
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

        if random.random() > 0.5:
            return 2
        else:
            return 3


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
        ## original
        y_pred_0 = self.matchzoo_answer(X)

        colums = X.columns
        ## exchanging the order
        exchanging_df = pd.concat([X[colums[1]], X[colums[0]]], axis=1)
        y_pred_1 = self.matchzoo_answer(exchanging_df)

        ##
        topic_out_df = self.delete_topic_words(X)
        y_pred_2 = self.matchzoo_answer(topic_out_df)
        ## similarity
        y_pred_0_similarity = self.text_similarity_answer(X,no_topic=False)
        y_pred_2_similarity = self.text_similarity_answer(topic_out_df, no_topic=True)
        y_pred_multiple = np.array([y_pred_0,y_pred_0_similarity,y_pred_1,y_pred_2,y_pred_2_similarity])
        y_pred = stats.mode(y_pred_multiple)[0][0]

        logger.info("\nvoting y_pred:\n{}".format(y_pred[:10]))
        return y_pred

    def score(self,X,y):
        y_pred = self.voting(X)
        result = classification_report(y,y_pred,digits=4)
        logger.info(result)
        output = '{}_results.txt'.format(self.model_name)
        with open(output,'w',encoding='utf8') as f:
            f.write(result)


if __name__ == '__main__':
    train_df = pd.read_csv('pairs_data/stage_3/train_cut.csv')
    X = pd.concat([train_df['question1'], train_df['question2']], axis=1)
    y = train_df['label']

    # dev_df = pd.read_csv('pairs_data/original/dev.csv')
    dev_df = pd.read_csv('pairs_data/stage_3/test_cut.csv')
    X_dev = pd.concat([dev_df['question1'], dev_df['question2']], axis=1)
    y_dev = dev_df['label']

    models = ['esim','anmm','conv_knrm','arcii','match_pyramid']
    for model in models:
        model_path = os.path.join('matchzoo_models',model)
        mzcls = MacthZooClassifer(model_type=model,
                                  epochs=10,model_path=model_path)
        mzcls.fit(X, y)
        paper_approach = Papaer_Approach(model_name=model,model=mzcls)
        paper_approach.score(X_dev,y_dev)

