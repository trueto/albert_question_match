import os
import torch
import joblib
import logging
import numpy as np
import matchzoo as mz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.metrics import classification_report

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

        preprocessor = MATCHZOO_MODELS[self.model_type].get_default_preprocessor()
        train_pack_processed = preprocessor.fit_transform(train_pack_raw)
        dev_pack_processed = preprocessor.transform(dev_pack_raw)
        logger.info("\n preprocessor.context:\n{}".format(preprocessor.context))

        fasttext_embedding = mz.datasets.embeddings.load_fasttext_embedding(language=self.language)
        term_index = preprocessor.context['vocab_unit'].state['term_index']
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
        if self.model_type == 'esim':
            model.params['task'] = task
            model.params['embedding'] = embedding_matrix
            model.params['mask_value'] = 0
            model.params['dropout'] = 0.2
            model.params['hidden_size'] = 200
            model.params['lstm_layer'] = 1
        model.build()

        logger.info("\n model:\n{}".format(model))
        logger.info('Trainable params: %d' % sum(p.numel() for p in model.parameters() if p.requires_grad))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)


        trainer = mz.trainers.Trainer(
            model=model,
            optimizer=optimizer,
            trainloader=trainloader,
            validloader=devloader,
            validate_interval=None,
            epochs=self.epochs,
            save_dir=self.model_path,
            save_all=True
        )
        trainer.run()
        save_dict = {
            "model":model,
            "preprocessor": preprocessor,
        }
        joblib.dump(save_dict, os.path.join(self.model_path,'{}.joblib'.format(self.model_type)))
        return self

    def predict(self,X):

        test_pack_raw = self._data_pack(X, None, stage='test')

        load_dict = joblib.load(os.path.join(self.model_path,'{}.joblib'.format(self.model_type)))
        preprocessor = load_dict['preprocessor']

        test_pack_processed = preprocessor.transform(test_pack_raw)

        testset = mz.dataloader.Dataset(
            data_pack=test_pack_processed,
            mode='point',
            batch_size=self.eval_bacth_size
        )
        padding_callback = MATCHZOO_MODELS[self.model_type].get_default_padding_callback()
        testloader = mz.dataloader.DataLoader(
            dataset=testset,
            stage='dev',
            callback=padding_callback
        )
        model = load_dict['model']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(os.path.join(self.model_path,'model.pt'),map_location=device)
        model.load_state_dict(state_dict=state_dict)
        model.to(device)

        with torch.no_grad():
            model.eval()
            predictions = []
            for batch in testloader:
                inputs = batch[0]
                outputs = model(inputs).detach().cpu()
                predictions.append(outputs)

            return torch.cat(predictions, dim=0).numpy()

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
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


if __name__ == '__main__':
    mzcls = MacthZooClassifer()
