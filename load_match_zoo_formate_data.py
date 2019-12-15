"""CHIP 2019 shared task 2 Question Pairs data loader"""
import jieba
import typing
import pandas as pd
import matchzoo
from matchzoo.engine.base_task import BaseTask
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
# from nltk.parse.corenlp import  CoreNLPTokenizer
def load_data(
    path: str,
    stage: str = 'train',
    task: typing.Union[str, BaseTask] = 'classification',
    return_classes: bool = False,
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load CHIP 2019 shared task 2 data.

    :param path: `None` for download from quora, specific path for
        downloaded data.
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param return_classes: Whether return classes for classification task.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    data_pack = _read_data(path, stage, task)

    if task == 'ranking' or isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif task == 'classification' or isinstance(
            task, matchzoo.tasks.Classification):
        if return_classes:
            return data_pack, [0, 1]
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task.")



def _read_data(path, stage, task):
    data = pd.read_csv(path, error_bad_lines=False, dtype=object)
    data = data.dropna(axis=0, how='any').reset_index(drop=True)
    # data['question1_list'] = data['question1'].apply(lambda t: ' '.join(list(t)))
    if stage in ['train', 'dev']:
        df = pd.DataFrame({
            # 'id_left': data['qid1'],
            # 'id_right': data['qid2'],
            'text_left': data['question1_cut'],
            'text_right': data['question2_cut'],
            # 'text_left': data['question1'],
            # 'text_right': data['question2'],
            'label': data['label'].astype(int)
        })
    else:
        df = pd.DataFrame({
            'text_left': data['question1_cut'],
            'text_right': data['question2_cut']
        })
    return matchzoo.pack(df, task)

## Stanford Word Segmenter for chinese word space
def stanford_seg(path):
    data = pd.read_csv(path, error_bad_lines=False, dtype=object)
    data = data.dropna(axis=0, how='any').reset_index(drop=True)
    data_dir = '/home/trueto/stanford_segmenter/'
    seg = StanfordSegmenter(path_to_jar=data_dir + 'stanford-segmenter.jar',
                            java_class='edu.stanford.nlp.ie.crf.CRFClassifier',
                            path_to_sihan_corpora_dict=data_dir + "data",
                            path_to_model=data_dir + 'data/pku.gz',
                            path_to_dict=data_dir + "data/dict-chris6.ser.gz")
    data['ques1_cut'] = data['question1'].apply(lambda t: seg.segment(t))
    data['ques2_cut'] = data['question2'].apply(lambda t: seg.segment(t))
    data.to_csv(path,index=False)

if __name__ == '__main__':
    stanford_seg('pairs_data/original/train.csv')
