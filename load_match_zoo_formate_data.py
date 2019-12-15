"""CHIP 2019 shared task 2 Question Pairs data loader"""
import jieba
import os
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
def stanford_seg(path,stage='train'):
    df_dir = os.path.join(path,'{}.csv'.format(stage))
    data = pd.read_csv(df_dir, error_bad_lines=False, dtype=object)
    data = data.dropna(axis=0, how='any').reset_index(drop=True)
    data_dir = '/home/trueto/stanford_segmenter/'
    seg = StanfordSegmenter(path_to_jar=data_dir + 'stanford-segmenter.jar',
                            java_class='edu.stanford.nlp.ie.crf.CRFClassifier',
                            path_to_sihan_corpora_dict=data_dir + "data",
                            path_to_model=data_dir + 'data/pku.gz',
                            path_to_dict=data_dir + "data/dict-chris6.ser.gz")
    columns = data.columns
    for column in columns:
        if column in ['question1', 'question2']:
            column_file = os.path.join(path,'cut','{}_{}.txt'.format(stage,column))
            data[column].to_csv(column_file, index=False)
            cut_file = os.path.join(path,'cut','{}_{}_cut.txt'.format(stage,column))
            with open(cut_file,'w') as f:
                f.write(seg.segment_file(column_file))

def merge_text(path,stage='train'):
    columns = ['question1','question2']
    df_list = []
    for column in columns:
        file_path = os.path.join(path,'cut','{}_{}_cut.txt'.format(stage,column))
        df = pd.read_csv(file_path,names=[column])
        df_list.append(df)

    temp_df = pd.read_csv(os.path.join(path,'{}.csv'.format(stage)))
    df_list.append(temp_df['label'])
    df_list.append(temp_df['category'])
    data_df = pd.concat(df_list,axis=1,ignore_index=True)
    data_df.columns = temp_df.columns
    data_df.to_csv(os.path.join(path,'{}_cut.csv'.format(stage)),index=False)


if __name__ == '__main__':
    ## 1. segmenter
    # stanford_seg('pairs_data/original',stage='train')
    # stanford_seg('pairs_data/original', stage='dev')
    # stanford_seg('pairs_data/original', stage='test')

    ## 2. merge txt to csv
    merge_text('pairs_data/original',stage='train')
    merge_text('pairs_data/original', stage='dev')
    merge_text('pairs_data/original', stage='test')
