# albert_question_match
This code is for paper "ALBERT-QM:An ALBERT Model Based Method for 
Chinese Health Related Question Matching"

## data source 
The fifth China Health Information Processing Conference (CHIP2019) 
shared task 2(https://www.biendata.com/competition/chip2019/).

## pre-trained model
- albert_zh https://github.com/brightmart/albert_zh
- bert-base-chinese https://github.com/google-research/bert
- chinese-wwm-ext https://github.com/ymcui/Chinese-BERT-wwm
- roabert_zh https://github.com/brightmart/roberta_zh
- ERNIE-Pytorch https://github.com/nghuyong/ERNIE-Pytorch

## software requirement
- pandas https://pandas.pydata.org/
- pytorch https://pytorch.org/
- transformers https://github.com/huggingface/transformers
- scikit-learn https://scikit-learn.org/stable/
- MatchZoo-py https://github.com/NTMC-Community/MatchZoo-py
- transformers-sklearn https://github.com/trueto/transformers_sklearn

## ALBERT-QM
ALBERT-QM is implemented as class `paper_approach` The main function of 
file `run_models.py` is shown as following:

```python
if __name__ == '__main__':
   # train_and_score_albert()
   # train_and_score_bert()
   # test_albert()
   # test_bert()
   # ablation_albert_text("text")
   # ablation_albert_text("albert")
   # ablation_data_augmentation()
```

`train_and_score_albert()` is used to fine-tune and evaluate three pre-trained ALBERT
models downloaded from `albert_zh`. 

`train_and_score_bert()` is used to fine-tune and evaluate four pre-trained BERT
models downloaded from community. 

`test_albert()` is used to evaluate the performance of ALBERT-QM on the test set.`test_bert()` 
similar as `test_albert()` but the base model of which is BERT series.

`ablation_albert_text("text")` is used to get the results dropping text similarity module. `ablation_albert_text("albert")` 
is used to get the results dropping ALBERT module. `ablation_data_augmentation()` is used to get the result of ALBERT-QM
without data augmentation.

## MatchZoo models
Five models was compared with ALBERT-QM. To transform the train set into the input format, [fastText Chinese embedding vocabulary](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md) and [Stanford segmenter toolkits](https://nlp.stanford.edu/software/segmenter.shtml) were used.The results of five models would be generated when running `pyhton matchzoo_models.py` in the terminal.

## Performance
### Best F1 of ALBERT-QM
The model file size is 64.8MB. The F1-score is 86.69%.
```
                precision    recall  f1-score   support

           0     0.8705    0.8688    0.8696      1021
           1     0.8634    0.8652    0.8643       979

    accuracy                         0.8670      2000
   macro avg     0.8669    0.8670    0.8669      2000
weighted avg     0.8670    0.8670    0.8670      2000
```
### Best F1 of BERT series
The model file size is 393MB. The F1-score is 88.78%.
```
                precision    recall  f1-score   support

           0     0.8844    0.8913    0.8878      1021
           1     0.8857    0.8784    0.8821       979

    accuracy                         0.8850      2000
   macro avg     0.8850    0.8849    0.8849      2000
weighted avg     0.8850    0.8850    0.8850      2000
```

### Best F1 of MatchZoo models
```
              precision    recall  f1-score   support

           0     0.6481    0.4887    0.5572      1021
           1     0.5756    0.7232    0.6410       979

    accuracy                         0.6035      2000
   macro avg     0.6118    0.6060    0.5991      2000
weighted avg     0.6126    0.6035    0.5982      2000
```
