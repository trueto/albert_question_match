import torch
import logging
import numpy as np
import matchzoo as mz
from load_match_zoo_formate_data import load_data

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

logger.info('matchzoo version %s' % mz.__version__)

classification_task = mz.tasks.Classification(num_classes=2)
classification_task.metrics = ['acc']
logger.info("`classification_task` initialized with metrics: %s" % classification_task.metrics)

logger.info('data loading ...')
train_pack_raw = load_data(path='pairs_data/original/train.csv',
                           stage='train',task=classification_task)
dev_pack_raw = load_data(path='pairs_data/original/dev.csv',
                         stage='dev',task=classification_task)

test_pack_raw = load_data(path='pairs_data/original/test.csv',
                          stage='test',task=classification_task)
logger.info('data loaded as `train_pack_raw` `dev_pack_raw` `test_pack_raw`')

preprocessor = mz.models.ESIM.get_default_preprocessor()

train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)

logger.info("\n preprocessor.context:\n{}".format(preprocessor.context))

fasttext_embedding = mz.datasets.embeddings.load_fasttext_embedding(language='zh')
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = fasttext_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

logger.info("embedding_matrix shape:{}".format(embedding_matrix.shape))

trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='point',
    batch_size=32
)

devset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed,
    mode='point',
    batch_size=32
)

logger.info("trainset len:{}".format(len(trainset)))
logger.info("devset len:{}".format(len(devset)))

padding_callback = mz.models.ESIM.get_default_padding_callback()

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

model = mz.models.ESIM()

model.params['task'] = classification_task
model.params['embedding'] = embedding_matrix
model.params['mask_value'] = 0
model.params['dropout'] = 0.2
model.params['hidden_size'] = 200
model.params['lstm_layer'] = 1

model.build()

logger.info("\n model:\n{}".format(model))
logger.info('Trainable params: %d' % sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=devloader,
    validate_interval=None,
    epochs=5,
    save_dir='model/esim'
)

trainer.run()
