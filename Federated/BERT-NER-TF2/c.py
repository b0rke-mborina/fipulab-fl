import os
import logging
os.environ['TF_CUDNN_DETERMINISTIC']='1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# print("Using only CPU")
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# logging.getLogger('tensorflow').setLevel(logging.FATAL)

from dataset.dataset_loader import NerProcessor, FewNERDProcessor, create_tf_dataset_for_client, split_to_tf_datasets, batch_features
from utils.fl_utils import *

import tensorflow as tf
from tqdm.notebook import tqdm

import numpy as np

from models.model import build_BertNer, MaskedSparseCategoricalCrossentropy
from tokenization import FullTokenizer



# Pretrained models
TINY = 'uncased_L-2_H-128_A-2'
MINI = 'uncased_L-4_H-256_A-4'
SMALL = 'uncased_L-4_H-512_A-8'
MEDIUM = 'uncased_L-8_H-512_A-8'
BASE = 'uncased_L-12_H-768_A-12'

MODEL = os.path.join("models", TINY)
SEQ_LEN = 128
BATCH_SIZE = 32
PRETRAINED = False

# processor = NerProcessor('dataset/conll')
processor = FewNERDProcessor('dataset/few_nerd')
tokenizer = FullTokenizer(os.path.join(MODEL, "vocab.txt"), True)
train_features = processor.get_train_as_features(SEQ_LEN, tokenizer)
eval_features = processor.get_test_as_features(SEQ_LEN, tokenizer)


def eval_model(model, eval_data, do_print=True):
    return evaluate(model, eval_data, 
                    processor.get_label_map(), 
                    processor.token_ind('O'), 
                    processor.token_ind('[SEP]'),
                    processor.token_ind('[PAD]'), 
                    do_print=do_print)
    
eval_data_batched = batch_features(eval_features, processor.get_labels(), SEQ_LEN, tokenizer, batch_size=64)

@tf.function
def train_batch(model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = model.loss(y, logits)
    grad = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grad, model.trainable_variables))

def train_single(epochs=1, lr=5e-5, batch_size=32, pretrained=True):
    model = build_BertNer(MODEL, processor.label_len(), SEQ_LEN)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=MaskedSparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE))
    
    if pretrained:
        restore_model_ckpt(model, MODEL)
    
    data = split_to_tf_datasets(train_features, 1, batch_size)[0]
    for e in range(epochs):
        for x, y in tqdm(data, position=0, leave=False, desc="Training"):
            train_batch(model, x, y)
        eval_model(model, eval_data_batched)
    return model

train_single(50)