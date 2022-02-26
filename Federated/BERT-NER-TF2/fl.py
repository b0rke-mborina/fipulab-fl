DEVICES = "GPU:0" # Moguce vrijednosti: "CPU", "GPU:0", "GPU:1", "GPUS"
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if DEVICES == 'CPU':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
elif DEVICES == 'GPU:0':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
elif DEVICES == 'GPU:1':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from models.model import build_BertNer, MaskedSparseCategoricalCrossentropy
from tokenization import FullTokenizer

from dataset.dataset_loader import NerProcessor, FewNERDProcessor, create_tf_dataset_for_client, split_to_tf_datasets, batch_features
from utils.fl_utils import *

import nest_asyncio
nest_asyncio.apply()

import tensorflow_federated as tff
from tqdm.notebook import tqdm
import json
import numpy as np



if DEVICES == 'CPU':
    cl_tf_devices = tf.config.list_logical_devices('CPU')
elif DEVICES == 'GPUS':
    cl_tf_devices = tf.config.list_logical_devices('GPU')
else:
    cl_tf_devices = tf.config.list_logical_devices('GPU')[:1]
tff.backends.native.set_local_execution_context(
    server_tf_device=tf.config.list_logical_devices('CPU')[0],
    client_tf_devices=cl_tf_devices)


# Pretrained models
TINY = 'uncased_L-2_H-128_A-2'
MINI = 'uncased_L-4_H-256_A-4'
SMALL = 'uncased_L-4_H-512_A-8'
MEDIUM = 'uncased_L-8_H-512_A-8'
BASE = 'uncased_L-12_H-768_A-12'

MODEL = os.path.join("models", TINY)
SEQ_LEN = 128
BATCH_SIZE = 32
PRETRAINED = True

# processor = NerProcessor('dataset/conll')
processor = FewNERDProcessor('dataset/few_nerd')
tokenizer = FullTokenizer(os.path.join(MODEL, "vocab.txt"), True)
train_features = processor.get_train_as_features(SEQ_LEN, tokenizer)
eval_features = processor.get_test_as_features(SEQ_LEN, tokenizer)


# Wrap a Keras model for use with TFF.
def model_fn(m_name, num_labels, seq_len, input_spec):
    model = build_BertNer(m_name, num_labels, seq_len)
    return tff.learning.from_keras_model(
        model,
        input_spec=input_spec,
        loss=MaskedSparseCategoricalCrossentropy()) # reduction=tf.keras.losses.Reduction.NONE))

def eval_model(model, eval_data, do_print=True):
    return evaluate(model, eval_data, 
                    processor.get_label_map(), 
                    processor.token_ind('O'), 
                    processor.token_ind('[SEP]'),
                    processor.token_ind('[PAD]'), 
                    do_print=do_print)
    
# eval_features = eval_features[:5_000] # Samo prvih par tisuca za testiranje, maknuti za konačne ekperimente
eval_data_batched = batch_features(eval_features, processor.get_labels(), SEQ_LEN, tokenizer, batch_size=64)



num_clients, num_train_clients = 100, 10
assert num_clients >= num_train_clients

dataset_list = split_to_tf_datasets(train_features, num_clients, batch_size=BATCH_SIZE)
trainer = tff.learning.build_federated_averaging_process(
    model_fn=lambda: model_fn(MODEL, processor.label_len(), SEQ_LEN, dataset_list[0].element_spec),
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(5e-3),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(5e-3),
    use_experimental_simulation_loop=True
)

state = trainer.initialize()
if PRETRAINED:
    state = state_from_checkpoint(state, build_BertNer(MODEL, processor.label_len(), SEQ_LEN), MODEL)

res_list=[]
examples = 0
for rnd_ind in range(1, 501):
    train_data = list(np.random.choice(dataset_list, num_train_clients, replace=False))
    state, metrics = trainer.next(state, train_data)
    print("Round", rnd_ind, "Loss:", metrics['train']['loss'], "Examples:", metrics['stat']['num_examples'])
    examples = metrics['stat']['num_examples']
    
    # Ne treba svaku rundu gledati tocnost, moze svakih 10 (jedna epoha kumulativno)
    if rnd_ind % num_train_clients == 0:
        state_model = state_to_model(state, build_BertNer(MODEL, processor.label_len(), SEQ_LEN))
        res = eval_model(state_model, eval_data_batched, do_print=True)
        res['Round'] = rnd_ind
        res['Examples'] = examples
        res_list.append(res)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

with open("log/results-tiny-pretrained-few_ner.json", "w") as outfile:
    json.dump({'results': res_list, 'model': MODEL, 'seq_len': SEQ_LEN, 
               'pretrained': PRETRAINED, 'batch_size': BATCH_SIZE}, outfile, indent=None, cls=NpEncoder)