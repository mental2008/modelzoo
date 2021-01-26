import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import os
from tensorflow import keras
import time


models = [
    {'name': 'albert_en_base', 'version': '2',
     'preprocessor': 'http://tfhub.dev/tensorflow/albert_en_preprocess/2',
     'encoder': 'https://tfhub.dev/tensorflow/albert_en_base/2'},
    {'name': 'albert_en_large', 'version': '2',
     'preprocessor': 'http://tfhub.dev/tensorflow/albert_en_preprocess/2',
     'encoder': 'https://tfhub.dev/tensorflow/albert_en_large/2'},
    {'name': 'albert_en_xlarge', 'version': '2',
     'preprocessor': 'http://tfhub.dev/tensorflow/albert_en_preprocess/2',
     'encoder': 'https://tfhub.dev/tensorflow/albert_en_xlarge/2'},
    {'name': 'albert_en_xxlarge', 'version': '2',
     'preprocessor': 'http://tfhub.dev/tensorflow/albert_en_preprocess/2',
     'encoder': 'https://tfhub.dev/tensorflow/albert_en_xxlarge/2'},

    {'name': 'bert_en_uncased_L-2_H-128_A-2', 'version': '1',
     'preprocessor': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
     'encoder': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1'},
    {'name': 'bert_en_uncased_L-4_H-256_A-4', 'version': '1',
     'preprocessor': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
     'encoder': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1'},
    {'name': 'bert_en_uncased_L-4_H-512_A-8', 'version': '1',
     'preprocessor': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
     'encoder': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'},
    {'name': 'bert_en_uncased_L-8_H-512_A-8', 'version': '1',
     'preprocessor': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
     'encoder': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1'},

    {'name': 'bert_en_uncased_L-12_H-768_A-12', 'version': '3',
     'preprocessor': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
     'encoder': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'},
    {'name': 'bert_en_uncased_L-24_H-1024_A-16', 'version': '2',
     'preprocessor': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
     'encoder': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/3'},

    {'name': 'electra_small', 'version': '2',
     'preprocessor': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
     'encoder': 'https://tfhub.dev/google/electra_small/2'},
    {'name': 'electra_base', 'version': '2',
     'preprocessor': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
     'encoder': 'https://tfhub.dev/google/electra_base/2'},
    {'name': 'electra_large', 'version': '2',
     'preprocessor': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
     'encoder': 'https://tfhub.dev/google/electra_large/2'},

    {'name': 'nnlm-en-dim50', 'version': '2',
     'encoder': 'https://tfhub.dev/google/nnlm-en-dim50/2'},
    {'name': 'nnlm-en-dim128', 'version': '2',
     'encoder': 'https://tfhub.dev/google/nnlm-en-dim128/2'},

    {'name': 'Wiki-words-250', 'version': '2',
     'encoder': 'https://tfhub.dev/google/Wiki-words-250/2'},
    {'name': 'Wiki-words-500', 'version': '2',
     'encoder': 'https://tfhub.dev/google/Wiki-words-500/2'},

    {'name': 'universal-sentence-encoder', 'version': '4',
     'encoder': 'https://tfhub.dev/google/universal-sentence-encoder/4'},
    {'name': 'universal-sentence-encoder-large', 'version': '5',
     'encoder': 'https://tfhub.dev/google/universal-sentence-encoder-large/5'},

    {'name': 'universal-sentence-encoder-multilingual', 'version': '3',
     'encoder': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'},
    {'name': 'universal-sentence-encoder-multilingual-large', 'version': '3',
     'encoder': 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'},

    {'name': 'elmo', 'version': '3',
     'encoder': 'https://tfhub.dev/google/elmo/3'},
]


if __name__ == '__main__':
    for model in models:
        save_path = os.path.join(os.getcwd(), 'tensorflow_model', 'nlp', model['name'], model['version'])
        if os.path.exists(save_path):
            continue
        print(save_path)

        # text_input = ["This is a test!"]
        if 'preprocessor' in model:
            text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
            model['preprocessor'] = '/Users/mental/Downloads/bert_en_uncased_preprocess_3'
            model['encoder'] = '/Users/mental/Downloads/' + \
                               model['encoder'].replace('https://tfhub.dev/tensorflow/', '').replace('/', '_')
            print(model['preprocessor'], model['encoder'])
            preprocessor = hub.KerasLayer(model['preprocessor'])(text_input)
            outputs = hub.KerasLayer(model['encoder'])(preprocessor)
            net = keras.models.Model(inputs=text_input, outputs=outputs)
        else:
            net = hub.KerasLayer(model['encoder'], input_shape=[], dtype=tf.string)

        @tf.function()
        def my_predict(inputs):
            if 'preprocessor' in model:
                prediction = net(inputs)["pooled_output"]
            else:
                prediction = net(inputs)
            return {"predictions": prediction}

        my_signatures = my_predict.get_concrete_function(
            inputs=tf.TensorSpec(shape=[None, ], dtype=tf.string, name="inputs")
        )

        tf.saved_model.save(net, save_path, signatures=my_signatures)
