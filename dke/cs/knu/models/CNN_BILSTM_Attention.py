# Load Libraries
import warnings
import pdb
import numpy as np
import matplotlib.pyplot as plt

import model_evaluate
import model_preproecess
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.layers import Bidirectional
from keras.layers import Input, ELU, Embedding, BatchNormalization, Convolution1D, concatenate, Permute, multiply
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import Adam
#from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from keras.utils import plot_model

warnings.filterwarnings("ignore")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

#with tf.device("/cpu"):
TIME_STEPS = 1
INPUT_DIM = 77

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)

        return frozen_graph

def min_max_normalize(lst):
    normalized = []

    for value in lst:
        normalized_num = (value - min(lst)) / (max(lst) - min(lst))
        normalized.append(normalized_num)

    return normalized

def get_layer_outputs(model, layer_name, input_data, learning_phase=1):
    outputs = [layer.output for layer in model.layers if layer_name in layer.name]
    layers_fn = K.function([model.input, K.learning_phase()], outputs)
    return layers_fn([input_data, learning_phase])

def sum_1d(X):
    return K.sum(X, axis=1)

def get_conv_layer(emb, kernel_size=5, filters=256):
    # Conv layer
    conv = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb)
    conv = ELU()(conv)
    conv = Dropout(0.5)(conv)
    return conv

def cnn_bidirection_lstm_with_attention(max_len=77, emb_dim=32, max_vocab_len=128, lstm_output_size=32, W_reg=regularizers.l2(1e-4)):
    # Input
    main_input = Input(shape=(max_len,),  name='main_input')

    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, dropout=0.2, W_regularizer=W_reg)(main_input)

    conv2 = get_conv_layer(emb, kernel_size=2, filters=256)
    conv3 = get_conv_layer(emb, kernel_size=3, filters=256)
    conv4 = get_conv_layer(emb, kernel_size=4, filters=256)
    conv5 = get_conv_layer(emb, kernel_size=5, filters=256)

    merged = concatenate([conv2, conv3, conv4, conv5])

    # Bi-directional LSTM layer

    lstm = Bidirectional(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(emb)
    att = Permute((2, 1))(lstm)
    att = Dense(77, activation='softmax')(att)
    a_probs = Permute((2,1), name='attention_vec')(att)
    att = multiply([lstm, a_probs])


    cnnlstm_merged = concatenate([merged, att])
    cnnlstm_merged = Flatten()(cnnlstm_merged)

    hidden1 = Dense(640)(cnnlstm_merged)
    hidden1 = ELU()(hidden1)
    hidden1 = BatchNormalization(mode=0)(hidden1)
    hidden1 = Dropout(0.5)(hidden1)

    # Output layer (last fully connected layer)
    output = Dense(21, activation='softmax', name='real/output')(hidden1)

    # Compile model and define optimizer
    model = Model(input=[main_input], output=[output])
    model.summary()
    plot_model(model, to_file='model.png')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy',  metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(), preprocess.fmeasure, preprocess.recall, preprocess.precision])
    return model

#with tf.device("/cpu"):
epochs = 1
batch_size = 64

# Load data using model preprocessor
preprocess = model_preproecess.Preprocessor()

X_train, X_test, y_train, y_test = preprocess.load_data()
# Define LSTM with attention model
model_name = "CNN_BILSTM_ATT"
model = cnn_bidirection_lstm_with_attention()

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11)

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "some_directory", "save_model.pbtxt", as_text=True)
builder = tf.saved_model.builder.SavedModelBuilder("./savemodel")
builder.add_meta_graph_and_variables(K.get_session(), [tf.saved_model.tag_constants.SERVING])
builder.save()
