# Load Libraries
import warnings
import pdb
import numpy as np
import matplotlib.pyplot as plt

import model_evaluate
import Attention_model_preprocess
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

with tf.device("/GPU:0"):
    TIME_STEPS = 1
    INPUT_DIM = 77

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
        #conv = MaxPooling1D(5)(conv)
        #conv = Lambda(sum_1d, output_shape=(filters,))(conv)
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

        #att = SeqSelfAttention(name='attention_vec')(lstm)

        cnnlstm_merged = concatenate([merged, att])
        cnnlstm_merged = Flatten()(cnnlstm_merged)

        hidden1 = Dense(640)(cnnlstm_merged)
        hidden1 = ELU()(hidden1)
        hidden1 = BatchNormalization(mode=0)(hidden1)
        hidden1 = Dropout(0.5)(hidden1)

        # Output layer (last fully connected layer)
        output = Dense(21, activation='softmax', name='output')(hidden1)

        # Compile model and define optimizer
        model = Model(input=[main_input], output=[output])
        model.summary()
        plot_model(model, to_file='model.png')
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy',  metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(), preprocess.fmeasure, preprocess.recall, preprocess.precision])
        return model

with tf.device("/GPU:0"):
    epochs = 1
    batch_size = 64

    # Load data using model preprocessor
    preprocess = Attention_model_preprocess.Preprocessor()

    X, Y, sample_word, sample_word_len = preprocess.load_data()
    word = X[0:1]
    print(X[0:1])
    print(sample_word)
    print(sample_word_len)
    print(word)
    print(word[0][77-sample_word_len:])

    # Define LSTM with attention model
    model_name = "CNN_BILSTM_ATT"
    model = cnn_bidirection_lstm_with_attention()

    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.11)

    #attention output
    layer_outputs = [layer.output for layer in model.layers if layer.name == 'attention_vec']
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    predict_output = activation_model.predict(X[0:1])
    dols = np.mean(predict_output, axis=2).squeeze()
    assert (np.sum(dols) - 1.0) < 1e-5
    dols = dols.tolist()
    dols = dols[77-sample_word_len:]
    dols = min_max_normalize(dols)
    print(dols)

    #visualize
    plt.bar(range(len(dols)), dols)
    plt.show()
    plt.savefig('fig1.png', dpi=300)
    '''
    y_pred = model.predict(X_test, batch_size=64)

    evaluator = model_evaluate.Evaluator()

    # Validation curves
    evaluator.plot_validation_curves(model_name, history)
    evaluator.print_validation_report(history)

    # Experimental result
    evaluator.calculate_measrue(model, X_test, y_test)

    # Save confusion matrix
    evaluator.plot_confusion_matrix(model_name, y_test, y_pred, title='Confusion matrix', normalize=True)

    # model.summary()

    # Save final training model
    preprocess.save_model(model, "../models/" + model_name + ".json", "../models/" + model_name + ".h5")
    '''