import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torch.utils import data as torch_data
from keras import backend as K
from tf.keras.layers import Layer


class AttentionForClassification(Layer):
    def __init__(self,
                 activation='tanh'):
        self.activation = activation
        super(AttentionForClassification, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.b = self.add_weight(name='bias',
                                 shape=(1,),
                                 initializer='zeros',
                                 trainable=True)
        self.W = self.add_weight(name='kernel',
                                      shape=(int(input_shape[1]), 1),
                                      initializer='uniform',
                                      trainable=True)
        super(AttentionForClassification, self).build(input_shape)

    def call(self, x):
        output = K.tanh(
            K.squeeze(K.dot(K.permute_dimensions(x, (0, 2, 1)), self.W)+self.b, 2))
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


def fasttext(input_dim, emb_dim, hidden_dim, output_dim):
    """
    build a fasttext model using keras, without hierachical softmax.
    ======================================================================
    param input_dim:
        int, total num of words that we are going to use.
    param emb_dim:
        int, the num of dimension a word will be embedded.
    param hidden_dim:
        int, add a hidden dim to make the model deep.
    param output_dim:
        int, num of labels, if output_dim == 1, use linear activation func.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(
        input_dim=input_dim, output_dim=emb_dim))
    model.add(tf.keras.layers.Dense(units=hidden_dim, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(units=hidden_dim, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    if output_dim == 1:
        model.add(tf.keras.layers.Dense(units=output_dim, activation='linear'))
        model.compile(optimizer='adam',
                      loss='mean_squared_error', metrics=['mse'])
    else:
        model.add(tf.keras.layers.Dense(
            units=output_dim, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def textcnn(input_dim, seq_length, emb_dim, output_dim):
    """
    build a textcnn model using keras, with 2/3/4-gram .
    ======================================================================
    param input_dim:
        int, total num of words that we are going to use.
    param seq_length:
        int, the padding length of each sentence.
    param emb_dim:
        int, the num of dimension a word will be embedded.
    param output_dim:
        int, num of labels, if output_dim == 1, use linear activation func.
    """
    x = tf.keras.layers.Input(shape=(seq_length,))
    emb = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=emb_dim)(x)
    n_gram_conv = [2, 3, 4]
    pooling_vec = []
    for n_gram in n_gram_conv:
        conv_i_vec = tf.keras.layers.Conv1D(filters=2, kernel_size=n_gram)(emb)
        pooling_i_vec = tf.keras.layers.MaxPool1D(
            pool_size=int(conv_i_vec.shape[1]))(conv_i_vec)
        pooling_vec.append(pooling_i_vec)
    concated_vec = tf.keras.layers.Lambda(
        lambda x: K.concatenate(x, axis=2))(pooling_vec)
    concated_vec = tf.keras.layers.Flatten()(concated_vec)
    if output_dim == 1:
        output = tf.keras.layers.Dense(
            units=output_dim, activation='linear')(concated_vec)
        loss = 'mse'
    else:
        output = tf.keras.layers.Dense(
            units=output_dim, activation='softmax')(concated_vec)
        loss = 'categorical_crossentropy'
    model = tf.keras.models.Model(inputs=x, outputs=output)
    model.compile(optimizer='adam', loss=loss)
    return model


def rcnn(input_dim, seq_length, emb_dim, hidden_dim, output_dim):
    """
    build a rcnn model using keras.
    ======================================================================
    param input_dim:
        int, total num of words that we are going to use.
    param seq_length:
        int, the padding length of each sentence.
    param emb_dim:
        int, the num of dimension a word will be embedded.
    param hidden_dim:
        int, the num of hidden units of recurrent layer.
    param output_dim:
        int, num of labels, if output_dim == 1, use linear activation func.
    """
    x = tf.keras.layers.Input(shape=(seq_length,))
    emb = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=emb_dim)(x)
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=hidden_dim, return_sequences=True))(emb)
    concating = tf.keras.layers.Lambda(
        lambda x: K.concatenate(x, axis=2))([emb, bi_lstm])
    pooling = tf.keras.layers.GlobalMaxPooling1D()(concating)
    if output_dim > 1:
        output = tf.keras.layers.Dense(
            activation='sigmoid', units=output_dim)(pooling)
        loss = 'mse'
    elif output_dim == 1:
        output = tf.keras.layers.Dense(
            activation='sigmoid', units=output_dim)(pooling)
        loss = 'categorical_crossentropy'
    model = tf.keras.Model(inputs=x, outputs=output)
    model.compile(optimizer='adam', loss=loss)
    return model


def create_torch_dataset(x, y, batch_size):
    """
    create a iterable dataset in pytorch mode
    ======================================================================
    param x:
        torch.Tensor, features
    param y:
        torch.Tensor, target
    """
    dataloader = torch_data.DataLoader(dataset=torch_data.TensorDataset(x, y),
                                       batch_szie=batch_size)
    return dataloader


class TorchFastText(nn.Module):
    """
    build a fasttext model using pytorch, without hierachical softmax.
    ======================================================================
    param input_dim:
        int, total num of words that we are going to use.
    param emb_dim:
        int, the num of dimension a word will be embedded.
    param hidden_dim:
        int, add a hidden dim to make the model deep.
    param time_step:
        int, the sequence length of each sentence to feed in the model.
    param output_dim:
        int, num of labels.
    """

    def __init__(self, input_dim, emb_dim, hidden_dim, time_step, output_dim):
        super(TorchFastText, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.emb_layer = nn.Embedding(num_embeddings=input_dim,
                                      embedding_dim=emb_dim)
        self.hidden_layer_1 = nn.Linear(in_features=emb_dim,
                                        out_features=hidden_dim)
        self.relu_1 = nn.ReLU()
        self.batch_norm_1 = nn.BatchNorm1d(num_features=time_step)
        self.hidden_layer_2 = nn.Linear(in_features=hidden_dim,
                                        out_features=hidden_dim)
        self.relu_2 = nn.ReLU()
        self.batch_norm_2 = nn.BatchNorm1d(num_features=hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.output_layer = nn.Linear(in_features=hidden_dim,
                                      out_features=output_dim)

    def forward(self, x):
        emb = self.emb_layer(x)
        h1 = self.relu_1(self.hidden_layer_1(emb))
        norm_h1 = self.batch_norm_1(h1)
        avg_pooling = torch.mean(norm_h1, 1)
        h2 = self.relu_2(self.hidden_layer_2(avg_pooling))
        norm_h2 = self.batch_norm_2(h2)
        output = self.output_layer(norm_h2)
        output = self.softmax(output)
        return output


class TorchTextCNN(nn.Module):
    """
    build a textcnn model using pytorch, with 2/3/4-gram .
    ======================================================================
    param input_dim:
        int, total num of words that we are going to use.
    param seq_length:
        int, the padding length of each sentence.
    param emb_dim:
        int, the num of dimension a word will be embedded.
    param output_dim:
        int, num of labels, if output_dim == 1, use linear activation func.
    """

    def __init__(self, input_dim, seq_length, emb_dim, output_dim):
        super(TorchTextCNN, self).__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.emb_layer = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=emb_dim)
        self.n_gram = [2, 3, 4]
        self.conv_layers = []
        self.max_pooling_layers = []
        for i in self.n_gram:
            self.conv_layers.append(nn.Conv1d(in_channels=emb_dim,
                                              out_channels=2,
                                              kernel_size=i))
            self.max_pooling_layers.append(
                nn.MaxPool1d(kernel_size=seq_length-i+1))
        self.output_layer = nn.Linear(in_features=len(
            self.n_gram)*2, out_features=output_dim)

    def forward(self, x):
        emb = self.emb_layer(x)
        after_pooling = []
        for i in range(len(self.n_gram)):
            conv_i_vec = self.conv_layers[i](torch.transpose(emb, 1, 2))
            max_pooling_i = torch.squeeze(
                self.max_pooling_layers[i](conv_i_vec))
            after_pooling.append(max_pooling_i)
        cat_pooling = torch.cat(after_pooling, 1)
        output = self.output_layer(cat_pooling)
        return output


class TorchRCNN(nn.Module):
    """
    build a rcnn model using pytorch.
    ======================================================================
    param input_dim:
        int, total num of words that we are going to use.
    param seq_length:
        int, the padding length of each sentence.
    param emb_dim:
        int, the num of dimension a word will be embedded.
    param hidden_dim:
        int, the num of hidden units of recurrent layer.
    param output_dim:
        int, num of labels.
    """

    def __init__(self, input_dim, seq_length, emb_dim, hidden_dim, output_dim):
        super(TorchRCNN, self).__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.emb_layer = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=emb_dim)
        self.lstm_layer = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden_dim, bidirectional=True)
        self.pooling_layer = nn.MaxPool1d(kernel_size=seq_length)
        self.output_layer = nn.Linear(
            in_features=2*hidden_dim+emb_dim, out_features=output_dim)

    def forward(self, x):
        emb = self.emb_layer(x)
        out, h = self.lstm_layer(emb)
        after_cat = torch.cat([emb, out], dim=2)
        after_pooling = self.pooling_layer(
            after_cat.permute(0, 2, 1)).squeeze(2)
        output = self.output_layer(after_pooling)
        return output
