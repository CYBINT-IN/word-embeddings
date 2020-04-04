#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 05:00:04 2020

@author: tanmay

"""

import fasttext
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras.datasets import imdb
from gensim.models import Word2Vec, FastText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data(vocab_size,max_len):
    """
        Loads the keras imdb dataset
        Args:
            vocab_size = {int} the size of the vocabulary
            max_len = {int} the maximum length of input considered for padding
        Returns:
            X_train = tokenized train data
            X_test = tokenized test data
    """
    
    INDEX_FROM = 3
    # skip the words "the" and "and"

    (X_train,y_train),(X_test,y_test) = imdb.load_data(num_words = vocab_size,index_from = INDEX_FROM)
    
        
    return X_train,X_test,y_train,y_test


def prepare_data_for_word_vectors_imdb(X_train):
    """
        Prepares the input
        Args:
            X_train = tokenized train data
        Returns:
            sentences = {list} sentences containing words as tokens
            word_index = {dict} word and its indexes in whole of imdb corpus
    """
    INDEX_FROM = 3
    word_to_index = imdb.get_word_index()
    word_to_index = {k:(v+INDEX_FROM) for k,v in word_to_index.items()}

    word_to_index["<START>"] = 1
    word_to_index["<UNK>"] = 2

    index_to_word = {v:k for k,v in word_to_index.items()}

    sentences = []
    for i in range(len(X_train)):
        temp = [index_to_word[ids] for ids in X_train[i]]
        sentences.append(temp)

    return sentences, word_to_index


def  prepare_data_for_word_vectors_imdb_tf(corpus):
    """
        Prepares the input in a tf tokenizer format
        Args:
            X_train = tokenized train data
        Returns:
            sentences = {list} sentences containing words as tokens
            word_index = {dict} word and its indexes in whole of imdb corpus
    """
    tokenizer = Tokenizer()
    sentences = tokenizer.fit_on_texts(corpus)
    word_indexes = tokenizer.word_index
    
    return sentences, word_indexes


def padding_input(X_train,X_test,maxlen):
    """
        Pads the input upto considered max length
        Args:
            X_train = tokenized train data
            X_test = tokenized test data
        Returns:
            X_train_pad = padded tokenized train data
            X_test_pad = padded tokenized test data
    """

    X_train_pad = pad_sequences(X_train, maxlen = maxlen, padding = "post")

    X_test_pad = pad_sequences(X_test, maxlen = maxlen, padding = "post")

    return X_train_pad, X_test_pad


def data_prep_ELMo(train_x,train_y,test_x,test_y,max_len):

    INDEX_FROM = 3
    word_to_index = imdb.get_word_index()
    word_to_index = {k:(v+INDEX_FROM) for k,v in word_to_index.items()}

    word_to_index["<START>"] =1
    word_to_index["<UNK>"]=2

    index_to_word = {v:k for k,v in word_to_index.items()}

    sentences=[]
    for i in range(len(train_x)):
        temp = [index_to_word[ids] for ids in train_x[i]]
        sentences.append(temp)

    test_sentences=[]
    for i in range(len(test_x)):
        temp = [index_to_word[ids] for ids in test_x[i]]
        test_sentences.append(temp)

    train_text = [' '.join(sentences[i][:max_len]) for i in range(len(sentences))]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_y.tolist()

    test_text = [' '.join(test_sentences[i][:500]) for i in range(len(test_sentences))]
    test_text = np.array(test_text , dtype=object)[:, np.newaxis]
    test_label = test_y.tolist()

    return train_text,train_label,test_text,test_label


def building_word_vector_model(option, sentences, embed_dim, workers, window, y_train):
    """
        Builds the word vector
        Args:
            type = {bool} 0 for Word2vec. 1 for gensim Fastext. 2 for Fasttext 2018.
            sentences = {list} list of tokenized words
            embed_dim = {int} embedding dimension of the word vectors
            workers = {int} no. of worker threads to train the model (faster training with multicore machines)
            window = {int} max distance between current and predicted word
            y_train = y_train
        Returns:
            model = Word2vec/Gensim fastText/ Fastext_2018 model trained on the training corpus
    """
    if option == 0:
        print("Training a word2vec model")
        model = Word2Vec(sentences = sentences, size = embed_dim, workers = workers, window = window, epochs = 10)
        print("Training complete")

    elif option == 1:
        print("Training a Gensim FastText model")
        model = FastText(sentences = sentences, size = embed_dim, workers = workers, window = window, iter = 10)
        print("Training complete")

    elif option == 2:
        print("Training a Fasttext model from Facebook Research")
        y_train = ["__label__positive" if i == 1 else "__label__negative" for i in y_train]

        with open("imdb_train.txt","w") as text_file:
            for i in range(len(sentences)):
                print(sentences[i],y_train[i],file = text_file)

        model = fasttext.skipgram("imdb_train.txt","model_ft_2018_imdb",dim = embed_dim)
        print("Training complete")

    return model


def ELMoEmbedding(x):
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/1", trainable = True)
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict = True)["default"]