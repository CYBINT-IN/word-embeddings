#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 05:00:04 2020

@author: tanmay

"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
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
    # save np.load
    np_load_old = np.load
    
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    
    INDEX_FROM = 3

    (X_train,y_train),(X_test,y_test) = imdb.load_data(num_words = vocab_size,index_from = INDEX_FROM)
    
    # restore np.load for future normal usage
    np.load = np_load_old
        
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

    return sentences,word_to_index


def  prepare_data_for_word_vectors_imdb_tf(X_train):
    """
        Prepares the input in a tf tokenizer format
        Args:
            X_train = tokenized train data
        Returns:
            sentences = {list} sentences containing words as tokens
            word_index = {dict} word and its indexes in whole of imdb corpus
    """
    tokenizer = Tokenizer()
    sentences = tokenizer.fit_on_texts(X_train)
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


