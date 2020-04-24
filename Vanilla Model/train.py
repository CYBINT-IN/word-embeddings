#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 05:38:25 2020

@author: tanmay
"""

import utils


x_train, x_test, y_train, y_test = utils.load_data(vocab_size = 10000, max_len = 100)

sentences, word_ix = utils.prepare_data_for_word_vectors_imdb(x_train)

model_wv = utils.building_word_vector_model(option = 1,embed_dim = 200,
                                       workers = 3, window = 1, sentences = sentences, y_train = y_train)

model_wv.save('model.model')