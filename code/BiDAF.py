from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn, masked_softmax
from QAModel import QAModel

logging.basicConfig(level=logging.INFO)

class BiDAF(QAModel):
    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        super(BiDAF, self).__init__(FLAGS, id2word, word2id, emb_matrix)

    def add_embedding_layer(self, emb_matrix):
        with vs.variable_scope("embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)

    def build_graph(self):
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        if self.FLAGS.max_word_len:
            context_hiddens = encoder.build_graph(tf.concat([self.context_embs, self.context_char_hidden],2), self.context_mask) # (batch_size, context_len, hidden_size*2)
            question_hiddens = encoder.build_graph(tf.concat([self.qn_embs, self.qn_char_hidden],2), self.qn_mask) # (batch_size, question_len, hidden_size*2)
        else:
            context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
            question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)

        attn_layer = BiDAF_Attn(self.keep_prob, self.FLAGS.hidden_size*2, [self.FLAGS.batch_size, self.FLAGS.context_len, self.FLAGS.question_len])
        output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens, self.context_mask) # attn_output is shape (batch_size, context_len, hidden_size*2)

        blended_reps_final = tf.contrib.layers.fully_connected(output, num_outputs=self.FLAGS.hidden_size)

        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)

        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)


class BiDAF_Attn(BasicAttn):
    def __init__(self, keep_prob, hidden_vec_size, shape):
        super(BiDAF_Attn,self).__init__(keep_prob, hidden_vec_size)
        self.shape = shape

    def build_graph(self, values, values_mask, keys, keys_mask):
        with vs.variable_scope("Attention"):
            dense_layer1 = partial(tf.layers.dense, activation = None, use_bias=False, kernel_regularizer = tf.contrib.layers.l1_regularizer(0.001))
            dense_layer2 = partial(tf.layers.dense, activation = None, use_bias=False, kernel_regularizer = tf.contrib.layers.l1_regularizer(0.001))

            score1 = dense_layer1(keys, 1) #shape (batch_size, num_keys, 1)
            score2 = dense_layer2(values, 1) #shape (batch_size, num_values, 1)

            #version1. too much memory. Or do (batch, k_len, 1, ndim) * (batch, 1, v_len, ndim).
            #k = tf.expand_dims(tf.traspose(keys, perm=[0,2,1]), 3)  # shape (batch_size, hidden_size, num_keys, 1).
            #v = tf.expand_dims(tf.traspose(values, perm=[0,2,1]), 2)
            #matrix = tf.traspose(tf.matmul(k, v), perm=[0,2,3,1])

            #version2. seems infeasible.
            # def matrix_func(keys, values, weight):
            #     mat = np.zeros(self.shape)
            #     for k in xrange(self.shape[0]):
            #         for i in xrange(self.shape[1]):
            #             for j in xrange(self.shape[2]):
            #                 for m in xrange(self.vec_size):
            #                     mat[k,i,j] += weight[m]*keys[k,i,m]*values[k,j,m]
            #     return mat
            # weight = tf.Variable(tf.random_normal([self.vec_size]), dtype=tf.float32, name="similarity_weight_3")
            # similarity_scores = tf.cast(tf.py_func(matrix_func, [keys, values, weight], tf.double), tf.float32)
            # similarity_scores.set_shape(self.shape[0:])

            #version3. memory efficient. associate the channel weight weight with keys in advance, then multiply the result with values.
            weight = tf.Variable(tf.random_normal([1,1,self.hidden_vec_size]), dtype=tf.float32, name="similarity_weight_3")
            weighted_keys = weight*keys
            similarity_scores = tf.matmul(weighted_keys, tf.transpose(values, perm=[0,2,1]))
            similarity_scores = score1 + tf.transpose(score2, perm=[0,2,1]) + similarity_scores # shape (batch_size, num_keys, num_values)

            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, C2Q_softmax = masked_softmax(similarity_scores, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values
            C2Q_output = tf.matmul(C2Q_softmax, values) # shape (batch_size, num_keys, value_vec_size)

            max_i = tf.reduce_max(similarity_scores,2)
            _, Q2C_softmax = masked_softmax(max_i, keys_mask, 1)  # shape(batch_size, num_keys)
            Q2C_softmax = tf.expand_dims(Q2C_softmax, -1)
            Q2C_output = tf.reduce_sum(Q2C_softmax * keys, 1, keepdims=True)  #or Q2C_output = tf.matmul(tf.transpose(keys, (0, 2, 1)), tf.expand_dims(Q2C_softmax, -1))

            output = tf.concat([keys, C2Q_output, tf.broadcast_to(Q2C_output, tf.shape(keys))], 2)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return output
