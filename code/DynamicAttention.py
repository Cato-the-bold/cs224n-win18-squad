from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import sys
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops
from functools import partial
from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn, masked_softmax
from QAModel import QAModel

logging.basicConfig(level=logging.INFO)

class DynamicAttention(QAModel):
    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        super(DynamicAttention, self).__init__(FLAGS, id2word, word2id, emb_matrix)

    def add_placeholders(self):
        super(DynamicAttention, self).add_placeholders()

    def build_graph(self):
        attn_layer = DynamicAttention_Attn(self.keep_prob, self.FLAGS)
        output = attn_layer.build_graph(self.qn_embs, self.qn_mask, self.context_embs, self.context_mask) # attn_output is shape (batch_size, context_len, hidden_size*2)

        encoder = RNNEncoder(self.FLAGS.embedding_size*2, self.keep_prob)
        context_hiddens = encoder.build_graph(output, self.context_mask) # (batch_size, context_len, embedding_size*4)

        blended_reps_final = tf.contrib.layers.fully_connected(context_hiddens, num_outputs=self.FLAGS.hidden_size) # blended_reps_final is shape (batch_size, context_len, hidden_size)

        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)

        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)


class DynamicAttention_Attn(BasicAttn):
    def __init__(self, keep_prob, flag):
        self.keep_prob = keep_prob
        self.FLAGS = flag
        self.keys_sentinel = tf.Variable(tf.random_normal([1, 1, self.FLAGS.embedding_size]), dtype=tf.float32)
        self.values_sentinel = tf.Variable(tf.random_normal([1, 1, self.FLAGS.embedding_size]), dtype=tf.float32)

    def build_graph(self, values, values_mask, keys, keys_mask):
        sentinel_padding = tf.constant(1, shape=[1, 1])
        batch_size = self.FLAGS.batch_size
        with vs.variable_scope("Attention"):
            # Calculate attention distribution
            dense_layer = partial(tf.layers.dense, activation = tf.nn.tanh, kernel_regularizer = tf.contrib.layers.l1_regularizer(0.001))
            projected_values_t = dense_layer(values, self.FLAGS.embedding_size)

            values_t = tf.concat([projected_values_t, tf.broadcast_to(self.values_sentinel, [batch_size, 1, self.FLAGS.embedding_size])], 1) # (batch_size, value_vec_size, num_values)

            #augmented context vectors.
            keys_t = tf.concat([keys, tf.broadcast_to(self.keys_sentinel, [batch_size, 1, self.FLAGS.embedding_size])], 1)

            affinity_scores = tf.matmul(keys_t, tf.transpose(values_t, perm=[0, 2, 1])) # shape (batch_size, num_keys, num_values)

            values_mask_1 = tf.expand_dims(tf.concat([values_mask, tf.broadcast_to(sentinel_padding, [batch_size, 1])],1), 1) #shape (batch_size, 1, num_values).
            _, C2Q_softmax = masked_softmax(affinity_scores, values_mask_1, 2) # shape (batch_size, num_keys, num_values). take softmax over values
            attn_output_1 = tf.matmul(C2Q_softmax, values_t) # shape (batch_size, num_keys, value_vec_size)

            keys_mask_1 = tf.expand_dims(tf.concat( [keys_mask, tf.broadcast_to(sentinel_padding, [batch_size, 1]) ],1), 2)  #shape (batch_size, num_keys, 1)
            _, Q2C_softmax = masked_softmax(affinity_scores, keys_mask_1, 1)
            Q2C_output = tf.matmul(tf.transpose(Q2C_softmax, perm=[0,2,1]),keys_t)

            attn_output_2 = tf.matmul(C2Q_softmax, Q2C_output)

            key_hidden = tf.concat([attn_output_2, attn_output_1], 2)

            key_hidden = key_hidden[:, :self.FLAGS.context_len, :]

            # Apply dropout
            output = tf.nn.dropout(key_hidden, self.keep_prob)

            return output
