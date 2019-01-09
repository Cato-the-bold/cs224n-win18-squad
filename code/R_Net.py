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

class R_Net(QAModel):
    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        super(R_Net, self).__init__(FLAGS, id2word, word2id, emb_matrix)

    def build_graph(self):
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_embs,
                                                  self.context_mask)  # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.qn_embs,
                                                   self.qn_mask)  # (batch_size, question_len, hidden_size*2)

        # Use context hidden states to attend to question hidden states
        attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size * 2)
        _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask,
                                                context_hiddens)  # attn_output is shape (batch_size, context_len, hidden_size*2)

        attn_layer = R_Net_Attn(self.keep_prob, self.FLAGS.hidden_size * 2, self.FLAGS)
        output = attn_layer.build_graph(attn_output, self.context_mask) # attn_output is shape (batch_size, context_len, hidden_size*2)

        blended_reps_final = tf.contrib.layers.fully_connected(tf.concat([attn_output, output], 2), num_outputs=self.FLAGS.hidden_size) # blended_reps_final is shape (batch_size, context_len, hidden_size)

        with vs.variable_scope("StartDist"):
            softmax_layer_start = SimpleSoftmaxLayer()
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)

        with vs.variable_scope("EndDist"):
            softmax_layer_end = SimpleSoftmaxLayer()
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)


class R_Net_Attn(BasicAttn):
    def __init__(self, keep_prob, hidden_vec_size, flag):
        self.keep_prob = keep_prob
        self.hidden_vec_size = hidden_vec_size
        self.FLAGS = flag

    def build_graph(self, keys, keys_mask):
        with vs.variable_scope("Attention"):
            dense_layer_1 = partial(tf.layers.dense, activation = None, use_bias=False, kernel_regularizer = tf.contrib.layers.l1_regularizer(0.001))
            dense_layer_2 = partial(tf.layers.dense, activation = None, use_bias=False, kernel_regularizer = tf.contrib.layers.l1_regularizer(0.001))
            projected_keys_1 = dense_layer_1(keys, self.hidden_vec_size) # (batch_size, num_keys, hidden_vec_size)
            projected_keys_2 = dense_layer_2(keys, self.hidden_vec_size) # (batch_size, num_keys, hidden_vec_size)
            keys_t = tf.expand_dims(projected_keys_1,2)+tf.expand_dims(projected_keys_2,1)
            keys_t.set_shape([self.FLAGS.batch_size, self.FLAGS.context_len, self.FLAGS.context_len, self.hidden_vec_size])
            keys_t = tf.nn.tanh(keys_t)
            V = partial(tf.layers.dense, activation = None, use_bias=False, kernel_regularizer = tf.contrib.layers.l1_regularizer(0.001))
            self_attn_keys = tf.squeeze(V(keys_t,1))

            _, self_attn_softmax = masked_softmax(self_attn_keys, tf.expand_dims(keys_mask, 1), 1)
            output = tf.matmul(self_attn_softmax, keys) #no tranpose needed due to symmetric, shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return output
