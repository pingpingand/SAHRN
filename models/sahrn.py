import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class SAHRN(BaseGAttN):
    def inference(inputs_list, nb_classes, nb_nodes, training, attn_drop, ffd_drop, keep_prob,
            bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False):


        h_1 = []  # attention result of the first layer

        for inputs, bias_mat in zip(inputs_list, bias_mat_list):

            # pro
            w_meta = tf.Variable(tf.random_normal([inputs.shape[-1], 300], stddev=0.1))
            x = tf.tensordot(inputs, w_meta, axes=1)
            # wo_pro
            # x = inputs
            attns = []
            for _ in range(n_heads[0]):  # n_heads[0] = 8
                attns.append(layers.attn_head(x, bias_mat=bias_mat,
                                              out_sz=hid_units[0], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=False))

            h_1.append(tf.concat(attns, axis=-1))

        # projection
        h_2 = tf.concat(h_1, axis=-1)
        # without_projection
        # h_2 = tf.zeros_like(h_1[0], dtype=tf.float32)
        # for h in h_1:
        #     h_2 = h_2 + h
        # h_2 = tf.nn.dropout(h_2, keep_prob)

        out = tf.layers.dense(h_2, nb_classes, activation=None)
        # out = tf.layers.dense(h_2, nb_classes, activation=tf.nn.elu)

        return out, h_2
