from gru4rec_tf import SessionDataIterator
import tensorflow as tf
import numpy as np

def batch_eval(gru, test_data, cutoff=[20], batch_size=512, mode='conservative', item_key='ItemId', session_key='SessionId', time_key='Time'):
    if gru.error_during_train: 
        raise Exception('Attempting to evaluate a model that wasn\'t trained properly (error_during_train=True)')
    recall = dict()
    mrr = dict()
    for c in cutoff:
        recall[c] = 0
        mrr[c] = 0
    H = []
    for i in range(len(gru.layers)):
        H.append(tf.zeros((batch_size, gru.layers[i]), dtype='float32'))
    n = 0
    ii = 0
    data_iterator = SessionDataIterator(test_data, batch_size, 0, 0, 0, item_key, session_key, time_key, itemidmap=gru.data_iterator.itemidmap)
    reset_hook = lambda n_valid, finished_mask, valid_mask: gru._adjust_hidden(n_valid, finished_mask, valid_mask, H)
    for in_idx, out_idx in data_iterator(enable_neg_samples=False, reset_hook=reset_hook):
            O, H2 = gru.model.call(in_idx, H, None, training=False)
            for i in range(len(H)):
                 H[i] = H2[i]
            oscores = tf.transpose(O)
            tscores = tf.linalg.diag_part(tf.gather(oscores, out_idx))
            if mode == 'standard': ranks = tf.reduce_sum(tf.cast(oscores > tscores, dtype='int32'), axis=0) + 1
            elif mode == 'conservative': ranks = tf.reduce_sum(tf.cast(oscores >= tscores, dtype='int32'), axis=0)
            elif mode == 'median':  ranks = tf.reduce_sum(tf.cast(oscores > tscores, dtype='int32'), axis=0) + 0.5*(tf.reduce_sum(tf.cast(oscores == tscores, dtype='int32'), axis=0) - 1) + 1
            else: raise NotImplementedError
            for c in cutoff:
                recall[c] += tf.reduce_sum(tf.cast(ranks <= c, dtype='int32'))
                mrr[c] += tf.reduce_sum(tf.cast(ranks <= c, dtype='int32') / ranks)
            n += O.shape[0]
    for c in cutoff:
        recall[c] /= n
        mrr[c] /= n
    return recall, mrr
