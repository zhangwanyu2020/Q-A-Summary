# help_functions
import tensorflow as tf


def load_word2vec(params):
    word2vec_dict = load_pkl(params.word2vec_output)
    vocab_dict = open(params.vocab_path, encoding='utf-8').readlines()
    embedding_matrix = np.zeros((params.vocab_size, params.embed_size))

    for line in vocab_dict[:params.vocab_size]:
        word_id = line.split()
        word, i = word_id
        embedding_vector = word2vec_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[int(i)] = embedding_vector

    return embedding_matrix


def calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
    # p_gen*vocab_dists,(1-p_gen)*attn_dists
    vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
    attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]
    # 词典加上oov词
    extended_size = vocab_size + batch_oov_len
    # extra_zeros.shape=[8,batch_oov_len] 假设batch_oov_len=2
    extra_zeros = tf.zeros((batch_size, batch_oov_len))
    # vocab_dists.shape=[8,50,30000]--->[8,50,30002]
    vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)

    attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
    # tf.tile 平铺张量
    batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
    indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
    shape = [batch_size, extended_size]
    # list length max_dec_steps (batch_size, extended_size)
    # tf.scatter_nd 通过索引将单个元素插入张量
    attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

    final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                   zip(vocab_dists_extended, attn_dists_projected)]

    return final_dists


# data_utils
import numpy as np
import pickle
import os


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def load_word2vec(params):
    word2vec_dict = load_pkl(params.word2vec_output)
    vocab_dict = open(params.vocab_path, encoding='utf-8').readlines()
    embedding_matrix = np.zeros((params.vocab_size, params.embed_size))

    for line in vocab_dict[:params.vocab_size]:
        word_id = line.split()
        word, i = word_id
        embedding_vector = word2vec_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[int(i)] = embedding_vector

    return embedding_matrix


# 损失函数
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')


def loss_function(real, outputs, padding_mask, cov_loss_wt, use_coverage):
    pred = outputs["logits"]
    attn_dists = outputs["attentions"]
    if use_coverage:
        loss = pgn_log_loss_function(real, pred, padding_mask) + cov_loss_wt * _coverage_loss(attn_dists, padding_mask)
        return loss
    else:
        return seq2seq_loss_function(real, pred, padding_mask)


def seq2seq_loss_function(real, pred, padding_mask):
    """
    跑seq2seq时用的Loss
    :param real: shape=(16, 50)
    :param pred: shape=(16, 50, 30000)
    :return:
    """
    loss = 0
    for t in range(real.shape[1]):
        loss_ = loss_object(real[:, t], pred[:, t])
        mask = tf.cast(padding_mask[:, t], dtype=loss_.dtype)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        loss_ = tf.reduce_mean(loss_)
        loss += loss_
    return loss / real.shape[1]


def pgn_log_loss_function(real, final_dists, padding_mask):
    # Calculate the loss per step
    # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
    loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
    batch_nums = tf.range(0, limit=real.shape[0])  # shape (batch_size)
    for dec_step, dist in enumerate(final_dists):
        # The indices of the target words. shape (batch_size)
        targets = real[:, dec_step]
        indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
        gold_probs = tf.gather_nd(dist, indices)  # shape (batch_size). prob of correct words on this step
        losses = -tf.math.log(gold_probs)
        loss_per_step.append(losses)
    # Apply dec_padding_mask and get loss
    _loss = _mask_and_avg(loss_per_step, padding_mask)
    return _loss


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)
    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
    Returns:
      a scalar
    """
    # padding_mask is Tensor("Cast_2:0", shape=(64, 400), dtype=float32)
    padding_mask = tf.cast(padding_mask, dtype=values[0].dtype)
    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex)  # overall average


def _coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.
    Args:
      attn_dists: The attention distributions for each decoder timestep.
      A list length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).
    Returns:
      coverage_loss: scalar
    """
    coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
    # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    covlosses = []
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss