#data_utils
import numpy as np
import pickle
import os

def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result

def load_word2vec(params):
    """
    load pretrain word2vec weight matrix
    :param vocab_size:
    :return:
    """
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
