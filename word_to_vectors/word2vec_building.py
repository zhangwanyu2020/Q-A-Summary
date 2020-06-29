from jieba import posseg
import jieba
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
import csv
import os
import pickle
from collections import defaultdict
import numpy as np


# 读取文件
def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip('\n'))
    return lines


# 保存词向量文件
def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s ok." % pkl_path)


# 定义分词模式
def segment(sentence, cut_type='word', pos=False):
    seg_words = []
    seg_pos = []

    if cut_type == 'word':
        if pos == True:
            seg_word_pos = posseg.lcut(sentence)
            for word, pos in seg_word_pos:
                seg_words.append(word)
                seg_pos.append(pos)
            return seg_words, seg_pos
        elif pos == False:
            seg_words = jieba.lcut(sentence)
            return seg_words

    if cut_type == 'char':
        if pos == True:
            for char in sentence:
                seg_word_pos = posseg.lcut(char)
                for word, pos in seg_word_pos:
                    seg_words.append(word)
                    seg_pos.append(pos)
            return seg_words, seg_pos
        elif pos == False:
            for char in sentence:
                seg_words.append(char)
            return seg_words

        # 读取原始数据,输出X Y


def parse_data(train_path, test_path):
    train_df = pd.read_csv(train_path, encoding='utf-8')
    train_df.dropna(subset=['Report'], how='any', inplace=True)  # inplace=True,保留对原数据的修改
    train_df.fillna('', inplace=True)
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    train_y = train_df.Report

    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_y = []

    assert len(train_x) == len(train_y)

    return train_x, train_y, test_x, test_y


# 输入X Y,输出分词后的x y文件
def save_data(train_x, train_y, test_x, path_train_x, path_train_y, path_test_x, path_stopwords):
    stop_words = read_file(path_stopwords)
    with open(path_train_x, 'w', encoding='utf-8-sig') as f1:
        count1 = 0
        writer = csv.writer(f1)
        for line in train_x:
            if isinstance(line, str):
                line = line.strip().replace(' ', '')
                seg_words = segment(line, cut_type='word', pos=False)
                seg_words = [word for word in seg_words if word not in stop_words]
                if len(seg_words) > 0:
                    seg_words = ' '.join(seg_words)
                    writer.writerow([seg_words])
                    count1 += 1
        print('len of train x is {}'.format(count1))

    with open(path_train_y, 'w', encoding='utf-8-sig') as f2:
        count2 = 0
        writer = csv.writer(f2)
        for line in train_y:
            if isinstance(line, str):
                line = line.strip().replace(' ', '')
                seg_words = segment(line, cut_type='word', pos=False)
                seg_words = [word for word in seg_words if word not in stop_words]
                if len(seg_words) > 0:
                    seg_words = ' '.join(seg_words)
                    writer.writerow([seg_words])
                else:
                    writer.writerow([seg_words])
                count2 += 1
        print('len of train_y is {}'.format(count2))

    with open(path_test_x, 'w', encoding='utf-8-sig') as f3:
        count3 = 0
        writer = csv.writer(f3)
        for line in test_x:
            if isinstance(line, str):
                line = line.strip().replace(' ', '')
                seg_words = segment(line, cut_type='word', pos=False)
                seg_words = [word for word in seg_words if word not in stop_words]
                if len(seg_words) > 0:
                    seg_words = ' '.join(seg_words)
                    writer.writerow([seg_words])
                    count3 += 1
        print('len of test_x is {}'.format(count3))

    # 输入分词后的x y文件，输出合并后的文件


def save_sentences(path_train_x, path_train_y, path_test_x, path_train_union_test):
    # 读三个文件并union
    sentences = read_file(path_train_x)
    sentences = sentences + read_file(path_train_y)
    sentences = sentences + read_file(path_test_x)

    # 将合并语料写出到文件
    with open(path_train_union_test, 'w', encoding='utf-8-sig', newline='') as f:
        for sentence in sentences:
            f.write(sentence)


# 由于语料生成词典
def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # sort
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)

    vocab = [(w, i) for i, w in enumerate(result)]
    reverse_vocab = [(i, w) for i, w in enumerate(result)]
    return vocab, reverse_vocab


# 训练词向量，并保存词向量
def build_w2v(path_train_union_test, path_words_vectors, w2v_model_path='w2v.model', min_count=10):
    w2v = Word2Vec(sentences=LineSentence(path_train_union_test), size=256, window=5, min_count=min_count, iter=5)
    w2v.save(w2v_model_path)

    model = Word2Vec.load(w2v_model_path)
    model = KeyedVectors.load(w2v_model_path)

    words_vectors = {}
    for word in model.wv.vocab:
        words_vectors[word] = model[word]

    dump_pkl(words_vectors, path_words_vectors, overwrite=True)

if __name__=='__main__':
    train_x, train_y, test_x, _ = parse_data('/Users/zhangwanyu/AutoMaster_TrainSet.csv',
                                         '/Users/zhangwanyu/AutoMaster_TestSet.csv')
    save_data(train_x, train_y, test_x, '/Users/zhangwanyu/Desktop/test_data/train_x.txt',
          '/Users/zhangwanyu/Desktop/test_data/train_y.txt', '/Users/zhangwanyu/Desktop/test_data/test_x.txt',
          '/Users/zhangwanyu/stop_words.txt')
    save_sentences('/Users/zhangwanyu/Desktop/test_data/train_x.txt', '/Users/zhangwanyu/Desktop/test_data/train_y.txt',
               '/Users/zhangwanyu/Desktop/test_data/test_x.txt',
               '/Users/zhangwanyu/Desktop/test_data/train_union_test.txt')
    build_w2v('/Users/zhangwanyu/Desktop/test_data/train_union_test.txt',
          '/Users/zhangwanyu/Desktop/test_data/words_vectors.txt')

    lines = read_file('/Users/zhangwanyu/Desktop/test_data/train_union_test.txt')
    vocab, reverse_vocab = build_vocab(lines)

    save_word_dict(vocab, '/Users/zhangwanyu/Desktop/test_data/dict_from_corpus.txt')

