import tensorflow as tf

# 模型参数
class Args:
    max_enc_len = 200
    max_dec_len = 50
    max_dec_steps = 100
    min_dec_steps = 30
    batch_size = 64
    vocab_size = 30000
    embed_size = 256
    enc_units = 256
    dec_units = 256
    attn_units = 256
    learning_rate = 0.001
    seq2seq_model_dir = '/Users/zhangwanyu/Desktop/test_data/seq2seq'
    model_path = ""
    train_seg_x_dir = 'train_x.txt'
    train_seg_y_dir = 'train_y.txt'
    test_seg_x_dir = 'test_x.txt'
    vocab_path = 'dict_from_corpus.txt'
    word2vec_output = 'word_vectors.txt'
    test_save_dir = '/Users/zhangwanyu/Desktop/test_data/'
    test_x_dir = 'AutoMaster_TestSet.csv'
    steps_per_epoch = 200
    checkpoints_save_steps = 10
    max_steps = 10000
    num_to_test = 10
    epochs = 5
    mode = 'test'
    model = 'SequenceToSequence'
    pointer_gen = True
    is_coverage = True
    greedy_decode = True
    transformer = False


params = Args()

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')

if params.mode == "train":
    train(params)

elif params.mode == "test":
    test(params)
    pass

