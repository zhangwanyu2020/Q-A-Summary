import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import bert_master.modeling as modeling
from data_loader import TextLoader
import numpy as np
import tensorflow as tf


class Project_model():
    def __init__(self, bert_root, data_path, model_save_path, batch_size, max_len, max_sent, lr, keep_prob,data_mode):
        self.bert_root = bert_root
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_sent = max_sent
        self.lr = lr
        self.keep_prob = keep_prob
        self.data_mode = data_mode

        self.bert_config()
        self.get_output()
        self.get_loss(True)
        self.get_accuracy()
        self.get_trainOp()
        # self.init_saver()

    def bert_config(self):
        bert_config_file = os.path.join(self.bert_root, 'bert_config.json')
        # 获取预训练模型的参数文件
        self.bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        self.init_checkpoint = os.path.join(self.bert_root, 'bert_model.ckpt')
        self.bert_vocab_file = os.path.join(self.bert_root, 'vocab.txt')
        # 初始化变量
        self.input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.max_sent, 1], name="input_y")
        self.global_step = tf.Variable(0, trainable=False)
        output_weights = tf.get_variable("output_weights", [768, self.max_sent],
                                         initializer=tf.random_normal_initializer(stddev=0.1))
        output_bias = tf.get_variable("output_bias", [self.max_sent, ],
                                      initializer=tf.random_normal_initializer(stddev=0.01))
        self.w_out = output_weights
        self.b_out = output_bias
        # 初始化bert model
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=False,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)
        # 变量赋值
        tvars = tf.trainable_variables()
        (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                               self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, assignment)
        # 这个获取句子的output，shape = 768
        output_layer_pooled = model.get_pooled_output()
        # 添加dropout层，减轻过拟合
        self.output_layer_pooled = tf.nn.dropout(output_layer_pooled, keep_prob=self.keep_prob)
        # return self.output_layer_pooled

    def get_output(self):
        # pred 全连接层 768-->max_sent
        self.pred = tf.add(tf.matmul(self.output_layer_pooled, self.w_out), self.b_out, name="pre1")
        # probabilities = [概率分布]，shape=[batch_size,max_sent]
        self.probabilities = tf.nn.softmax(self.pred, axis=-1, name='probabilities')  # 和sigmoid
        # log_probs = [对数概率分布]，shape=[batch_size,max_sent]
        self.log_probs = tf.nn.log_softmax(self.pred, axis=-1, name='log_probs')
        # 维度转换
        self.pred = tf.reshape(self.pred, shape=[-1, self.max_sent], name='pre')
        return self.pred

    def get_loss(self, if_regularization=True):
        net_loss = tf.square(tf.reshape(self.pred, [-1]) - tf.reshape(self.input_y, [-1]))

        if if_regularization:
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.w_out)
            tf.add_to_collection(tf.GraphKeys.BIASES, self.b_out)
            regularizer = tf.contrib.layers.l1_regularizer(scale=5.0 / 50000)
            reg_loss = tf.contrib.layers.apply_regularization(regularizer)
            net_loss = net_loss + reg_loss
        self.loss = tf.math.reduce_sum(net_loss)/(self.batch_size*self.max_sent)
        return self.loss

    def get_trainOp(self):
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        return self.train_op

    def get_accuracy(self):
        self.predicts = tf.argmax(self.pred, axis=-1)
        self.predicts_bool = tf.math.greater(self.probabilities, tf.constant(0.03))
        self.actuals = tf.argmax(self.input_y, axis=-1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicts, self.actuals), dtype=tf.float32))

    def evaluate(self, sess, devdata):
        data_loader = TextLoader(devdata, self.batch_size, self.max_sent,self.data_mode)
        predictions = []
        reals = []
        tokens = []
        for i in range(data_loader.num_batches):
            x_train, y_train, z_train= data_loader.next_batch(i)
            x_input_ids = x_train[:, 0]
            x_input_mask = x_train[:, 1]
            x_segment_ids = x_train[:, 2]
            feed_dict = {self.input_ids: x_input_ids, self.input_mask: x_input_mask,
                         self.segment_ids: x_segment_ids,
                         self.input_y: y_train}
            pre = sess.run(self.probabilities , feed_dict=feed_dict)
            predictions.append(pre)
            real = feed_dict[self.input_y]
            real.resize([self.batch_size, 32])
            reals.append(real)
            tokens.append(z_train)
        return predictions, reals, tokens


    def predict(self, sess, devdata):
        data_loader = TextLoader(devdata, self.batch_size, self.max_sent,self.data_mode)
        predictions = []
        tokens = []
        for i in range(data_loader.num_batches):
            x_train, z_train= data_loader.next_batch(i)
            x_input_ids = x_train[:, 0]
            x_input_mask = x_train[:, 1]
            x_segment_ids = x_train[:, 2]
            feed_dict = {self.input_ids: x_input_ids, self.input_mask: x_input_mask,
                         self.segment_ids: x_segment_ids}
            pre = sess.run(self.probabilities , feed_dict=feed_dict)
            predictions.append(pre)
            tokens.append(z_train)
        print('num_batches:',i+1)
        return predictions,tokens

    def run_step(self, sess, x_train, y_train):
        x_input_ids = x_train[:, 0]
        x_input_mask = x_train[:, 1]
        x_segment_ids = x_train[:, 2]
        step, loss_, _ = sess.run([self.global_step, self.loss, self.train_op],
                                  feed_dict={self.input_ids: x_input_ids, self.input_mask: x_input_mask,
                                             self.segment_ids: x_segment_ids,
                                             self.input_y: y_train})
        return step, loss_

    def get_quater(self,probabilities):
        print(probabilities)
        with tf.Session() as session:
            init = tf.global_variables_initializer()
            session.run(init)
            prob_list = probabilities.eval()
        p = []
        for i in range(len(prob_list)):
            b = np.percentile(self.prob_list[i], [75])
            p.append(b)
        p = np.array(p)
        print(p, type(p))
        return p