# encoder_decoder
import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units // 2
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        # tf.keras.layers.GRU自动匹配cpu、gpu
        self.gru = tf.keras.layers.GRU(self.enc_units,  # 输出单元的维度
                                       return_sequences=True,  # 是否返回整个序列的output
                                       return_state=True,  # 是否返回最后一个单元的state
                                       recurrent_initializer='glorot_uniform')  # 'glorot_uniform':基于方差缩放的初始化器

        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')
        # x.shape=[batch_size,sequence_len]=[8,200]

    def call(self, x, hidden):
        # 经过词嵌入，200个词变成200个向量，x.shape=[8,200,256]
        x = self.embedding(x)
        # 双向GRU的state先分裂，再拼接，而且state和序列长度没有关系的，state.shape=[batch_size,enc_units]=[8,256//2]
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        # output返回的是整个序列的state,output.shape=[batch_size,sequence_len,enc_uints]=[8,200,256]
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        # state.shape=[8,256] state是一个时间状态下的结果，它没有时间这个维度，所以计算时需扩展第二个维度
        state = tf.concat([forward_state, backward_state], axis=1)
        # [8,200,256],[8,256]
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2 * self.enc_units))


class BahdanauAttentionCoverage(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttentionCoverage, self).__init__()
        self.Wc = tf.keras.layers.Dense(units)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output, enc_padding_mask, use_coverage=False, prev_coverage=None):
        """
        :param dec_hidden: shape=(16, 256)
        :param enc_output: shape=(16, 200, 256)
        :param enc_padding_mask: shape=(16, 200)
        :param use_coverage:
        :param prev_coverage: None
        :return:
        """
        # dec_hidden.shape=(8,256)--->(8,1,256)
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        def masked_attention(score):
            """
            :param score: shape=(16, 200, 1)
                        ...
              [-0.50474256]
              [-0.47997713]
              [-0.42284346]]]
            :return:
            """
            # attn_dist.shape:[8, 200,1]--->[8,200]
            attn_dist = tf.squeeze(score, axis=2)
            # softmax不会改变维度，只是把score值变成概率值，shape=(8, 200)
            attn_dist = tf.nn.softmax(attn_dist, axis=1)
            mask = tf.cast(enc_padding_mask, dtype=attn_dist.dtype)
            attn_dist *= mask
            masked_sums = tf.reduce_sum(attn_dist, axis=1)
            attn_dist = attn_dist / tf.reshape(masked_sums, [-1, 1])
            attn_dist = tf.expand_dims(attn_dist, axis=2)
            return attn_dist
            # 计算1时刻之后的coverage

        if use_coverage and prev_coverage is not None:
            # e.shape=[batch_size,sequece_len,1]=[8,200,1]:{[8,200,256]+[8,200,256]+[8,200,1]}--->[8,200,1]
            e = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis) + self.Wc(prev_coverage)))
            # mask+softmax
            attn_dist = masked_attention(e)
            # t时刻的coverage就是t时刻之前的attn_dist之和
            coverage = attn_dist + prev_coverage

        else:
            #  如果不使用coverage，则attention的计算不需要之前的attention之和
            e = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
            attn_dist = masked_attention(e)
            # 如果使用coverage，则第一步的prev_coverage需用attn_dist初始化
            if use_coverage:
                coverage = attn_dist
            else:
                coverage = []
        # context_vectore.shape=[8,200,256]:[8,200,1]*[8,200,256]
        context_vector = attn_dist * enc_output
        # context_vector.shape=(8,256)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # coverage.shape=(8, 200, 1)
        # attn_dist.shape=(8,200)
        return context_vector, tf.squeeze(attn_dist, -1), coverage


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)

    def call(self, x, context_vector):
        # x.shape=[batch_size,1,embedding_dim]=[8,1,256] decoder时是一个词一个词解码，所以时间维度=1
        x = self.embedding(x)
        # x.shape=concat[[8,1,256],[(8,1,256]]=[8,1,512] 前一个256是enc_units，后一个256是词嵌入维度
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # output.shape=[8,1,256]
        # state.shape=[8,256]
        output, state = self.gru(x)
        # output.shape=[8,256]
        output = tf.reshape(output, (-1, output.shape[2]))
        # out.shape=(8,30000)
        out = self.fc(output)

        return x, out, state


class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, state, dec_inp):
        return tf.nn.sigmoid(self.w_s_reduce(state) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))