import tensorflow as tf


class PGN(tf.keras.Model):
    def __init__(self, params):
        super(PGN, self).__init__()
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        self.encoder = Encoder(params.vocab_size,
                               params.embed_size,
                               params.enc_units,
                               params.batch_size,
                               self.embedding_matrix
                               )
        self.attention = BahdanauAttentionCoverage(params.attn_units)
        self.decoder = Decoder(params.vocab_size,
                               params.embed_size,
                               params.dec_units,
                               params.batch_size,
                               self.embedding_matrix
                               )
        self.pointer = Pointer()
        # 第一步encoder得到一个序列的enc_output,和最后单元的enc_hidden

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call(self, enc_output, dec_hidden, enc_inp, enc_extended_inp, dec_inp,
             batch_oov_len, enc_padding_mask, use_coverage, pre_coverage):
        predictions = []
        attentions = []
        coverages = []
        p_gens = []
        # 第二步用enc_output,enc_hidden=dec_hidden，enc_padding_mask计算attn_dist，初始化pre_coverage
        context_vector, attn_dist, coverage_next = self.attention(dec_hidden,
                                                                  enc_output,
                                                                  enc_padding_mask,
                                                                  use_coverage,
                                                                  pre_coverage)
        # 第三步dec_inp.shape[1]=50，逐步对这50个词解码，按batch_size处理的 ':'==8行
        for t in range(dec_inp.shape[1]):
            # t时刻dec_input和context_vector拼接，得到pred为[8,30000]的概率分布
            dec_x, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1),
                                                   context_vector
                                                   )
            # 计算t+1时刻的context_vector
            context_vector, atten_dist, coverage_next = self.attention(dec_hidden,  # shape=[8,256]
                                                                       enc_output,  # shape=[8,200,256]
                                                                       enc_padding_mask,  # shape=[8,200]
                                                                       use_coverage,
                                                                       coverage_next)
            # 计算t+1时刻的p_gen系数
            p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))
            # predictions.shape=[8,50,30000]
            predictions.append(pred)
            # coverages.shape=[8,200,1]
            coverages.append(coverage_next)
            # attentions.shape=[8,200]
            attentions.append(atten_dist)
            # p_gens.shape=[8,50]
            p_gens.append(p_gen)
        # 计算一个batch最终的词汇分布
        final_dists = calc_final_dist(enc_extended_inp,
                                      predictions,
                                      attentions,
                                      p_gens,
                                      batch_oov_len,
                                      self.params.vocab_size,
                                      self.params.batch_size
                                      )
        if self.params.mode == "train":
            outputs = dict(logits=final_dists, dec_hidden=dec_hidden, attentions=attentions, coverages=coverages,
                           p_gens=p_gens)
        else:
            outputs = dict(logits=tf.stack(final_dists, 1),
                           dec_hidden=dec_hidden,
                           attentions=tf.stack(attentions, 1),
                           coverages=tf.stack(coverages, 1),
                           p_gens=tf.stack(p_gens, 1))

        return outputs
