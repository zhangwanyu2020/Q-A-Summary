#seq2seq
import time

class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super(SequenceToSequence, self).__init__()
        self.embedding_matrix = load_word2vec(params)
        self.params = params
        self.encoder = Encoder(params.vocab_size,
                                           params.embed_size,
                                           params.enc_units,
                                           params.batch_size,
                                           self.embedding_matrix)
        self.attention = BahdanauAttention(params.attn_units)
        self.decoder = Decoder(params.vocab_size,
                                           params.embed_size,
                                           params.dec_units,
                                           params.batch_size,
                                           self.embedding_matrix)

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        # [batch_sz, max_train_x, enc_units], [batch_sz, enc_units]
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_hidden

    def call(self, enc_output, dec_inp, dec_hidden, dec_tar):
        predictions = []
        attentions = []
        context_vector, _ = self.attention(dec_hidden,  # shape=(16, 256)
                                           enc_output)  # shape=(16, 200, 256)

        for t in range(dec_tar.shape[1]):  # 50
            # Teachering Forcing
            _, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1),
                                               dec_hidden,
                                               enc_output,
                                               context_vector)
            context_vector, attn_dist = self.attention(dec_hidden, enc_output)

            predictions.append(pred)
            attentions.append(attn_dist)

        return tf.stack(predictions, 1), dec_hidden