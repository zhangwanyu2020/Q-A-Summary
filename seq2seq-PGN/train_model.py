# train_helper
import time

START_DECODING = '[START]'


def train_model(model, dataset, params, ckpt, ckpt_manager):
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params.learning_rate)

    # @tf.function()
    def train_step(enc_inp, enc_extended_inp, dec_inp, dec_tar, batch_oov_len, enc_padding_mask, padding_mask):
        # loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = model.call_encoder(enc_inp)
            dec_hidden = enc_hidden
            # start index
            outputs = model(enc_output,  # shape=(8, 200, 256)
                            dec_hidden,  # shape=(8, 256)
                            enc_inp,  # shape=(8, 200)
                            enc_extended_inp,  # shape=(8, 200)
                            dec_inp,  # shape=(8, 50)
                            batch_oov_len,  # shape=()
                            enc_padding_mask,  # shape=(8, 200)
                            params.is_coverage,
                            pre_coverage=None
                            )
            loss = loss_function(dec_tar,
                                 outputs,
                                 padding_mask,
                                 params.cov_loss_wt,
                                 params.is_coverage
                                 )
        variables = model.encoder.trainable_variables + \
                    model.attention.trainable_variables + \
                    model.decoder.trainable_variables + \
                    model.pointer.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    best_loss = 20
    epochs = params.epochs
    for epoch in range(epochs):
        t0 = time.time()
        step = 0
        total_loss = 0

        for batch in dataset:
            loss = train_step(batch[0]["enc_input"],  # shape=(8, 200)
                              batch[0]["extended_enc_input"],  # shape=(8, 200)
                              batch[1]["dec_input"],  # shape=(8, 50)
                              batch[1]["dec_target"],  # shape=(8, 50)
                              batch[0]["max_oov_len"],
                              batch[0]["sample_encoder_pad_mask"],  # shape=(8, 200)
                              batch[1]["sample_decoder_pad_mask"])  # shape=(8, 50)

            step += 1
            total_loss += loss
            if step % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, total_loss / step))

        if epoch % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            best_loss = total_loss / step
            print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))
            # lr = params.learning_rate * np.power(0.8,epoch+1)
            # optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=lr)
            print("learning_rate=", optimizer.get_config()["learning_rate"])



