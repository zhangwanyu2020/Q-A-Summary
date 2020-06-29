# test_helper
from tqdm import tqdm
import math


def greedy_decode(model, dataset, vocab, params):
    # 存储结果
    batch_size = params.batch_size
    results = []

    sample_size = 2000
    # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
    # steps_epoch 这里+1就报错，原因是data_iter=iter(dataset)放在循环外面了，如果放在里面，又会造成enc_data一直在遍历第一个batch
    steps_epoch = sample_size // batch_size
    data_iter = iter(dataset)
    for i in tqdm(range(steps_epoch)):
        enc_data, _ = next(data_iter)
        # print(enc_data,'*******************')
        results += batch_greedy_decode(model, enc_data, vocab, params)
    return results


def batch_greedy_decode(model, enc_data, vocab, params):
    # 判断输入长度
    batch_data = enc_data["enc_input"]
    batch_size = enc_data["enc_input"].shape[0]
    # 开辟结果存储list
    predicts = [''] * batch_size

    inputs = tf.convert_to_tensor(batch_data)
    enc_output, enc_hidden = model.call_encoder(inputs)

    dec_hidden = enc_hidden
    # 2:[vocab.word_to_id(START_DECODING)]
    dec_input = tf.constant([2] * batch_size)
    dec_input = tf.expand_dims(dec_input, axis=1)
    enc_padding_mask = enc_data["sample_encoder_pad_mask"]

    context_vector, attn_dist, coverage_next = model.attention(dec_hidden, enc_output, enc_padding_mask,
                                                               use_coverage=True, prev_coverage=None)
    for t in range(params.max_dec_len):
        # 单步预测
        _, pred, dec_hidden = model.decoder(dec_input, context_vector)

        context_vector, attn_dist, coverage_next = model.attention(dec_hidden, enc_output, enc_padding_mask,
                                                                   use_coverage=True, prev_coverage=coverage_next)
        predicted_ids = tf.argmax(pred, axis=1).numpy()

        for index, predicted_id in enumerate(predicted_ids):
            predicts[index] += vocab.id_to_word(predicted_id) + ' '

        # using teacher forcing
        dec_input = tf.expand_dims(predicted_ids, 1)

    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断vocab.word_to_id('[STOP]')
        if '[STOP]' in predict:
            # 截断stop
            predict = predict[:predict.index('[STOP]')]
        # 保存结果
        results.append(predict)
    # print(results)
    return results