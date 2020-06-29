# train_eval_test
import pandas as pd


def train(params):
    assert params.mode.lower() == "train", "change training mode to 'train'"

    vocab = Vocab(params.vocab_path, params.vocab_size)
    print('true vocab is ', vocab)

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Building the model ...")
    model = PGN(params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params.pgn_model_dir)
    ckpt = tf.train.Checkpoint(PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    print("Starting the training ...")
    train_model(model, b, params, ckpt, ckpt_manager)


def test(params):
    assert params.mode.lower() == "test", "change training mode to 'test' or 'eval'"
    # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    print("Building the model ...")
    model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params.vocab_path, params.vocab_size)

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Creating the checkpoint manager")
    checkpoint_dir = "{}/checkpoint".format(params.pgn_model_dir)
    ckpt = tf.train.Checkpoint(PGN=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Model restored")

    if params.greedy_decode:
        predict_result(model, params, vocab)


def predict_result(model, params, vocab):
    dataset = batcher(vocab, params)
    # 预测结果
    results = greedy_decode(model, dataset, vocab, params)
    results = list(map(lambda x: x.replace(" ", ""), results))
    # print(results)
    # 保存结果
    save_predict_result(results, params)

    return results


def save_predict_result(results, params):
    # 读取结果
    test_df = pd.read_csv(params.test_x_dir)
    # 填充结果
    test_df['Prediction'] = results[:2000]
    # 　提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果
    test_df.to_csv(params.result_save_path, index=None, sep=',')
