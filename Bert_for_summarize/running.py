import os
import numpy as np
import tensorflow as tf
import pandas as pd
from processer import process_function
from modeling import Project_model
from data_loader import TextLoader
from utils import bool_to_value,id_to_labeltext,prob_to_bool

data_path = '/Users/zhangwanyu/Desktop/test_data/data_bert_sum'
bert_root = '/Users/zhangwanyu/Desktop/data_project_2/bert/bert_model_chinese'
bert_vocab_file = os.path.join(bert_root, 'vocab.txt')
model_save_path = '/Users/zhangwanyu/Desktop/test_data/data_bert_sum/model/'
epochs = 1
save_checkpoint_steps = 100
max_len = 10
max_sent = 32
batch_size = 8
lr = 1e-4
keep_prob = 0.8
mode = 'eval'

train_input,eval_input,test_input =process_function(data_path,bert_vocab_file,False,True,False,max_len,max_sent,batch_size)
model = Project_model(bert_root,data_path,model_save_path,batch_size,max_len,max_sent,lr,keep_prob,mode)


if mode == 'train':

    with tf.Session() as sess:
        # with tf.device('/gpu:0'):
        writer = tf.summary.FileWriter('./tf_log/', sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        data_loader = TextLoader(train_input,batch_size,max_sent,mode)
        for i in range(epochs):
            data_loader.shuff()
            for j in range(data_loader.num_batches):
                x_train,y_train,_= data_loader.next_batch(j)
                step, loss_= model.run_step(sess,x_train,y_train)
                saver.save(sess, save_path=model_save_path+'model.ckpt', global_step=save_checkpoint_steps)
                print('the epoch number is : %d the index of batch is :%d, the loss value is :%f'%(i, j, loss_))

if mode == 'eval' or mode == 'test':
    saver = tf.train.Saver()
    with tf.Session() as sess:
        model_path = model_save_path+'model.ckpt-100'
        saver.restore(sess, model_path)
        if mode == 'eval':
            ps,rs,token_x= model.evaluate(sess,eval_input)
            rs = np.concatenate((rs), axis=0)
        elif mode == 'test':
            ps, token_x = model.predict(sess, test_input)
        ps = np.concatenate((ps),axis=0)
        ps_bool = prob_to_bool(ps) #概率转bool值

        token_x = np.concatenate((token_x),axis=0)
        ps = bool_to_value(ps_bool)
        text_a = id_to_labeltext(ps,token_x)
        # print('***',token_x)
        print(len(text_a))

        # 将预测文本写入 csv
        df = pd.DataFrame()
        df['Prediction'] = text_a
        df.to_csv(data_path+'/results.csv',index=None)