import numpy as np

class TextLoader(object):
    def __init__(self, dataSet,batch_size,max_sent,data_mode):
        self.data = dataSet
        self.batch_size = batch_size
        self.max_sent = max_sent
        self.data_mode = data_mode
        self.shuff()

    def shuff(self):
        self.num_batches = int(len(self.data) // self.batch_size)
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'
        #np.random.shuffle(self.data)


    def next_batch(self,k):
        x = []
        y = []
        z = []
        # 这里是按顺序采集每一个batch的对应数据
        for i in range(self.batch_size):
            tmp = list(self.data)[k*self.batch_size + i][:3]
            x.append(tmp)
            if self.data_mode == 'train' or self.data_mode == 'eval':
                y_ = list(self.data)[k*self.batch_size + i][3]
                y.append(y_)
                z_ = list(self.data)[k*self.batch_size + i][4]
                z.append(z_)
            elif self.data_mode == 'test':
                z_ = list(self.data)[k * self.batch_size + i][3]
                z.append(z_)
        x = np.array(x)
        if self.data_mode == 'train' or self.data_mode == 'eval':
            y = np.array(y)
            y = y.reshape(self.batch_size, self.max_sent, 1)
            return x,y,z
        elif self.data_mode == 'test':
            z = np.array(z)
            return x,z










