import numpy as np
import tensorflow as tf


def bool_to_value(pred_bool):
    predictions = []
    for i in range(len(pred_bool)):
        p_sigle = [0] * 32
        for j in range(len(pred_bool[i])):
            if pred_bool[i][j]:
                p_sigle[j] = 1
            else:
                p_sigle[j] = 0
        predictions.append(p_sigle)
    return predictions

def id_to_labeltext(id_list,token_x):
    abstracts = []
    for i,x in enumerate(token_x):
        abstract = ''
        ids = []
        for m in range(len(id_list[i])):
            if id_list[i][m] == 1:
                ids.append(m)
        for j in ids:
            # 预测id 可能超出了原始句子的范围
            if j>len(x)-1:continue
            else:
                s_list = x[j]
                s = ''.join(s_list)
                s = s.replace('0','')
                s = s.replace('"','')
                abstract = abstract+'，'+s
                abstract = abstract.strip('，')
        abstracts.append(abstract)
    print(i)
    return abstracts

def prob_to_bool(probilities):
    prob_cut = []
    predicts_bool = []
    for i in range(len(probilities)):
        prob_arr = np.array(probilities[i])
        b = np.percentile(prob_arr, [75])
        prob_list = prob_arr.tolist()
        _bool = []
        for j in range(len(prob_list)):
            if prob_list[j]<b[0]:_bool.append(False)
            else:_bool.append(True)
        prob_cut.append(b)
        predicts_bool.append(_bool)
    return predicts_bool


