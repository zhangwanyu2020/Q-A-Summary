import csv
import json
import re
import itertools
import jieba


def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip('\n').split()
            lines.extend(line)
    return lines


def sort_nums(nums: list):
    for i in range(len(nums) - 1, 0, -1):
        for j in range(i):
            if nums[i] < nums[j]:
                nums[j], nums[i] = nums[i], nums[j]
    return nums


stopwords = read_file('/Users/zhangwanyu/stop_words.txt')

save_path = "/Users/zhangwanyu/Desktop/test_data/data_bert_sum/"

data_mode = 'test'

with open("/Users/zhangwanyu/AutoMaster_TestSet.csv", "r") as csv_file:
    formatted_dic = {}
    output_data = []
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print("column names:", ", ".join(row))
            line_count += 1
            continue
        elif len(row) != 5:
            continue
        else:
            qid, brand, model, question, dialogue = (row[0], row[1], row[2], row[3], row[4])
            dialogue = " ".join(dialogue.split())
            if "|" in dialogue:
                splitted_dialogue = [txt.replace("技师说：", "").replace("车主说：", "") for txt in
                                     re.split(r'[，。\ ! ? ？！|]', dialogue)]

            elif "技师说" in dialogue:
                splitted_dialogue = [txt.replace("技师说：", "") for txt in re.split(r'[，。\ ! ? ？！|]', dialogue)]

            else:
                splitted_dialogue = re.split(r'[，。\ ! ? ？| ！]', dialogue)
            splitted_dialogue = [txt for txt in splitted_dialogue if
                                 "[语音]" not in txt and "[视频]" not in txt and "[图片]" not in txt and txt]
            # question 可以放入processed_text，看训练效果
            question = question.split('，')
            processed_text = splitted_dialogue
            formatted_dic['src'] = processed_text
            formatted_dic['qid'] = qid

            output_data.append(formatted_dic.copy())
    print(len(output_data))
    pt_file = "{:s}test.json".format(save_path)
    with open(pt_file, "w") as save:
        save.write(json.dumps(output_data, ensure_ascii=False))


