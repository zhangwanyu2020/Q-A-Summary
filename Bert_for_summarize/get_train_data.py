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


def get_ids(src: list, tgt: list):
    d = {}
    for word in tgt:
        for i in range(len(src)):
            if word in src[i] and i not in d.keys():
                d[i] = 1
            elif word in src[i] and i in d.keys():
                d[i] += 1
    d_sort = sorted(d.items(), key=lambda item: item[1], reverse=True)

    max_len = 4
    ids = []
    for item in d_sort[:max_len]:
        ids.append(item[0])

    ids = sort_nums(ids)
    return ids


stopwords = read_file('/Users/zhangwanyu/stop_words.txt')

save_path = "/Users/zhangwanyu/Desktop/test_data/data_bert_sum/"

data_mode = 'train'

with open("/Users/zhangwanyu/AutoMaster_TrainSet.csv", "r") as csv_file:
    formatted_dic = {}
    output_data = []
    csv_reader = csv.reader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print("column names:", ", ".join(row))
            line_count += 1
            continue
        elif len(row) != 6:
            continue
        else:
            brand, model, question, dialogue, report = (row[1], row[2], row[3], row[4], row[5])
            if report == "随时联系":
                continue
            dialogue = " ".join(dialogue.split())
            if "|" in dialogue:
                splitted_dialogue = [txt.replace("技师说：", "").replace("车主说：", "") for txt in
                                     re.split(r'[，。\ ! ? |]', dialogue)]

            elif "技师说" in dialogue:
                splitted_dialogue = [txt.replace("技师说：", "") for txt in re.split(r'[，。\ ! ? |]', dialogue)]

            else:
                splitted_dialogue = re.split(r'[，。\ ! ? |]', dialogue)
            splitted_dialogue = [txt for txt in splitted_dialogue if
                                 "[语音]" not in txt and "[视频]" not in txt and "[图片]" not in txt and txt]

            if len("".join(splitted_dialogue)) < 1:
                continue
            question = question.split('，')
            processed_text = splitted_dialogue

            report = [word for word in jieba.lcut(report) if word not in stopwords]
            formatted_dic['ids'] = get_ids(processed_text, report)
            formatted_dic['src'] = processed_text

            output_data.append(formatted_dic.copy())

            if len(output_data) > 50000-1:
                pt_file = "{:s}{}.json".format(save_path,data_mode)
                with open(pt_file, "w") as save:
                    save.write(json.dumps(output_data, ensure_ascii=False))
                    print('{0}_length_{1}'.format(data_mode, len(output_data)))
                    output_data = []
