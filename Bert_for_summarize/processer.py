import json
import os
import sys
import tensorflow as tf
import csv
csv.field_size_limit(sys.maxsize)
import bert_master.tokenization as tokenization

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

      When running eval/predict on the TPU, we need to pad the number of examples
      to be a multiple of the batch size, because the TPU requires a fixed batch
      size. The alternative is to drop the last batch, which is bad because it means
      the entire output data won't be generated.

      We use this class instead of `None` because treating `None` as padding
      battches could cause silent errors.
      """

class InputExample(object):
  def __init__(self, guid, text_a, text_b=None, label=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class MyProcessor():
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self,data_dir):
      """See base class."""
      return self._create_examples(
          self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    label_list = []
    with open('label_id.json', encoding='utf-8') as f:
        line = f.readline()
        d = json.loads(line)
        for values in d.values():
            label_list.append(values)
    f.close()
    return  label_list

  def _read_json(self,input_file):
      lines = []
      with open(input_file, encoding='utf-8') as f:
          line = f.readline()
          item = json.loads(line)
          lines.extend(item)
      return lines

  def _create_examples(self,lines, set_type):
      """Creates examples for the training and dev sets."""
      examples = []
      for (i, line) in enumerate(lines):
          guid = "%s-%s" % (set_type, i)
          # line 是一个字典
          text_a = [tokenization.convert_to_unicode(sent) for sent in line['src']]
          # text_a = tokenization.convert_to_unicode(line[1])
          if set_type=='train' or set_type=='dev':
              label = line['ids']
              examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
          else:
              examples.append(InputExample(guid=guid, text_a=text_a, text_b=None))
      # 返回的text_a label 都是list
      return examples

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               tokens,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.tokens = tokens
    self.is_real_example = is_real_example

class InputFeaturesTest(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               tokens,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.tokens = tokens
    self.is_real_example = is_real_example

def file_based_convert_examples_to_features(examples, max_seq_length, max_sent_length, tokenizer,data_mode='train'):
  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 19999 == 0:
      print("Writing example %d of %d" % (ex_index, len(examples)))
    # 这一步输入的是[文本 文本 ...文本]，输出是[[cls] id id ...id [seg] id id...[seg]]以及mask\pad\truncate
    feature = convert_single_example(ex_index, example,max_seq_length, max_sent_length, tokenizer,data_mode)
    temp = []
    temp.append(feature.input_ids)
    temp.append(feature.input_mask)
    temp.append(feature.segment_ids)
    if data_mode == 'train' or data_mode == 'dev':
        temp.append(feature.label_id)
    temp.append(feature.tokens)
    features.append(temp)
  print('features length : ',len(features))
  # features is a list ,包含所有符合bert格式的样本
  return features

def convert_single_example(ex_index, example, max_seq_length, max_sent_length, tokenizer,data_mode='train'):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * (max_seq_length+2)*max_sent_length,
        input_mask=[0] * (max_seq_length+2)*max_sent_length,
        segment_ids=[0] * (max_seq_length+2)*max_sent_length,
        label_id=[0]*32,
        is_real_example=False)

  # 这一步就是做Word pieces
  tokens_a_list_1 = [tokenizer.tokenize(line) for line in example.text_a]
  # 如果句子个数超出，做截断
  tokens_a_list = tokens_a_list_1[:max_sent_length]
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # 这一步是在做bert得输入格式：[]cls] token token ... token[seg] token token ...token [seg]
    tokens = []
    segment_ids = []
    input_mask = []
    i = 0
    for line in tokens_a_list:
        tokens.append("[CLS]")
        input_mask.append(1)
        if i%2==0:
            segment_ids.append(0)
        else:
            segment_ids.append(1)
        # 这一步是做截断,Account for [CLS] and [SEP] with "- 2"
        if len(line) > max_seq_length:
            line = line[0:max_seq_length]
            input_mask.extend([1] * len(line))
        else:
            line.extend(['0'] * (max_seq_length - len(line)))
            input_mask.extend([0] * (max_seq_length - len(line)))
        tokens.extend(line)
        if i%2==0:
            segment_ids.extend([0]*len(line))
        else:
            segment_ids.extend([1] * len(line))
        tokens.append("[SEP]")
        input_mask.append(1)
        if i % 2 == 0:
            segment_ids.append(0)
        else:
            segment_ids.append(0)
        i += 1
    m = i
    # 如果句子个数不够，做补全
    if m<max_sent_length:
        for j in range(max_sent_length-m):
            tokens.append("[CLS]")
            tokens.extend(['0']*max_seq_length)
            tokens.append("[SEP]")
            if i % 2 == 0:
                segment_ids.extend([0]*(max_seq_length+2))
            else:
                segment_ids.extend([1]*(max_seq_length+2))
            i += 1

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask_2 = [0] * (len(input_ids)-len(input_mask))
  input_mask.extend(input_mask_2)
  # 核对长度是否一致
  assert len(input_ids) == (max_seq_length+2)*max_sent_length
  assert len(input_mask) == (max_seq_length+2)*max_sent_length
  assert len(segment_ids) == (max_seq_length+2)*max_sent_length

  # 取出当前样本label对应的id,label_id is a list
  if data_mode=='train' or data_mode=='dev':
      label_list = [i for i in example.label if i<max_sent_length]
      label_id =[0]*max_sent_length
      for j in label_list:
          label_id[j] = 1


  # if ex_index < 5:
  #   tf.logging.info("*** Example ***")
  #   tf.logging.info("guid: %s" % (example.guid))
  #   tf.logging.info("tokens: %s" % " ".join(
  #       [tokenization.printable_text(x) for x in tokens]))
  #   tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
  #   tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
  #   tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
  #   tf.logging.info("label: {0} (id = {1})".format(example.label, label_id))
  # 返回当前样本的feature
  if data_mode=='train' or data_mode=='dev':
      feature = InputFeatures(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          label_id=label_id,
          tokens=tokens_a_list_1,
          is_real_example=True)
  else:
      feature = InputFeaturesTest(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          tokens=tokens_a_list_1,
          is_real_example=True)
  return feature

def process_function(data_dir,vocab_file_path,do_train,do_eval,do_test,max_seq_length,max_sent_length,batch_size):
    train_input = None
    eval_input = None
    test_input = None
    processor = MyProcessor()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path, do_lower_case=True)
    if do_train:
        # train_examples is a list, 每个元素为InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
        train_examples = processor.get_train_examples(data_dir)
        # InputExample ---> features
        train_input = file_based_convert_examples_to_features(
            train_examples, max_seq_length, max_sent_length, tokenizer,data_mode='train')
        print('***start to training**')
        print('  Number training examples  %d',len(train_examples))
        print('   Batch size %d',batch_size)
    if do_eval:
        eval_examples = processor.get_dev_examples(data_dir)
        eval_input = file_based_convert_examples_to_features(
            eval_examples, max_seq_length, max_sent_length, tokenizer,data_mode='dev')
        print('***start to validation**')
        print('  Number validate examples  %d', len(eval_examples))
        print('   Batch size %d', batch_size)
    if do_test:
        test_examples = processor.get_test_examples(data_dir)
        test_input = file_based_convert_examples_to_features(
        test_examples, max_seq_length, max_sent_length, tokenizer,data_mode='test')
        print('***start to testing**')
        print('  Number test examples  %d', len(test_examples))
        print('   Batch size %d', batch_size)
    # 返回的 train_input,eval_input,predict_input 都是list
    return train_input,eval_input,test_input
