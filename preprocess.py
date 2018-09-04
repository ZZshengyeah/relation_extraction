'''
# @Time    : 18-8-25 下午2:49
# @Author  : ShengZ
# @FileName: preprocess.py.py
# @Software: PyCharm
# @Github  : https://github.com/ZZshengyeah
'''
import tensorflow as tf
import gensim
import numpy as np
import os
from collections import namedtuple

'''
下载好GloVe词向量文件后,可利用以下两个函数将文件转化成所需要的格式
def process_vector_file():
    with open('./data/glove.6B.50d.txt','r') as f:
        lines = f.readlines()
        print(lines[0])
        with open('./data/glove_vector_50.txt','a') as ff:
            ff.write('400001 50\n')
            for line in lines:
                ff.write(line)
                ff.write('\n')
def convert_google_embedding(glove_file,new_words_file,new_embed_npfile):
    model = gensim.models.KeyedVectors.load_word2vec_format(glove_file,binary=False)
    print('load finished.')

    embed = []
    with open(new_words_file, 'w') as f:
        for i, w in enumerate(model.index2word):
            if i % 1000000 == 0:
                print(i)
            embed.append(model[w])
            f.write('%s\n' % w)

    embed = np.asarray(embed)
    print('converted to np array')
    embed = embed.astype(np.float32)
    print('converted to float32')
    np.save(new_embed_npfile, embed)
convert_google_embedding('./data/glove_vector_50.txt','glove_words.txt','glove_vector_50.npy')
'''

PAD_WORD = "<pad>"
FLAGS = tf.app.flags.FLAGS

Raw_Example = namedtuple('Raw_Example','label entity1 entity2 sentence')
PositionPair = namedtuple('PosPair','first last')

def load_raw_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            words = line.strip('\n').split(' ')
            sent = words[5:]
            n = len(sent)
            if FLAGS.max_len < n:
                FLAGS.max_len = n
            label = int(words[0])
            entity1 = PositionPair(int(words[1]),int(words[2]))
            entity2 = PositionPair(int(words[3]),int(words[4]))
            raw_example = Raw_Example(label,entity1,entity2,sent)
            data.append(raw_example)
    return data
def maybe_build_vocab(raw_train_data,raw_test_data,vocab_file):
    if not os.path.exists(vocab_file):
        vocab = set()
        for example in raw_train_data + raw_test_data:
            for word in example.sentence:
                vocab.add(word)
        with open(vocab_file,'a') as f:
            for word in sorted(list(vocab)):
                f.write('%s\n' %word)
            f.write('%s\n' %PAD_WORD)

def load_vocab(vocab_file):
    '''
    :param vocab_file:
    :return: vocab list
    '''
    vocab = []
    with open(vocab_file,'r') as f:
        for line in f:
            word = line.strip()
            vocab.append(word)
    return  vocab

def load_embedding(embed_file,vocab_file):
    embed = np.load(embed_file)
    word2id = {}
    words = load_vocab(vocab_file)
    for id,w in enumerate(words):
        word2id[w] = id
    return embed,word2id

def maybe_trim_embedding(vocab_file,
                         pretrain_embed_file,
                         pretrain_vocab_file,
                         trimed_embed_file):
    if not os.path.exists(trimed_embed_file):
        pretrain_embed,pretrain_word2id = load_embedding(pretrain_embed_file,
                                                         pretrain_vocab_file)
        new_word_embed = []
        vocab_list = load_vocab(vocab_file)
        for word in vocab_list:
            if word in pretrain_word2id:
                id = pretrain_word2id[word]
                new_word_embed.append(pretrain_embed[id])
            else:
                vec = np.random.normal(0,0.1,[50])
                new_word_embed.append(vec)
        pad_id = -1
        new_word_embed[pad_id] = np.zeros([FLAGS.word_dim])
        new_word_embed = np.asarray(new_word_embed)
        np.save(trimed_embed_file,new_word_embed.astype(np.float32))
    new_word_embed, vocab2id = load_embedding(trimed_embed_file,vocab_file)
    return new_word_embed, vocab2id


def map_words_to_ids(raw_data,word2id):
    '''
    此处为论文中word_feature实现部分
    '''
    pad_id = word2id[PAD_WORD]
    for k,raw_example in enumerate(raw_data):
        new_sent = []
        for id, word in enumerate(raw_example.sentence):
            if id >= 1:
                left_word = raw_example.sentence[id-1]
                new_sent.append(word2id[left_word])
            else:
                new_sent.append(word2id[PAD_WORD])

            new_sent.append(word2id[word])

            if id >= len(raw_example.sentence) - 1:
                new_sent.append(word2id[PAD_WORD])
            else:
                right_word = raw_example.sentence[id+1]
                new_sent.append(word2id[right_word])
        raw_example = raw_example._replace(sentence=new_sent)
        pad_n = 3*FLAGS.max_len - len(raw_example.sentence)
        raw_example.sentence.extend(pad_n*[pad_id])
        raw_data[k] = raw_example

'''  
def map_words_to_ids(raw_data, word2id):
  """
  inplace convert sentence from a list of words to a list of ids
  Args:
    raw_data: a list of Raw_Example
    word2id: dict, {word: id, ...}
  """
  pad_id = word2id[PAD_WORD]
  for raw_example in raw_data:
    for idx, word in enumerate(raw_example.sentence):
      raw_example.sentence[idx] = word2id[word]
    #此时sentence已经变成数字编码
    # pad the sentence to FLAGS.max_len
    pad_n = FLAGS.max_len - len(raw_example.sentence)
    raw_example.sentence.extend(pad_n*[pad_id])
'''


def lexical_feature(raw_example):
    def entity_context(e_id,sent):
        context = []
        context.append(sent[e_id])
        if e_id >= 1:
            context.append(sent[e_id-1])
        else:
            context.append(sent[e_id])

        if e_id < len(sent) - 1:
            context.append(sent[e_id+1])
        else:
            context.append(sent[e_id])
        return context
    e1_id = raw_example.entity1.first
    e2_id = raw_example.entity2.first
    context1 = entity_context(e1_id, raw_example.sentence)
    context2 = entity_context(e2_id, raw_example.sentence)

    lexical = context1 + context2
    return lexical


def position_feature(raw_example):
    def relative_distance(n):
        if n < -60:
            return 0
        elif n >= -60 and n <= 60:
            return n+61
        return 122
    e1_id = raw_example.entity1.first
    e2_id = raw_example.entity2.first
    position1 = []
    position2 = []
    length = len(raw_example.sentence)
    for i in range(length):
        position1.append(relative_distance(i-e1_id))
        position2.append(relative_distance(i-e2_id))
    return position1, position2

def build_sequence_example(raw_example):
    ex = tf.train.SequenceExample()

    lexical = lexical_feature(raw_example)
    ex.context.feature['lexical'].int64_list.value.extend(lexical)

    label = raw_example.label
    ex.context.feature['label'].int64_list.value.append(label)

    for word_id in raw_example.sentence:
        word = ex.feature_lists.feature_list['sentence'].feature.add()
        word.int64_list.value.append(word_id)

    position1, position2 = position_feature(raw_example)
    for pos_val in position1:
        pos = ex.feature_lists.feature_list['position1'].feature.add()
        pos.int64_list.value.append(pos_val)
    for pos_val in position2:
        pos = ex.feature_lists.feature_list['position2'].feature.add()
        pos.int64_list.value.append(pos_val)

    return ex

def maybe_write_tfrecord(raw_data,tfrecord_path):
    if not os.path.exists(tfrecord_path):
        writer = tf.python_io.TFRecordWriter(tfrecord_path)
        for raw_example in raw_data:
            example = build_sequence_example(raw_example)
            writer.write(example.SerializeToString())
        writer.close()

def parse_tfexample(serialized_example):
    context_features = {'lexical':tf.FixedLenFeature([6],tf.int64),
                        'label' :tf.FixedLenFeature([],tf.int64),
                        }
    sequence_features = {
                            'sentence': tf.FixedLenSequenceFeature([],tf.int64),
                            'position1': tf.FixedLenSequenceFeature([],tf.int64),
                            'position2': tf.FixedLenSequenceFeature([],tf.int64),
                            }

    context_dict, sequence_dict = tf.parse_single_sequence_example(
                                        serialized_example,
                                        context_features=context_features,
                                        sequence_features=sequence_features)
    sentence = sequence_dict['sentence']
    position1 = sequence_dict['position1']
    position2 = sequence_dict['position2']

    lexical = context_dict['lexical']
    label = context_dict['label']

    return lexical, label, sentence, position1, position2

def read_tfrecord_to_batch(tfrecord_path,
                           epoch,batch_size,
                           shuffle=True):
    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset([tfrecord_path])
        dataset = dataset.map(parse_tfexample)
        dataset = dataset.repeat(epoch)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()

        return batch

def inputs():
    raw_train_data = load_raw_data(FLAGS.train_file)
    raw_test_data = load_raw_data(FLAGS.test_file)

    maybe_build_vocab(raw_train_data,raw_test_data,FLAGS.vocab_file)
    word_embed, vocab2id = maybe_trim_embedding(FLAGS.vocab_file,
                                                FLAGS.pretrain_embed_file,
                                                FLAGS.pretrain_vocab_file,
                                                FLAGS.trimed_embed_file)

    map_words_to_ids(raw_train_data,vocab2id)
    map_words_to_ids(raw_test_data,vocab2id)

    train_record = FLAGS.train_record
    test_record = FLAGS.test_record

    maybe_write_tfrecord(raw_train_data,train_record)
    maybe_write_tfrecord(raw_test_data,test_record)

    pad_value = vocab2id[PAD_WORD]
    train_data = read_tfrecord_to_batch(train_record,
                                        FLAGS.num_epoches,
                                        FLAGS.batch_size,
                                        shuffle=True)
    test_data = read_tfrecord_to_batch(test_record,
                                       FLAGS.num_epoches,
                                       2717,
                                       shuffle=False)
    print('train\n',train_data)
    print('test\n',test_data)

    return train_data, test_data, word_embed


def write_results(prediction,relations_file,results_file):
    relations = []
    with open(relations_file) as f:
        for line in f:
            segment = line.strip().split()
            relations.append(segment[1])
    start_no = 8001
    with open(results_file,'w') as f:
        for i, id in enumerate(prediction):
            rel = relations[id]
            f.write('%d\t%s\n' %(start_no+i,rel))
