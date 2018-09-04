'''
# @Time    : 18-8-28 下午11:48
# @Author  : ShengZ
# @FileName: run.py.py
# @Software: PyCharm
# @Github  : https://github.com/ZZshengyeah
'''
import sys
import tensorflow as tf
from model import *
from preprocess import inputs, write_results
import time

def train(sess,to_train,to_test):
    n = 1
    best = .0
    best_step = n

    start = time.time()
    orig_begin_time = start

    fetches = [to_train.train_steps, to_train.loss, to_train.accuracy]

    while True:
        try:
            _, loss , acc = sess.run(fetches)
            epoch = n // 50
            if n % 50 == 0:
                end = time.time()
                duration = end - start
                start = end
                test_acc = sess.run(to_test.accuracy)
                if best < test_acc:
                    best = test_acc
                    best_step = n
                print("Epoch %d, loss %.4f, acc %.4f test_acc %.4f, time %.2f" %
                                  (epoch, loss, acc, test_acc, duration))
            n += 1

        except tf.errors.OutOfRangeError:
            break

    duration = time.time() - orig_begin_time
    duration /= 3600
    to_train.save(sess, best_step)
    print('Done training, best_step: %d, best_acc: %.4f' % (best_step, best))
    print('duration: %.2f hours' % duration)

def test(sess, to_test):
    to_test.restore(sess)
    fetches = [to_test.accuracy, to_test.prediction]
    acc, pred = sess.run(fetches)
    print('accuracy: %.4f' % acc)
    write_results(pred,FLAGS.relations_file,FLAGS.results_file)

def main(_):
    goal = sys.argv[1].split('--')[1]
    flags = tf.app.flags
    flags.DEFINE_integer('max_len', 96, 'max length of sentences')
    flags.DEFINE_integer('word_dim', 50, 'word dimention')
    flags.DEFINE_integer('num_epoches', 200, 'train epoches')
    flags.DEFINE_integer('batch_size', 100, 'batch size')
    flags.DEFINE_integer('conv_deep', 100, 'conv deepth')
    flags.DEFINE_integer('fc_node', 200, 'full connect node')
    flags.DEFINE_integer("pos_num", 123, "number of position feature")
    flags.DEFINE_integer("pos_dim", 10, "position embedding size")
    flags.DEFINE_integer("num_relations", 19, "feature numbers")
    flags.DEFINE_float("keep_prob", 0.5, "dropout keep probability")
    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float("regularization_rate", 0.01, "regularization rate")
    flags.DEFINE_string("train_file", "data/train.cln",
                        "original training file")
    flags.DEFINE_string("test_file", "data/test.cln",
                        "original test file")
    flags.DEFINE_string("vocab_file", "data/vocab.txt",
                        "vocab of train and test data")
    flags.DEFINE_string('pretrain_embed_file', 'data/embed50.senna.npy',
                        '50 dimention glove vector')
    flags.DEFINE_string('pretrain_vocab_file', 'data/glove_words.txt', 'vocab')
    flags.DEFINE_string('trimed_embed_file', 'data/trimed_embed.npy',
                        'trim embedding')
    flags.DEFINE_string('train_record', 'data/train.tfrecord', 'train tfrecord')
    flags.DEFINE_string('test_record', 'data/test.tfrecord', 'test tfrecord')
    flags.DEFINE_string("relations_file", "data/relations.txt", "relations file")
    flags.DEFINE_string("results_file", "data/results.txt", "predicted results file")
    flags.DEFINE_string("logdir", "save_model/", "where to save the model")
    FLAGS = tf.app.flags.FLAGS

    with tf.Graph().as_default():
        train_data, test_data, word_embed = inputs()
        to_train, to_test = build_train_test_model(word_embed,train_data,test_data)
        to_train.set_saver('cnn-%d-%d' %(FLAGS.num_epoches, FLAGS.word_dim))
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            if goal == 'train':
                train(sess,to_train,to_test)
            elif goal == 'test':
                test(sess,to_test)

if __name__ == '__main__':
    tf.app.run()
