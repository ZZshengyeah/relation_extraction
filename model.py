# -*- coding:utf-8 -*-
'''
# @Time    : 18-8-26 下午5:00
# @Author  : ShengZ
# @FileName: model.py.py
# @Software: PyCharm
# @Github  : https://github.com/ZZshengyeah
'''

from preprocess import *
from load_model import BaseModel
FLAGS = tf.app.flags.FLAGS

def full_connect_layer(name,input_tensor, fc_node, output_node, regularizer=False):
  with tf.variable_scope(name):
    loss_l2 = tf.constant(0, dtype=tf.float32)
    fc_weight = tf.get_variable('fc_weight',[fc_node,output_node],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc_biases = tf.get_variable('bias',[output_node],
                                initializer=tf.constant_initializer(0.1))
    fc_output = tf.nn.xw_plus_b(input_tensor, fc_weight, fc_biases)
    if regularizer:
        loss_l2 += tf.nn.l2_loss(fc_weight) + tf.nn.l2_loss(fc_biases)
    
    return fc_output, loss_l2

def cnn_forward(input_tensor, lexical, conv_deep):
    with tf.variable_scope('cnn_forward'):
        pool_outputs = []
        input_tensor = tf.expand_dims(input_tensor,axis=-1)
        input_width = input_tensor.shape.as_list()[2]
        for filter_size in [3,4,5]:
            with tf.variable_scope('conv-%s' % filter_size):
                conv_weights = tf.get_variable('conv_weight',
                                              [filter_size,input_width,1,conv_deep],
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv_biases = tf.get_variable('bias',[conv_deep],
                                              initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(input_tensor,
                                    conv_weights,
                                    strides=[1,1,input_width,1],
                                    padding='SAME')
                conv = tf.nn.relu(tf.add(conv,conv_biases))
                max_len = FLAGS.max_len
                pool = tf.nn.max_pool(conv,
                                      ksize=[1,3*max_len,1,1],  # 1,height,width,1
                                      strides=[1,3*max_len,1,1],
                                      padding='SAME')
                pool_outputs.append(pool)
        pools = tf.reshape(tf.concat(pool_outputs,axis=3),[-1, 3*conv_deep])
        feature = pools
        if lexical is not None:
            feature = tf.concat([lexical,feature],axis=1)
        return feature

class Model(BaseModel):
    def __init__(self,data,
              word_embed,word_dim,
              pos_num,pos_dim,num_relations,
              keep_prob,conv_deep,learning_rate,is_train=True):
        lexical, label, sentence, pos1, pos2 = data
        w_trainable = True if FLAGS.word_dim == 50 else False
        word_embed = tf.get_variable('word_embedding',initializer=word_embed,
                                     dtype=tf.float32,trainable=w_trainable)
        pos1_embed = tf.get_variable('pos1_embedding',shape=[pos_num,pos_dim],
                                     dtype=tf.float32)
        pos2_embed = tf.get_variable('pos2_embedding',shape=[pos_num,pos_dim],
                                     dtype=tf.float32)

        lexical = tf.nn.embedding_lookup(word_embed, lexical)

        lexical = tf.reshape(lexical, [-1, 6*word_dim])

        self.labels = tf.one_hot(label, num_relations)

        sentence = tf.nn.embedding_lookup(word_embed, sentence)
        pos1 = tf.nn.embedding_lookup(pos1_embed, pos1)
        pos2 = tf.nn.embedding_lookup(pos2_embed, pos2)
        outs = []
        sentence_level_features = tf.concat([sentence,pos1,pos2],axis = 2)
        if is_train:
            sentence_level_features = tf.nn.dropout(sentence_level_features,keep_prob)
           
        feature = cnn_forward(sentence_level_features,lexical,conv_deep)
        feature_size = feature.shape.as_list()[1]
        self.feature = feature

        if is_train:
            feature = tf.nn.dropout(feature,keep_prob)

        logits, loss_l2 = full_connect_layer('fc',feature, feature_size, num_relations,regularizer=True)

        prediction = tf.nn.softmax(logits=logits)
        prediction = tf.argmax(prediction,axis=1)

        accuracy = tf.equal(prediction, tf.argmax(self.labels, axis=1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

        losses = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))+\
                 FLAGS.regularization_rate*(loss_l2)

        self.logits = logits
        self.prediction = prediction
        self.accuracy = accuracy
        self.loss = losses

        if not is_train:
            return
        global_step = tf.Variable(0, trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(update_ops):
            self.train_steps = optimizer.minimize(self.loss,global_step)   
        self.global_step = global_step

def build_train_test_model(word_embed,train_data,test_data):
    with tf.name_scope('Train'):
        with tf.variable_scope('CNNModel',reuse=None):
            to_train = Model(train_data,word_embed,FLAGS.word_dim,
                             FLAGS.pos_num,FLAGS.pos_dim,FLAGS.num_relations,
                             FLAGS.keep_prob,FLAGS.conv_deep,
                             FLAGS.learning_rate,is_train=True)
    with tf.name_scope('Test'):
        with tf.variable_scope('CNNModel',reuse=True):
            to_test = Model(test_data,word_embed,FLAGS.word_dim,
                             FLAGS.pos_num,FLAGS.pos_dim,FLAGS.num_relations,
                             1.0,FLAGS.conv_deep,
                             FLAGS.learning_rate,is_train=False)
    return to_train,to_test
