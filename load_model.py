'''
# @Time    : 18-8-28 下午11:08
# @Author  : ShengZ
# @FileName: load_model.py.py
# @Software: PyCharm
# @Github  : https://github.com/ZZshengyeah
'''

import tensorflow as tf
import  os
FLAGS = tf.app.flags.FLAGS

class BaseModel(object):
    @classmethod
    def set_saver(cls,save_dir):
        cls.saver = tf.train.Saver()
        cls.save_dir = os.path.join(FLAGS.logdir,save_dir)
        cls.save_path = os.path.join(cls.save_dir,'model.ckpt')

    @classmethod
    def restore(cls,session):
        ckpt = tf.train.get_checkpoint_state(cls.save_dir)
        cls.saver.restore(session,ckpt.model_checkpoint_path)

    @classmethod
    def save(cls,session,global_step):
        cls.saver.save(session,cls.save_path,global_step)