import tensorflow as tf
import pickle
import os
import json


class load_model(object):

    def __init__(self, sess, MODEL, CHECKPOINT=None):

        self._sess = sess
        self.MODEL_PATH = MODEL
        self.CHECKPOINT_PATH = os.path.join(self.MODEL_PATH, 'checkpoints')
        ckpt = tf.train.latest_checkpoint(self.CHECKPOINT_PATH)
        self._saver  = tf.compat.v1.train.import_meta_graph(os.path.join(ckpt+'.meta'))
        self._graph = tf.compat.v1.get_default_graph()
        self.setup = pickle.load(open(self.MODEL_PATH + '/setup.pkl', 'rb'))
        self.load_ckpt(CHECKPOINT)
        self.tensors = self._load_tensors(self.setup['tensors'])
        
    def load_ckpt(self, ckpt=None):
        # Load ckpt
        if ckpt is None:
            ckpt_path = tf.train.latest_checkpoint(self.CHECKPOINT_PATH)
        else:
            ckpt_path = self.CHECKPOINT_PATH + '/-' + ckpt
            
        self._saver.restore(self._sess,os.path.join(ckpt_path))
    
    def _load_tensors(self, src):
        tree = {}
        if isinstance(src, dict):
            for k, v in src.items():
                if isinstance(v, dict):
                    sub_tree = self._load_tensors(v)
                    tree[k] = sub_tree
                else:
                    if isinstance(v, list):
                        tree[k] = [self._graph.get_tensor_by_name(t) for t in v]
                    else:
                        tree[k] = self._graph.get_tensor_by_name(v) if v is not None else None

        return tree

    
class saver(object):

    def __init__(self, sess, model_setup, n_hours=1000000, max_to_keep=1):

        self.sess = sess
        self.model_setup = model_setup
        self.name = self.model_setup['model_name']
        
        self.MODEL_DIR = os.path.join('saved_models', self.name)
        self.CHECKPOINT_DIR = os.path.join(self.MODEL_DIR, "checkpoints/")
        self.SUMMARY_DIR = os.path.join(self.MODEL_DIR, "summary")
        self.LOG_DIR = os.path.join(self.MODEL_DIR, "logs/")
        
        self._saver = tf.train.Saver(keep_checkpoint_every_n_hours=n_hours,
                                     max_to_keep=max_to_keep)
        self._summary_writer = None

        #     if not os.path.exists(self.LOG_DIR):
        #         os.makedirs(self.LOG_DIR)
        #     self._log_writer = open(self.LOG_DIR+'training_log.ndjson', 'w')

        
    def get_model_path(self):
        return self.MODEL_DIR
    
    def save_checkpoint(self, step):

        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)
            
        if not os.path.exists(self.CHECKPOINT_DIR):
            os.makedirs(self.CHECKPOINT_DIR)

        if not os.path.exists(self.SUMMARY_DIR):
            os.makedirs(self.SUMMARY_DIR)
            graph = tf.get_default_graph()
            self._summary_writer = tf.summary.FileWriter(self.SUMMARY_DIR, graph=graph)

        if not os.path.exists(self.MODEL_DIR + '/setup.pkl') and any(self.model_setup):
            pickle.dump(self.model_setup, open(self.MODEL_DIR + '/setup.pkl', 'wb'))

            
        self._saver.save(self.sess, 
                         global_step = step,
                         save_path=self.CHECKPOINT_DIR)

    def load_checkpoint(self, ckp_path):

        self._saver.restore(self.sess, ckp_path)

    def save_summary(self, step, summary):

        if self._summary_writer is None:
            graph = tf.get_default_graph()
            self._summary_writer = tf.summary.FileWriter(self.SUMMARY_DIR, graph=graph)
        
        # self.log_writer.write(json.dumps(summary) + '\n')
        # self.log_writer.flush()

        ep_summary = tf.Summary()
        for k, v in summary.items():
            ep_summary.value.add(simple_value=v, tag=k)

        self._summary_writer.add_summary(ep_summary, step)
        self._summary_writer.flush()

