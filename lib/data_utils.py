import sys
sys.path.append('..')

import numpy as np
import os
from sklearn import utils as skutils
import h5py
import ipdb
import pickle
import itertools
import librosa

seed=42
np_rng = np.random.RandomState(seed)

def rand_idx_iter(idx, batch_size):
    _st, _idx = 0, np.random.permutation(idx)

    while True:
        if _st + batch_size < _idx.size:
            yield _idx[_st:_st+batch_size]
            _st += batch_size
        else:
            _st, _idx = 0, np.random.permutation(idx)

            
def mnist_iter(fn, batch_size, relation_sel, train=True):

    data = pickle.load(open(fn+'.pkl', 'rb'))
    relation_map = data['relation_map']
    relation_set = data['relation_set']
    
    with h5py.File(fn+'.h5', 'r') as fh:
        
        X = fh['images'][:60000] if train else fh['images'][60000:]
        Y = fh['labels'][:60000] if train else fh['labels'][60000:]

        if relation_sel == 'continuous':
            rot_aug = RotateNN([32, 32])
            X_idx = np.arange(X.shape[0])
        else:
            X_idx = [i for i in range(X.shape[0]) if Y[i] in list(relation_sel.keys())]

        idx_iter = rand_idx_iter(X_idx, batch_size)
        
        while True:
            
            smp_idx = next(idx_iter)

            if relation_sel == 'continuous':
                imb1 = np.expand_dims(X[smp_idx,...,0], axis=3)
                imb2 = rot_aug(imb1)
                imb = [imb1, imb2]
                rel_idx = None
            else:
                src_idx = [np.random.choice(list(relation_sel[Y[i]].keys())) for i in smp_idx]
                tgt_idx = [np.random.choice(relation_sel[Y[i]][s]) for i, s in zip(smp_idx, src_idx)]
                imb = [np.expand_dims(X[smp_idx,...,src_idx], axis=3),
                       np.expand_dims(X[smp_idx,...,tgt_idx], axis=3)]
                rel_idx = [relation_map[s,t] for s,t in zip(src_idx, tgt_idx)]

                # rel_idx = [t for t in tgt_idx]

            imb = np.concatenate(imb, axis=3)
            
            yield [imb, rel_idx]


def mnist_sample(fn, relation_sel, num_example = 10):

    with h5py.File(fn+'.h5', 'r') as fh:

        teX, teY = fh['images'][60000:], fh['labels'][60000:]
        
        idx = np.arange(teY.shape[0])
        
        tex_idx = np.asarray([[i for i in np.random.choice(idx[teY==y], num_example)] for y in relation_sel.keys()]).reshape(-1)
        
        if relation_sel == 'continuous':
            rot_aug = RotateNN([32, 32])
            vaX1 = np.expand_dims(teX[tex_idx, ..., 0], axis=3)
            vaX2 = rot_aug(vaX1)
            vaX = [vaX1, vaX2]
        else:
            src_idx = [np.random.choice(list(relation_sel[teY[i]].keys())) for i in tex_idx]
            tgt_idx = [np.random.choice(relation_sel[teY[i]][s]) for i, s in zip(tex_idx, src_idx)]            
            vaX = [np.expand_dims(teX[tex_idx, ..., src_idx], axis=3),
                   np.expand_dims(teX[tex_idx, ..., tgt_idx], axis=3)]

        vaX = np.concatenate(vaX, axis=3)
        
        return vaX

            
def omniglot_iter(fn, batch_size, relation_sel, train=True):

    data = pickle.load(open(fn+'.pkl', 'rb'))
    relation_map = data['relation_map']
    relation_set = data['relation_set']
    
    with h5py.File(fn+'.h5', 'r') as fh:
        
        X = fh['images'][:data['train_num']] if train else fh['images'][data['train_num']:]

        if relation_sel == 'continuous':
            rot_aug = RotateNN([32, 32])
            X_idx = np.arange(X.shape[0])
        else:
            X_idx = [i for i in range(X.shape[0])]

        idx_iter = rand_idx_iter(X_idx, batch_size)
        
        while True:
            
            smp_idx = next(idx_iter)

            if relation_sel == 'continuous':
                imb1 = np.expand_dims(X[smp_idx,...,0], axis=3)
                imb2 = rot_aug(imb1)
                imb = [imb1, imb2]
                rel_idx = None
            else:
                src_idx = [np.random.choice(list(relation_sel.keys())) for _ in smp_idx]
                tgt_idx = [np.random.choice(relation_sel[s]) for s in src_idx]

                imb = [np.expand_dims(X[smp_idx,...,src_idx], axis=3),
                       np.expand_dims(X[smp_idx,...,tgt_idx], axis=3)]
                rel_idx = [relation_map[s,t] for s,t in zip(src_idx, tgt_idx)]
                
                # rel_idx = [t for t in tgt_idx]

            imb = np.concatenate(imb, axis=3)
            
            yield [imb, rel_idx]
            
            
def yale_iter(fn, batch_size, relation_sel):

    f = pickle.load(open(fn+'.pkl', 'rb'))
    relation_map = f['relation_map']
    relation_set = f['relation_set']
    data = f['data']
    
    while True:
        idx = np.random.choice(range(data.shape[0]), batch_size)
        a_idx = np.random.choice(range(len(relation_sel)), batch_size)

        # b_idx = np.random.choice(range(len(relation_sel)), batch_size)

        b_idx = (a_idx+np.random.choice(range(1,len(relation_sel)),batch_size))%len(relation_sel)

        imb = [np.expand_dims(data[idx,...,np.asarray(relation_sel)[a_idx]], axis=3),
               np.expand_dims(data[idx,...,np.asarray(relation_sel)[b_idx]], axis=3)]
        
        # rel_idx = [relation_map[s,t] for s,t in zip(a_idx, b_idx)]
        
        rel_idx = [relation_map[relation_sel[s],relation_sel[t]] if s<t else relation_map[relation_sel[t],relation_sel[s]] for s,t in zip(a_idx, b_idx)]

        imb = np.concatenate(imb, axis=3)
        
        yield [imb, rel_idx]


def ravdess_iter(fn, batch_size, relation_sel):

    f = pickle.load(open(fn+'.pkl', 'rb'))
    relation_map = f['relation_map']
    relation_set = f['relation_set']
    data = f['data']
    
    while True:
        idx = np.random.choice(range(data.shape[0]), batch_size)
        a_idx = np.random.choice(range(len(relation_sel)), batch_size)
        b_idx = (a_idx+np.random.choice(range(1,len(relation_sel)),batch_size))%len(relation_sel)

        imb = [np.expand_dims(data[idx,...,np.asarray(relation_sel)[a_idx]], axis=3),
               np.expand_dims(data[idx,...,np.asarray(relation_sel)[b_idx]], axis=3)]
        
        # rel_idx = [relation_map[s,t] for s,t in zip(a_idx, b_idx)]
        
        rel_idx = [relation_map[relation_sel[s],relation_sel[t]] if s<t else relation_map[relation_sel[t],relation_sel[s]] for s,t in zip(a_idx, b_idx)]

        imb = np.concatenate(imb, axis=3)
        
        yield [imb, rel_idx]

        
class ZoomNN(object):
    def __init__(self, sh):
        
        self._w, self._h = sh[1], sh[0]
        fn = f'.zoomMap_{self._w}_{self._h}'
        self.scale = [s/10 for s in range(5, 16)]
        
        try:
            data= pickle.load(open(fn, 'rb'))
            assert (data['image_size'] == (self._w, self._h) and
                    data['image_center'] == (self._w//2, self._h//2))
            self._imap = data['imap']

        except:
            self._imap = np.ones((len(self.scale), self._w*self._h), dtype=np.int)*-1
            cx, cy = self._w//2, self._h//2
            
            for i, s in enumerate(self.scale):
                for x, y in itertools.product(range(self._w), range(self._h)):
                    y_new = int((y-cy)*s + cy)
                    x_new = int((x-cx)*s + cx)
                    if 0 <= x_new < self._w and 0 <= y_new < self._h:
                        self._imap[i, y*self._h + x] = (y_new*self._h + x_new)
            
            data = {'imap':self._imap,
                    'image_size': (self._w, self._h),
                    'image_center': (cx, cy)}
            
            pickle.dump(data, open(fn,'wb'))

        self._nz = [np.argwhere(self._imap[a]>-1) for a in range(len(self.scale))]

            
    def _zoom(self, x, scale):
        
        assert x.shape[0]==self._h and x.shape[1]==self._w
        assert scale in self.scale
        sidx = self.scale.index(scale)
        out = np.zeros(x.shape)
        out.reshape(-1)[self._nz[sidx]] = x.reshape(-1)[self._imap[sidx, self._nz[sidx]]]
        
        return out

    
    def __call__(self, X, scale=None):

        if not isinstance(X, list): X = [X]
            
        if scale is None:
            _scale = np.random.choice(self.scale, size=X[0].shape[0])
        else:
            _scale = np.random.choice(scale, size=X[0].shape[0])

        rX = [np.zeros(x.shape) for x in X]
        
        for i, j in itertools.product(range(len(X)), range(X[0].shape[0])):
            rX[i][j,...,0] = self._zoom(X[i][j,...,0], _scale[j])

        return rX[0] if len(rX)==1 else rX

    
class RotateNN(object):
    
    def __init__(self, sh):
        
        self._w, self._h = sh[1], sh[0]
        fn = f'.rotMap_{self._w}_{self._h}'

        try:        
            data= pickle.load(open(fn, 'rb'))
            assert (data['image_size'] == (self._w, self._h) and
                    data['image_center'] == (self._w//2, self._h//2))
            self._imap = data['imap']

        except:
            self._imap = np.ones((360, self._w*self._h), dtype=np.int)*-1
            cx, cy = self._w//2, self._h//2
            
            for ang in range(360):
                cosa = np.cos(np.deg2rad(-1*ang))
                sina = np.sin(np.deg2rad(-1*ang))
                for x, y in itertools.product(range(self._w), range(self._h)):
                    y_new = int(np.rint(((x-cx)*sina + (y-cy)*cosa) + cy))
                    x_new = int(np.rint(((x-cx)*cosa - (y-cy)*sina) + cx))
                    if 0 <= x_new < self._w and 0 <= y_new < self._h:
                        self._imap[ang, y*self._h + x] = (y_new*self._h + x_new)
            
            data = {'imap':self._imap,
                    'image_size': (self._w, self._h),
                    'image_center': (cx, cy)}
            
            pickle.dump(data, open(fn,'wb'))

        self._nz = [np.argwhere(self._imap[a]>-1) for a in range(360)]

            
    def _rotate(self, x, ang):
        
        assert x.shape[0]==self._h and x.shape[1]==self._w
        
        aidx = int(ang)%360
        out = np.zeros(x.shape)
        out.reshape(-1)[self._nz[aidx]] = x.reshape(-1)[self._imap[aidx, self._nz[aidx]]]
        
        return out

    
    def __call__(self, X, ang=None):
            
        if ang is None:
            _ang = np.random.uniform(low=0,high=360,size=X.shape[0])
        else:
            _ang = np.asarray(ang).reshape(-1)
            if _ang.size == 1:
                _ang = np.tile(_ang, X.shape[0])
            assert _ang.size == X.shape[0]

        rX = np.zeros(X.shape)
        
        for i, j in itertools.product(range(X.shape[-1]), range(X.shape[0])):
            rX[j,...,i] = self._rotate(X[j,...,i], _ang[j])

            
        return rX

    
