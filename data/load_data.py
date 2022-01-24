import sys
sys.path.append('..')

import random
import numpy as np
import h5py

from lib.data_utils import *

data_dir = os.environ['DATA_PATH']

def load_data(fn, bsize, relsel, train=True, rpda=True):

    fn = os.path.join(data_dir, fn)
    
    if 'mnist' in fn:
        tr_iter = mnist_iter(fn=fn,
                             batch_size=bsize,
                             relation_sel=relsel,
                             train=train)

        vaX_iter = mnist_iter(fn=fn,
                              batch_size=100,
                              relation_sel=relsel,
                              train=False)

        vaX,_ = next(vaX_iter)

        rot_aug = RotateNN(vaX.shape[1:3])
        
        def rpda_aug(X):    
            X = rot_aug(X)
            return X


    elif 'omniglot' in fn:
        tr_iter = omniglot_iter(fn=fn,
                                batch_size=bsize,
                                relation_sel=relsel,
                                train=train)

        vaX_iter = omniglot_iter(fn=fn,
                                 batch_size=100,
                                 relation_sel=relsel,
                                 train=False)
        vaX,_ = next(vaX_iter)
        
        rot_aug = RotateNN(vaX.shape[1:3])

        def rpda_aug(X):    
            X = rot_aug(X)
            return X

    elif 'yale' in fn:
        tr_iter = yale_iter(fn=fn,
                            batch_size=bsize,
                            relation_sel=relsel)

        vaX, idx = next(tr_iter)

        rot_aug = RotateNN(vaX.shape[1:3])
        
        def rpda_aug(X):
            X_aug = None
            for i in range(X.shape[0]):
                _X = X[i,...]
                if (np.random.uniform(0, 1) > 0.5):
                    _X = np.flip(_X, axis=2)
                _X = np.expand_dims(_X, axis=0)
                X_aug = _X if X_aug is None else np.concatenate([X_aug, _X],axis=0)
                
            X_aug = rot_aug(X_aug)
            
            return X_aug

    elif 'ravdess' in fn:
        tr_iter = ravdess_iter(fn=fn,
                            batch_size=bsize,
                            relation_sel=relsel)
        
        vaX, idx = next(tr_iter)
        
        def rpda_aug(X):    
            X_aug = None
            for i in range(X.shape[0]):
                _st = np.random.randint(0, X.shape[2])
                _amp_scale = np.random.uniform(0.5, 1.5)
                _X = np.roll(X[i,...], _st, axis=1)*_amp_scale
                if (np.random.uniform(0, 1) > 0.5):
                    _X = np.flip(_X, axis=2)
                _X = np.expand_dims(_X, axis=0)
                X_aug = _X if X_aug is None else np.concatenate([X_aug, _X],axis=0)

            return X_aug

    else:
        raise ValueError('dataset not found!!')

    if not rpda:
        def rpda_aug(X):    
            return X

    return tr_iter, vaX, rpda_aug


def load_eval_img(fn, relsel, aug=None):

    fn = os.path.join(data_dir, fn)

    if aug is None:
        aug = lambda x:x

    if 'mnist' in fn:

        test_d = 3 # random.choice(list(relsel))
        test_src_idx = [random.choice(list(relsel[test_d]))]
        test_tgt_idx = relsel[test_d][test_src_idx[0]]
        test_src_idx *= len(test_tgt_idx)
        
        with h5py.File(fn+'.h5', 'r') as fh:

            teX, teY = fh['images'][60000:], fh['labels'][60000:]
            idx = np.arange(teY.shape[0])

            test_img_idx = np.random.choice(idx[teY==test_d])
            test_A = np.concatenate([teX[test_img_idx:test_img_idx+1, ...,i:i+1] for i in test_src_idx], axis=0)
            test_B = np.concatenate([teX[test_img_idx:test_img_idx+1, ...,i:i+1] for i in test_tgt_idx], axis=0)            
            srcX = np.concatenate([test_A, test_B], axis=3)
            
            idx = np.arange(teY.shape[0])
            vax_idx = np.asarray([[i for i in np.random.choice(idx[teY==y], 1)] for y in relsel.keys()]).reshape(-1)
            sel_idx = [np.random.choice(list(relsel[teY[i]].keys())) for i in vax_idx]
            vaX = np.expand_dims(teX[vax_idx, ..., sel_idx], axis=3)

            vaX = [aug(vaX) for _ in range(srcX.shape[0])]
            srcX = aug(srcX)
            
    elif 'omniglot' in fn:

        test_src_idx = [random.choice(list(relsel))]
        test_tgt_idx = relsel[test_src_idx[0]]
        test_src_idx *= len(test_tgt_idx)
        num_img = 10
        
        with h5py.File(fn+'.h5', 'r') as fh:

            teX = fh['images'][19280:]
            idx = np.arange(teX.shape[0])
            img_idx = np.random.choice(idx,
                                       size=num_img+1,
                                       replace=False)
            
            test_A = np.concatenate([teX[img_idx[0]:img_idx[0]+1, ...,i:i+1] for i in test_src_idx], axis=0)
            test_B = np.concatenate([teX[img_idx[0]:img_idx[0]+1, ...,i:i+1] for i in test_tgt_idx], axis=0)            
            srcX = np.concatenate([test_A, test_B], axis=3)
            
            vaX = np.concatenate([teX[img_idx[i+1]:img_idx[i+1]+1, ..., test_src_idx[0]:test_src_idx[0]+1] for i in range(num_img)], axis=0)

            vaX = [aug(vaX) for _ in range(srcX.shape[0])]
            srcX = aug(srcX)

    elif 'yale' in fn or 'ravdess' in fn:
        
        with open(fn+'.pkl', 'rb') as fh:

            f =  pickle.load(fh)
            relation_map = f['relation_map']
            relation_set = f['relation_set']
            data = f['data']

            num_rel = len(relsel)
            batch_size = num_rel

            test_idx = []
            for i in range(0, num_rel-1):
                for j in range(i+1, num_rel):
                    test_idx.append([relsel[i], relsel[j]])

            src_idx = 0
            test_A = np.concatenate([data[src_idx:src_idx+1, ...,i[0]:i[0]+1] for i in test_idx], axis=0)
            test_B = np.concatenate([data[src_idx:src_idx+1, ...,i[1]:i[1]+1] for i in test_idx], axis=0)

            srcX = np.concatenate([test_A, test_B], axis=3)

            vaX = []
            for i, ti in enumerate(test_idx):
                idx=np.random.choice(range(1,data.shape[0]), batch_size)
                imb=np.expand_dims(np.concatenate([data[i:i+1, ..., ti[n%2]] for n,i in enumerate(idx)], axis=0), axis=3)

                vaX.append(imb)
                
    else:
        raise ValueError('dataset not found!!')

    return srcX, vaX
