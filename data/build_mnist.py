import sys
sys.path.append('..')

import numpy as np
from scipy.ndimage import rotate, zoom
import os
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from tqdm import tqdm
import itertools
import pickle
import ipdb
import urllib.request
import gzip
import shutil

def build():
    data_dir = os.getenv('DATA_PATH')

    assert data_dir is not None

    # urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", os.path.join(data_dir,'train-images-idx3-ubyte.gz'))
    # urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", os.path.join(data_dir,'train-labels-idx1-ubyte.gz'))
    # urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", os.path.join(data_dir,'t10k-images-idx3-ubyte.gz'))
    # urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz'))

    with gzip.open(os.path.join(data_dir,'train-images-idx3-ubyte.gz'), 'rb') as f_in:
        with open(os.path.join(data_dir,'train-images-idx3-ubyte'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    with gzip.open(os.path.join(data_dir,'train-labels-idx1-ubyte.gz'), 'rb') as f_in:
        with open(os.path.join(data_dir,'train-labels-idx1-ubyte'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    with gzip.open(os.path.join(data_dir,'t10k-images-idx3-ubyte.gz'), 'rb') as f_in:
        with open(os.path.join(data_dir,'t10k-images-idx3-ubyte'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    with gzip.open(os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz'), 'rb') as f_in:
        with open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


    def transform(X):
        npx = 28
        nc = 1
        return ((X.astype(np.float32))/255.).reshape(-1, npx, npx, nc)

    def list_shuffle(*data):
        idxs = np_rng.permutation(np.arange(len(data[0])))
        if len(data) == 1:
            return [data[0][idx] for idx in idxs]
        else:
            return [[d[idx] for idx in idxs] for d in data]

    def shuffle(*arrays, **options):
        if isinstance(arrays[0][0], str):
            return list_shuffle(*arrays)
        else:
            return skutils.shuffle(*arrays, random_state=np_rng)

    def mnist():
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28*28)).astype(float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000))

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28*28)).astype(float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000))

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        return trX, teX, trY, teY

    trX, teX, trY, teY = mnist()

    trx_num, tex_num = trX.shape[0], teX.shape[0]

    trX = transform(trX)
    teX = transform(teX)

    # Binarization
    trX[trX >= .5] = 1.
    trX[trX < .5] = 0.
    teX[teX >= .5] = 1.
    teX[teX < .5] = 0.

    relation_set = {0: '+0',
                    1: '+72',
                    2: '+144',
                    3: '+216',
                    4: '+288',
                    5: '+0, x1.5',
                    6: '+72, x1.5',
                    7: '+144, x1.5',
                    8: '+216, x1.5',
                    9: '+288, x1.5',
                    10: '+0, x0.66',
                    11: '+72, x0.66',
                    12: '+144, x0.66',
                    13: '+216, x0.66',
                    14: '+288, x0.66',
                    15: '+0, x0.44',
                    16: '+72, x0.44',
                    17: '+144, x0.44',
                    18: '+216, x0.44',
                    19: '+288, x0.44',
                    20: '+0, x2.25',
                    21: '+72, x2.25',
                    22: '+144, x2.25',
                    23: '+216, x2.25',
                    24: '+288, x2.25',}

    relation_map = np.ones((15, 15))*-1
    inc = [0, 1, 2, 3, 4]
    for s, t in itertools.product(range(15), range(15)):
        if s in [0, 1, 2, 3, 4]:
            if t in [0, 1, 2, 3, 4]:
                b = 0
            elif t in [5, 6, 7, 8, 9]:
                b = 5
            elif t in [10, 11, 12, 13, 14]:
                b = 10
        elif s in [5, 6, 7, 8, 9]:
            if t in [0, 1, 2, 3, 4]:
                b = 10
            elif t in [5, 6, 7, 8, 9]:
                b = 0
            elif t in [10, 11, 12, 13, 14]:
                b = 15
        elif s in [10, 11, 12, 13, 14]:
            if t in [0, 1, 2, 3, 4]:
                b = 5
            elif t in [5, 6, 7, 8, 9]:
                b = 20
            elif t in [10, 11, 12, 13, 14]:
                b = 0

        relation_map[s,t] = b + inc[(t%5)-(s%5)]


    trX_aug = np.zeros((trX.shape[0], 32, 32, 15))
    teX_aug = np.zeros((teX.shape[0], 32, 32, 15))

    trX_aug[:, 2:30, 2:30 , 0:1] = trX
    teX_aug[:, 2:30, 2:30 , 0:1] = teX

    for X in [trX_aug, teX_aug]:
        for i in tqdm(range(X.shape[0])):
            X[i,...,1] = rotate(X[i,...,0], angle=72, order=1, reshape=False)
            X[i,...,2] = rotate(X[i,...,0], angle=144, order=1, reshape=False)
            X[i,...,3] = rotate(X[i,...,0], angle=216, order=1, reshape=False)
            X[i,...,4] = rotate(X[i,...,0], angle=288, order=1, reshape=False)

            X[i,...,5] = zoom(X[i,...,0], 1.5, order=1)[8:40, 8:40]
            X[i,...,6] = zoom(X[i,...,1], 1.5, order=1)[8:40, 8:40]
            X[i,...,7] = zoom(X[i,...,2], 1.5, order=1)[8:40, 8:40]
            X[i,...,8] = zoom(X[i,...,3], 1.5, order=1)[8:40, 8:40]
            X[i,...,9] = zoom(X[i,...,4], 1.5, order=1)[8:40, 8:40]

            X[i, 5:26, 5:26, 10] = zoom(X[i,...,0], 0.666, order=1)
            X[i, 5:26, 5:26, 11] = zoom(X[i,...,1], 0.666, order=1)
            X[i, 5:26, 5:26, 12] = zoom(X[i,...,2], 0.666, order=1)
            X[i, 5:26, 5:26, 13] = zoom(X[i,...,3], 0.666, order=1)
            X[i, 5:26, 5:26, 14] = zoom(X[i,...,4], 0.666, order=1)

        X[X > 0.5] = 1
        X[X < 0.5] = 0


    # write to file
    data = {'relation_set':relation_set,
            'relation_map':relation_map}

    pickle.dump(data, open(os.path.join(data_dir, 'mnist.pkl'), 'wb'))

    with h5py.File(os.path.join(data_dir, 'mnist.h5'), 'w') as fh:

        fh.create_dataset("images",
                          shape=(trx_num+tex_num,
                                 trX_aug.shape[1],
                                 trX_aug.shape[2],
                                 trX_aug.shape[3]), 
                          dtype = np.float32)

        fh.create_dataset("labels",
                          shape=(trx_num+tex_num,),
                          dtype = np.uint8)

        fh['images'][0:trx_num, ...] = trX_aug
        fh['images'][trx_num:, ...] = teX_aug

        fh['labels'][0:trx_num, ...] = trY
        fh['labels'][trx_num:, ...] = teY

        idx = np.arange(trx_num+tex_num)

        fh['train_indices'] = idx[:trx_num]
        train_ref = fh['train_indices'].ref
        fh['test_indices'] =idx[trx_num:]
        test_ref = fh['test_indices'].ref

        split_dict = {
             'train': {'images': (-1, -1, train_ref),
                       'targets': (-1, -1, train_ref),
                       'labels': (-1, -1, train_ref)},

             'test': {'images': (-1, -1, test_ref),
                      'targets': (-1, -1, test_ref),
                      'labels': (-1, -1, test_ref)}}

        fh.attrs['split'] = H5PYDataset.create_split_array(split_dict)

if __name__ == '__main__':
    build()
