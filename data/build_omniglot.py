import sys
sys.path.append('..')

import numpy as np
import os
import glob
import imageio
from scipy.ndimage import rotate, zoom
import os
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from tqdm import tqdm
import itertools
import pickle
import ipdb
import urllib.request
import zipfile

def build():
    data_dir = os.getenv('DATA_PATH')

    assert data_dir is not None

    # urllib.request.urlretrieve("https://github.com/brendenlake/omniglot/archive/refs/heads/master.zip", os.path.join(data_dir,'omniglot-master.zip'))

    with zipfile.ZipFile(os.path.join(data_dir,'omniglot-master.zip'), 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    with zipfile.ZipFile(os.path.join(data_dir, 'omniglot-master/python/images_background.zip'), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(data_dir, 'omniglot-master/python'))

    with zipfile.ZipFile(os.path.join(data_dir, 'omniglot-master/python/images_evaluation.zip'), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(data_dir, 'omniglot-master/python'))

    im_shape = (32, 32)
    train_path = os.path.join(data_dir, 'omniglot-master/python/images_background')
    test_path = os.path.join(data_dir, 'omniglot-master/python/images_evaluation')

    def get_im(path):
        im_type = ['png', 'jpg', 'pgm']
        im_file = []
        for f in glob.glob(f'{path}/*'):
            if f.split('.')[-1] in im_type:
                im_file.append(f)
            elif os.path.isdir(f):
                im_file += get_im(f)
        return im_file

    train_file = get_im(train_path)
    test_file = get_im(test_path)

    trX, teX = [], []
    for id, f in enumerate(train_file):
        im = imageio.imread(f)
        zim = zoom(im, [osh/ish for ish, osh in zip(im.shape, im_shape)],order=1)
        zim[zim >= 128] = 255
        zim[zim < 128] = 0
        trX.append(zim[np.newaxis, :, :, np.newaxis])

    for id, f in enumerate(test_file):
        im = imageio.imread(f)
        zim = zoom(im, [osh/ish for ish, osh in zip(im.shape, im_shape)],order=1)
        zim[zim >= 128] = 255
        zim[zim < 128] = 0
        teX.append(zim[np.newaxis, :, :, np.newaxis])

    trX = np.concatenate(trX, axis=0)
    teX = np.concatenate(teX, axis=0)

    trX = (255 - trX)/255
    teX = (255 - teX)/255

    trx_num, tex_num = trX.shape[0], teX.shape[0]

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

    trX_aug[:, :, : , 0:1] = trX
    teX_aug[:, :, : , 0:1] = teX

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
            'relation_map':relation_map,
            'train_num':trx_num}

    pickle.dump(data, open(os.path.join(data_dir, 'omniglot.pkl'), 'wb'))

    with h5py.File(os.path.join(data_dir,'omniglot.h5'), 'w') as fh:

        fh.create_dataset("images",
                          shape=(trx_num+tex_num,
                                 trX_aug.shape[1],
                                 trX_aug.shape[2],
                                 trX_aug.shape[3]), 
                          dtype = np.float32)

        fh['images'][0:trx_num, ...] = trX_aug
        fh['images'][trx_num:, ...] = teX_aug

        idx = np.arange(trx_num+tex_num)

        fh['train_indices'] = idx[:trx_num]
        train_ref = fh['train_indices'].ref
        fh['test_indices'] =idx[trx_num:]
        test_ref = fh['test_indices'].ref

        split_dict = {
             'train': {'images': (-1, -1, train_ref),
                       'targets': (-1, -1, train_ref)},

             'test': {'images': (-1, -1, test_ref),
                      'targets': (-1, -1, test_ref)}}

        fh.attrs['split'] = H5PYDataset.create_split_array(split_dict)


if __name__ == '__main__':
    build()
