import cv2
import os
import glob
from PIL import Image
import numpy as np
import pickle
import ipdb
import urllib.request
import tarfile


def build():
    data_dir = os.getenv('DATA_PATH')

    assert data_dir is not None

    data_path = os.path.join(data_dir, 'YALE/centered')

    # urllib.request.urlretrieve("https://vismod.media.mit.edu/vismod/classes/mas622-00/datasets/YALE.tar.gz", os.path.join(data_dir,'YALE.tar.gz'))

    with tarfile.open(os.path.join(data_dir,'YALE.tar.gz')) as gz_ref:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(gz_ref, data_dir)

    emotions = ['normal', 'happy', 'surprised', 'sad']

    relation_set = {0: 'neutral',
                    1: 'happy',
                    2: 'surprise',
                    3: 'sad',
                    4: 'neutral-happy',
                    5: 'neutral-surprise',
                    6: 'neutral-sad',
                    7: 'happy-neutral',
                    8: 'happy-surprise',
                    9: 'happy-sad',
                    10: 'surprise-neutral',
                    11: 'surprise-happy',
                    12: 'surprise-sad',
                    13: 'sad-neutral',
                    14: 'sad-happy',
                    15: 'sad-surprise'}

    relation_map = np.ones((len(emotions), len(emotions)))*-1
    relation_map[0, 0] = 0
    relation_map[1, 1] = 1
    relation_map[2, 2] = 2
    relation_map[3, 3] = 3
    relation_map[0, 1] = 4
    relation_map[0, 2] = 5
    relation_map[0, 3] = 6
    relation_map[1, 0] = 7
    relation_map[1, 2] = 8
    relation_map[1, 3] = 9
    relation_map[2, 0] = 10
    relation_map[2, 1] = 11
    relation_map[2, 3] = 12
    relation_map[3, 0] = 13
    relation_map[3, 1] = 14
    relation_map[3, 2] = 15

    fidx = np.array([1,2,3,5,6,7,8,9,10,11,12,13,14,15,4])
    data = np.zeros((fidx.shape[0]*2, 64, 64, len(emotions)))

    amp_scale = 1/50.

    for fn in glob.glob(data_path+'/*.pgm'):
        if any(e in fn for e in emotions):
            im = cv2.imread(fn, -1)
            om = im[60:220, 30:163]
            om = np.array(Image.fromarray(om).resize(data.shape[1:3][::-1]))
            om = om*amp_scale

            idx = np.where(fidx == int(((fn.split('/')[-1]).split('.')[0])[-2:]))[0][0]
            ch = np.argwhere(np.array([e in fn for e in emotions]) == True)[0][0]
            data[idx*2, :, :, ch] = om
            data[idx*2 + 1, :, :, ch] = om[:,::-1]

    data = {'relation_set':relation_set,
            'relation_map':relation_map,
            'data':data}

    pickle.dump(data, open(os.path.join(data_dir, 'yale_expression.pkl'), 'wb'))

if __name__ == '__main__':
    build()
