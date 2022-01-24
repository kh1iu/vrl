import cv2
import os
import glob
from PIL import Image
import numpy as np
import pickle
import ipdb
import urllib.request
import zipfile

def build():

    data_dir = os.getenv('DATA_PATH')

    assert data_dir is not None

    data_path = os.path.join(data_dir, 'CroppedYale')

    # urllib.request.urlretrieve("http://vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip", os.path.join(data_dir,'CroppedYale.zip'))

    with zipfile.ZipFile(os.path.join(data_dir,'CroppedYale.zip'), 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    directions = ['left', 'front', 'right', 'above']
    angle = ['P00A+060E+20',
             'P00A+000E+00',
             'P00A-060E+20',
             'P00A+000E+90']

    relation_set = {0: 'left',
                    1: 'front',
                    2: 'right',
                    3: 'above',
                    4: 'left-front',
                    5: 'left-right',
                    6: 'left-above',
                    7: 'front-left', 
                    8: 'front-right',
                    9: 'front-above',
                    10: 'right-left',
                    11: 'right-front',
                    12: 'right-above',
                    13: 'above-left',
                    14: 'above-front',
                    15: 'above-right',        
                    }

    relation_map = np.ones((len(directions), len(directions)))*-1
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

    fidx = []
    for fn in glob.glob(data_path+'/*'):
        sn = (fn.split('/')[-1]).split('_')[0]
        if sn not in fidx:
            fidx.append(sn)

    data = np.zeros((len(fidx), 64, 64, len(directions)))

    amp_scale = 1/50.
    
    for fn in glob.glob(data_path + '/*/*.pgm'):
        id = fn.split('/')[-1].split('.')[0].split('_')[0]
        ang = fn.split('/')[-1].split('.')[0].split('_')[1]
        if ang in angle:
            im = cv2.imread(fn, -1)
            om = im[12:-12, :]
            om = np.array(Image.fromarray(om).resize(data.shape[1:3][::-1]))

            om = om * amp_scale

            idx = fidx.index(id)
            ch = angle.index(ang)
            data[idx, :, :, ch] = om

    data = {'relation_set':relation_set,
            'relation_map':relation_map,
            'data':data}

    pickle.dump(data, open(os.path.join(data_dir, 'yale_illumination.pkl'), 'wb'))

if __name__ == '__main__':
    build()
