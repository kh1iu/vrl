import os
import glob
import math
import numpy as np
import pickle
import ipdb
import librosa
import librosa.display
import itertools
import scipy
import urllib.request
import zipfile


def build():
    
    data_dir = os.getenv('DATA_PATH')

    assert data_dir is not None

    data_path = os.path.join(data_dir, 'ravdess_speech')

    # urllib.request.urlretrieve("https://github.com/brendenlake/omniglot/archive/refs/heads/master.zip", os.path.join(data_dir, 'Audio_Speech_Actors_01-24.zip'))

    with zipfile.ZipFile(os.path.join(data_dir, 'Audio_Speech_Actors_01-24.zip'), 'r') as zip_ref:
        os.mkdir(os.path.join(data_dir, 'ravdess_speech'))
        zip_ref.extractall(os.path.join(data_dir, 'ravdess_speech'))


    _n_mels = 128
    _n_fft = 2048
    _hop_len = 512
    _duration = 3
    _mel_len = 128
    _zoom = 0.5

    emotions = {2: (0, 'calm'),
                5: (1, 'angry'),
                6: (2, 'fearful'),
                8: (3, 'surprised')}

    relation_set = {}
    relation_map = np.ones((len(emotions), len(emotions)))*-1
    for i, (a,b) in enumerate(itertools.product(list(emotions.keys()), list(emotions.keys()))):
        relation_map[emotions[a][0], emotions[b][0]] = i
        relation_set[i] = f'{emotions[a][1]}-{emotions[b][1]}'

    data = np.zeros((24*2*2, _n_mels, _mel_len, len(emotions)))

    data = np.zeros((24*2*2,
                     int(_n_mels*_zoom),
                     int(_mel_len*_zoom),
                     len(emotions)))

    eps = 1e-30
    amp_bias = -1*np.log(eps)
    amp_scale = 5./amp_bias

    for fn in glob.glob(data_path+'/*/*.wav'):

        _, _, _emotion, _intensity, _stmt, _rep, _id = [int(t) for t in  fn.split('/')[-1].split('.')[0].split('-')]
        
        if _emotion in emotions and _intensity==2:

            _y, _sr = librosa.load(fn)
            _y, _idx = librosa.effects.trim(_y, top_db=25)

            if _y.shape[0] >= _duration*_sr:
                _y = _y[:_duration*_sr]
            else:
                _y = np.pad(_y, (0, _duration*_sr - _y.shape[0]), "constant")

            _s = librosa.feature.melspectrogram(_y, _sr,
                                                n_mels=_n_mels,
                                                n_fft=_n_fft,
                                                hop_length=_hop_len,
                                                power=2.0)

            _s_db = librosa.amplitude_to_db(_s + 1e-8, ref=np.max)
            _s_db = _s_db[:,:_mel_len]    

            _s_db = scipy.ndimage.zoom(_s_db, _zoom)

            _s_db = np.clip((_s_db + amp_bias)*amp_scale, 0, None)

            idx = (_stmt-1)*24*2 + (_id-1)*2 + (_rep-1)
            data[idx, ..., emotions[_emotion][0]] = _s_db[::-1, :]

    data = {'relation_set':relation_set,
            'relation_map':relation_map,
            'data':data}

    pickle.dump(data, open(os.path.join(data_dir,'ravdess.pkl'), 'wb'))

if __name__ == '__main__':
    build()
