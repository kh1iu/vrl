import numpy as np
from collections import deque
import time

def listed(l):
    return l if isinstance(l, list) else [l]

def set_random_seed():
    t = int( time.time() * 1000.0 )
    np.random.seed( ((t & 0xff000000) >> 24) +
            ((t & 0x00ff0000) >>  8) +
            ((t & 0x0000ff00) <<  8) +
            ((t & 0x000000ff) << 24)   )

def random_iter(iters):
    pending = len(iters)
    while pending:
        try:
            iter_idx = np.random.randint(pending)
            yield next(iters[iter_idx])
                    
        except StopIteration:
            iters.pop(iter_idx)
            pending -= 1

def infinite_iter(stream):
    iters = random_iter([st.get_epoch_iterator() for st in stream])
    while True:
        try:
            yield next(iters)

        except StopIteration:
            iters = random_iter([st.get_epoch_iterator() for st in stream])


def roundrobin_iter(iters):
    pending = len(iters)
    nexts = cycle(it for it in iters)
    while pending:
        try:
            for n in nexts:
                yield next(n)

        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))  


class Stopwatch(object):

    def __init__(self, auto_reset=False):
        self.__time = time.time()
        self.__runtime = 0
        self.__pause = True
        self.__splits = deque(maxlen=1000)
        self.__splits.append(0)
        self.auto_reset = auto_reset
        self.reset()

    def __call__(self, type='lapse', fmt='s'):
        if type == 'split':
            t = self.__splits[0] if len(self.__splits)==1 else self.__splits[-1] - self.__splits[-2]
        else:
            t = self.__runtime
            if not self.__pause:
                t += (time.time() - self.__time)
            
        return self.format(t, fmt)

    def __enter__(self, *args):
        if self.auto_reset:
            self.reset()
        self.start()

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        if self.__pause:
            self.__time = time.time()
            self.__pause = False

    def stop(self):
        if not self.__pause:
            self.__runtime += (time.time() - self.__time)
            self.__pause = True

    def split(self):
        if not self.__pause:
            self.__splits.append(self.__call__())
            
    def reset(self):
        self.__runtime = 0
        self.__pause = True
        self.__splits.clear()
        self.__splits.append(0)
    
    def format(self, t, fmt):
        if fmt == 'ms':
            m, s = divmod(t, 60)
            return (int(m), s)
        elif fmt == 'hms':
            m, s = divmod(t, 60)
            h, m = divmod(m, 60)
            return (int(h), int(m), s)
        elif fmt == 'dhms':
            m, s = divmod(t, 60)
            h, m = divmod(m, 60)
            d, h = divmod(h, 24)
            return (int(d), int(h), int(m), s)
        else:
            return t
