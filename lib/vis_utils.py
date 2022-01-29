import numpy as np
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable


def grayscale_grid_vis(X, nh, nw, img_range=None, save_path=None):
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = int(n%nw)
        if np.max(x.reshape(-1)) > np.min(x.reshape(-1)):
            img[j*h:j*h+h, i*w:i*w+w] = x

    if img_range is not None:
        img = np.clip(img, img_range[0], img_range[1])
        img[0, 0] = img_range[0]
        img[-1, -1] = img_range[1]

    img = img - img.min()
    img = img/(img.max()+1e-8)
    
    if save_path is not None:
        imageio.imwrite(save_path, img)
        
    return img

def color_grid_vis(X, nh, nw, save_path=None):
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = int(n%nw)
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    if save_path is not None:
        imageio.imwrite(save_path, img)
    return img


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def fig2data ( fig ):
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data

def rgb2gray(img):
    # 0.21 R + 0.72 G + 0.07 B
    return 0.21*img[...,0] + 0.72*img[..., 1] + 0.07*img[..., 2]
