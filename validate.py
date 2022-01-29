import matplotlib
# matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = True

import sys
sys.path.append('..')

import os
import argparse
import tensorflow as tf
import numpy as np
import pickle 
import ipdb
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import imageio

from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.data_utils import *
from lib.vis_utils import *
from lib.saver import *
from data.load_data import * 


def get_marker_style(num):
    
    mtype = [["o",0],["v",0],["+",1],["x",1],["d",0],["s",0],[">",0],["<",0],["^",0]]
    mcolor = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
    
    marker_style = dict(marker = mtype[num % len(mtype)][0],
                        edgecolor = mcolor[num % len(mcolor)],
                        facecolor = mcolor[num % len(mcolor)] if mtype[num % len(mtype)][1] else 'none',
                        linewidths=1.1)
    return marker_style
    

def evaluate(MODEL, CHECKPOINT):

    data_dir = os.getenv('DATA_PATH')
    assert data_dir is not None

    config = tf.compat.v1.ConfigProto()
    with tf.compat.v1.Session(config = config) as sess:

        model = load_model(sess, MODEL, CHECKPOINT)

        test_rel = model.setup['relation_sel'] if 'test_relation_sel' not in model.setup else model.setup['test_relation_sel']
        
        data = pickle.load(open(os.path.join(data_dir,model.setup['data_file']+'.pkl'), 'rb'))
        relation_map = data['relation_map']
        relation_set = data['relation_set']

        
        # Learning discrete relationship with continuous or discrete
        # latent variable
        if model.setup['relation_sel'] != 'continuous':

            tr_iter,_,rpda_aug=load_data(fn=model.setup['data_file'],
                                         bsize=256,
                                         relsel=test_rel,
                                         train=False)
            
            im, z = None, None
            r = None

            for _ in range(8):

                imb, idx = next(tr_iter)

                imb = rpda_aug(imb)

                feed_dict = {model.tensors['input_image_pl']: imb}

                zb = sess.run(model.tensors['inference_op'],
                              feed_dict=feed_dict)

                zb = np.concatenate((zb, np.asarray(idx).reshape(-1, 1)), axis=1)
                z = zb if z is None else np.concatenate((z, zb), axis=0)
                im = imb if im is None else np.concatenate((im, imb), axis=0)

            # Calculate unsupervised clustering accuracy for continuous latent variable
            if 'N' not in model.setup and 'K' not in model.setup:
                
                unique = np.unique(z[:,-1])
                u_mstyle = {unique[i]:get_marker_style(i) for i in range(unique.shape[0])}

                #calculate accuracy
                ti = np.random.permutation(z.shape[0])[:int(z.shape[0]/20)]
                X = z[ti,:-1]
                y = z[ti,-1].astype(int)
                # clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
                clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', tol=1e-5))
                clf.fit(X, y)
                py = clf.predict(z[:, :-1])
                acc = np.array([a==b for a,b in zip(py, z[:,-1].astype(int))]).sum()/z.shape[0]
                print(f'Clustering accuracy:{acc}')

                # more than 2D
                if z.shape[1] > 3:
                    z_embedded = TSNE(n_components=2).fit_transform(z[:,:-1])
                    z = np.concatenate((z_embedded, z[:,-1:]), axis=1)

                # 1D
                if z.shape[1] < 3:
                    tz = np.zeros((z.shape[0], 2))
                    tz[:,:1] = z[:, :-1]
                    z = np.concatenate((tz, z[:,-1:]), axis=1)

                x_range, y_range = z[:,0].max()-z[:,0].min()+0.1, z[:,1].max()-z[:,1].min()+0.1

                x_min, x_max = z[:,0].min()-x_range*0.1, z[:,0].max()+x_range*0.1
                y_min, y_max = z[:,1].min()-y_range*0.1, z[:,1].max()+y_range*0.1
                fig, ax = plt.subplots(figsize=(6.5, 6))
                for u in unique:
                    ax.scatter(z[z[:,2]==u,0], z[z[:,2]==u,1], s=80,
                               label=relation_set[u], **u_mstyle[u])

                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max + 0.1*(y_max-y_min)])
                ax.set_xlabel(r'$\mathbf{z}_1$', fontsize=18, labelpad=0)
                ax.set_ylabel(r'$\mathbf{z}_2$', fontsize=18, labelpad=-5)
                ax.set_title(f'Unsupervised clustering accuracy: {acc*100:.2f}\%')
                ax.tick_params(labelsize=12)
                fontP = FontProperties()
                fontP.set_size('small')
                ax.legend(prop=fontP, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.))

                plt.tight_layout()

                FIG_PATH = os.path.join(model.MODEL_PATH, 'figures')
                if not os.path.isdir(FIG_PATH):
                    os.makedirs(FIG_PATH)
                fig.savefig(os.path.join(FIG_PATH, 'z_scatter.pdf'), bbox_inches="tight")


            # Calculate unsupervised clustering accuracy for discrete latent variable
            else:

                unique = np.unique(z[:,-1])
                uidx = {z:i for i,z in enumerate(unique)}

                N = model.setup['N']
                K = model.setup['K']

                #calculate accuracy
                ag = {}
                ti = np.random.permutation(z.shape[0])[:int(z.shape[0]/10)]
                for i in ti:
                    idx = []
                    for n in range(N):
                        idx.append(np.argmax(z[i, (n*K):(n*K+K)]))
                    idx = tuple(idx)
                    if idx in ag:
                        ag[idx].append(z[i,-1])
                    else:
                        ag[idx]=[z[i,-1]]

                mp = {}
                for k in ag:
                    mp[k] = max(set(ag[k]), key=ag[k].count)

                cc = 0
                for i in range(z.shape[0]):
                    idx = []
                    for n in range(N):
                        idx.append(np.argmax(z[i, (n*K):(n*K+K)]))
                    idx = tuple(idx)
                    if idx in mp and mp[idx] == z[i,-1]:
                        cc += 1

                acc = cc/z.shape[0]
                print(f'Clustering accuracy:{acc}')

                bar = np.zeros((unique.shape[0],z.shape[1]-1))

                for i in range(z.shape[0]):
                    bar[uidx[z[i,-1]],:] += z[i,:-1]

                rows, cols = unique.shape[0], 1
                # fig = plt.figure(constrained_layout=False, figsize=(fx, fy))
                fig = plt.figure(constrained_layout=False)
                gs = fig.add_gridspec(nrows=rows, ncols=cols)
                ax = [[fig.add_subplot(gs[r,c]) for c in range(cols)] for r in range(rows)]
                ax[0][0].set_title(f'Unsupervised clustering accuracy: {acc*100:.2f}\%')
                for i in range(unique.shape[0]):
                    ax[i][0].bar(np.arange(bar.shape[1]), bar[i,:])


            # Generate relational mapping example

            srcX, vaX = load_eval_img(model.setup['data_file'],
                                      test_rel,
                                      rpda_aug)

            feed_dict = {model.tensors['input_image_pl']: srcX}

            zb = sess.run(model.tensors['inference_op'], feed_dict=feed_dict)

            img_gen = []
            for i in range(zb.shape[0]):

                feed_dict = {model.tensors['input_image_c1_pl']: vaX[i],
                             model.tensors['z_pl']: np.tile(zb[i,...], (vaX[i].shape[0], 1))}

                imb = sess.run(model.tensors['image_gen_op'], feed_dict=feed_dict)
                vax_imb = np.concatenate((vaX[i], imb), axis=3)
                vax_imb = np.concatenate((srcX[i:i+1,...], vax_imb), axis=0)                
                img_gen.append(np.expand_dims(vax_imb, axis=0))

            img_gen = np.concatenate(img_gen, axis=0)

            nrows, ncols = srcX.shape[0], 1+vaX[0].shape[0]

            figx = 4.
            figy_s = 0.98
            figx_s = 0.98

            font_size = 10
            figy = figx*nrows/(ncols*2)
            fig = plt.figure(constrained_layout=False, figsize=(figx, figy), dpi=300)

            gs1 = fig.add_gridspec(nrows=1, ncols=1+vaX[0].shape[0])
            gs1.update(left=0.02,right=figx_s,top=figy_s,bottom=0.02,wspace=0.06, hspace=0.03)

            gs2a = []
            for c in range(1+vaX[0].shape[0]):
                gs2 = gs1[c].subgridspec(nrows, 2, wspace=0.02, hspace=0.03)
                gs2a.append(gs2)

            for c in range(1+vaX[0].shape[0]):
                for r in range(nrows):
                    ax = fig.add_subplot(gs2a[c][r,0])
                    ax.imshow(img_gen[r, c,...,0], cmap='gray')
                    ax.axis('off')

                    ax = fig.add_subplot(gs2a[c][r,1])
                    ax.imshow(img_gen[r, c,...,1], cmap='gray')
                    ax.axis('off')

            FIG_PATH = os.path.join(model.MODEL_PATH, 'figures')
            fn = f'{model.setup["data_file"]}_gen.pdf'
            if not os.path.isdir(FIG_PATH):
                os.makedirs(FIG_PATH)
            fig.savefig(os.path.join(FIG_PATH, fn), bbox_inches="tight")

            plt.show()

            
        # Learning continuous relative rotation relationship with
        # continuous 2-D latent variable
        else:

            tr_iter,_,rpda_aug=load_data(fn=model.setup['data_file'],
                                         bsize=10,
                                         relsel=test_rel,
                                         train=False)


            rot_aug = RotateNN(model.setup['image_patch_size'][:-1])

            im, z, c = None, None, None

            for d in range(0, 360):

                imb, idx = next(tr_iter)

                imb = np.concatenate([imb[...,0:1], rot_aug(imb[...,0:1], np.ones(imb.shape[0])*d)], axis=3)

                feed_dict = {model.tensors['input_image_pl']: imb,
                             model.tensors['bn_istraining_pl']: False}

                zb = sess.run(model.tensors['inference_op'], feed_dict=feed_dict)
                cb = np.ones(zb.shape[0])*d 
                c = cb if c is None else np.concatenate((c, cb), axis=0)
                z = zb if z is None else np.concatenate((z, zb), axis=0)
                im = imb if im is None else np.concatenate((im, imb), axis=0)

            #%% Create Color Map
            colormap = plt.get_cmap("hsv")
            norm = matplotlib.colors.Normalize(vmin=min(c),vmax=max(c))

            if z.shape[1] == 2:
            
                fig = plt.figure()
                ax = fig.add_subplot(111)
                s = ax.scatter(z[:,0], z[:,1], s=10, c=colormap(norm(c)), cmap='hsv', marker='o')

                sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm)
                cbar.set_ticks([0, 60, 120, 180, 240, 300, 359])
                cbar.set_ticklabels([r'$0$', r'$60$', r'$120$', r'$180$', r'$240$', r'$300$', r'$360$']) 
                ax.set_xlabel(r'$\mathbf{z}_1$', fontsize=18, labelpad=0)
                ax.set_ylabel(r'$\mathbf{z}_2$', fontsize=18, labelpad=-5)
                ax.tick_params(labelsize=12)

                FIG_PATH = os.path.join(model.MODEL_PATH, 'figures')
                if not os.path.isdir(FIG_PATH):
                    os.makedirs(FIG_PATH)
                fig.savefig(os.path.join(FIG_PATH, 'z_cont_scatter.pdf'), bbox_inches="tight")

            else:

                if z.shape[1] > 3:
                    z_embedded = TSNE(n_components=2).fit_transform(z[:,:-1])
                    z = np.concatenate((z_embedded, z[:,-1:]), axis=1)

                row_ang = [35, -35]
                col_ang = [215, -55, 35]
                rows, cols = len(row_ang), len(col_ang)
                fig = plt.figure(constrained_layout=False, figsize=(18, 8))
                gs = fig.add_gridspec(nrows=rows, ncols=cols)
                gs.update(left=0.02,right=0.93,top=0.98,bottom=0.02)
                gs.update(wspace=0.05, hspace=0.05)

                for _r, _c in itertools.product(range(rows), range(cols)):
                    ax = fig.add_subplot(gs[_r,_c], projection='3d')
                    ax.scatter(z[:,0], z[:,1], z[:,2], s=15, c=colormap(norm(c)), cmap='hsv', marker='o')
                    ax.view_init(elev=row_ang[_r], azim=col_ang[_c])
                    ax.set_xlabel(r'$\mathbf{z}_1$', fontsize=18)
                    ax.set_ylabel(r'$\mathbf{z}_2$', fontsize=18)
                    ax.set_zlabel(r'$\mathbf{z}_3$', fontsize=18)

                    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
                    sm.set_array([])

                fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02)

                cb_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
                sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, cax=cb_ax)
                cbar.set_ticks([0, 60, 120, 180, 240, 300, 359])
                cbar.set_ticklabels([r'$0$', r'$60$', r'$120$', r'$180$', r'$240$', r'$300$', r'$360$']) 
                cbar.ax.tick_params(labelsize=20)

                FIG_PATH = os.path.join(model.MODEL_PATH, 'figures')
                if not os.path.isdir(FIG_PATH):
                    os.makedirs(FIG_PATH)
                fig.savefig(os.path.join(FIG_PATH, 'z_cont_scatter.pdf'), bbox_inches="tight")

                
            plt.show()
            

            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("-c", "--checkpoint", help="checkpoint file", default=None)

    args = parser.parse_args()
    
    evaluate(MODEL = args.model,
             CHECKPOINT = args.checkpoint)
