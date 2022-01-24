import logging
import sys
sys.path.append('..')

import os
import argparse
import numpy as np
import tensorflow as tf
import ipdb

from lib.vis_utils import *
from lib.logging_utils import *
from lib.common_utils import *
from lib.data_utils import *
from data.load_data import * 
import model.vrl as vrl

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("-e", "--exp", type=int, default=0,
                        help="which experiment to run")
    args = parser.parse_args()

    setup = {
        'max_iter': 3e5,
        'batch_size': 100,
        'lr_init': 4e-4,
        'lr_decay_step': 1e5,
        'use_rpda': True
    }

    # MNIST 3-rel
    if args.exp == 0:
        setup['data_file'] = 'mnist'
        setup['z_dim'] = 2
        rel_sel = {0:list(np.random.permutation([0,1,2,3,4])[:3])}
        setup['relation_sel'] = {d:rel_sel for d in range(10)}
        setup['vrl_model'] = vrl.vrl2D_mlp_bce
        # setup['vrl_model'] = vrl.vrl2D_gumbel_bce
        # setup['use_rpda'] = False

    # MNIST 5-rel
    elif args.exp == 1:
        setup['data_file'] = 'mnist'
        setup['z_dim'] = 2
        rel_sel = {0:list(np.random.permutation([0,1,2,3,4])[:5])}
        setup['relation_sel'] = {d:rel_sel for d in range(10)}
        setup['vrl_model'] = vrl.vrl2D_mlp_bce
        # setup['vrl_model'] = vrl.vrl2D_gumbel_bce

    # Omniglot 3-rel
    elif args.exp == 2:
        setup['data_file'] = 'omniglot'
        setup['z_dim'] = 2
        rel_sel = {0:list(np.random.permutation([0,1,2,3,4])[:3])}
        setup['relation_sel'] = rel_sel
        setup['vrl_model'] = vrl.vrl2D_mlp_bce
        # setup['vrl_model'] = vrl.vrl2D_gumbel_bce
        # setup['use_rpda'] = False

    # Omniglot 5-rel
    elif args.exp == 3:
        setup['data_file'] = 'omniglot'
        setup['z_dim'] = 2
        rel_sel = {0:list(np.random.permutation([0,1,2,3,4])[:5])}
        setup['relation_sel'] = rel_sel
        setup['vrl_model'] = vrl.vrl2D_mlp_bce
        # setup['vrl_model'] = vrl.vrl2D_gumbel_bce

    # MNIST coupled-rel
    elif args.exp == 4:
        setup['data_file'] = 'mnist'
        setup['z_dim'] = 2
        setup['relation_sel'] = {d:{0:[d//2]} for d in range(10)}
        rel_sel = {0:list(np.random.permutation([0,1,2,3,4])[:5])}
        setup['test_relation_sel'] = {d:rel_sel for d in range(10)}
        setup['vrl_model'] = vrl.vrl2D_mlp_bce

    # Yale facial expression
    elif args.exp == 5:
        setup['data_file'] = 'yale_expression'
        setup['z_dim'] = 2
        setup['relation_sel'] = [1,2,3]
        setup['vrl_model'] = vrl.vrl2D_mlp_mse

    # Yale illumination condition
    elif args.exp == 6:
        setup['data_file'] = 'yale_illumination'
        setup['z_dim'] = 2
        setup['relation_sel'] = [0, 1, 2, 3]
        setup['vrl_model'] = vrl.vrl2D_mlp_mse

    # RAVDESS
    elif args.exp == 7:
        setup['data_file'] = 'ravdess'
        setup['z_dim'] = 2
        setup['relation_sel'] = [0,1,2]
        setup['vrl_model'] = vrl.vrl2D_mlp_mse

    # MNIST 10-rel
    elif args.exp == 8:
        setup['data_file'] = 'mnist'
        setup['z_dim'] = 2
        setup['relation_sel'] = {d:{s:[0,1,2,3,4,5,6,7,8,9] if s in [0,1,2,3,4] else [0,1,2,3,4,10,11,12,13,14] for s in [0,1,2,3,4,10,11,12,13,14]} for d in range(10)}
        setup['vrl_model'] = vrl.vrl2D_mlp_bce
        setup['n_hidden'] = 1024

    # MNIST continuous-rel
    elif args.exp == 9:
        setup['data_file'] = 'mnist'
        setup['z_dim'] = 2 #3
        setup['relation_sel'] = 'continuous'
        setup['vrl_model'] = vrl.vrl2D_cnn_bce

    else:
        raise ValueError('Unrecognized data file!!')


    setup['model_name'] = args.model
    MODEL_PATH = os.path.join('saved_models', args.model)
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    SAMPLE_PATH = os.path.join(MODEL_PATH, 'sample')
    if not os.path.isdir(SAMPLE_PATH):
        os.makedirs(SAMPLE_PATH)

    # Setup logging
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    setup_console_logger(logger=logger, level='info')
    setup_file_logger(fn=os.path.join(MODEL_PATH, 'training.log'),
                      logger=logger,
                      level='info')

    # Load data
    tr_iter, vaX, rpda_aug = load_data(setup['data_file'],
                                       setup['batch_size'],
                                       setup['relation_sel'])

    setup['image_patch_size'] = list(vaX.shape[1:])
    vis_ny, vis_nx = 8, 8

    grayscale_grid_vis(vaX[:vis_ny*vis_nx,...,1], vis_ny, vis_nx,
                       save_path=os.path.join(SAMPLE_PATH, 'sample_img_c2.png'))
    grayscale_grid_vis(vaX[:vis_ny*vis_nx,...,0], vis_ny, vis_nx,
                       save_path=os.path.join(SAMPLE_PATH, 'sample_img_c1.png'))

    # Start training
    config = tf.compat.v1.ConfigProto()

    with tf.compat.v1.Session(config = config) as sess:

        model = setup['vrl_model'](sess, setup)
        
        timer = Stopwatch()
        eval_steps = 10000
        tr_loss = []
        
        timer.start()        
        while model.num_updates() < setup['max_iter']:
            
            imb, idx = next(tr_iter)
            
            # relation-preserving data augmentation
            imb_aug1 = rpda_aug(imb)
            imb_aug2 = rpda_aug(imb)
            
            loss_value = model.train(imb = imb_aug1,
                                     c1 = imb_aug2[...,:1],
                                     c2 = imb_aug2[...,1:])
            
            tr_loss.append(loss_value)
            
            # evaluation
            if model.num_updates() % eval_steps == 0:
                tr_loss_avg = np.array(tr_loss).mean()
                tr_loss_std = np.array(tr_loss).std()
                tr_loss = []
                h, m, s = timer('lapse', 'hms')
                
                msg = (f"[{args.model}]: "
                       f"Time:{int(h):d}:{int(m):02d}:{int(s):02d}, "
                       f"Update:{model.num_updates()}, "
                       f"Tr_loss:{tr_loss_avg:+.2f} +/- {tr_loss_std:.2f}")
                
                logger.info(msg)

                model.saver.save_checkpoint(model.num_updates())

                _, rx_vax = model.validate(vaX)
                grayscale_grid_vis(rx_vax[:vis_ny*vis_nx,...,0],
                                   vis_ny, vis_nx,
                                   save_path=os.path.join(SAMPLE_PATH, f'{model.num_updates()}.png'))
