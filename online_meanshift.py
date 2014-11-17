#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
import numpy as np

import sklearn.metrics

import pdb
#import gtkutils.pdbwrap as pdbw
#import gtkutils.img_util as iu


def gaussian_kernel_func(x0, x1, kernel_params = {'sigma' : 0.005, 'bandwidth' : 0.01}):
    if x0.ndim == 2 or x1.ndim == 2:
        axis = 1
    else:
        axis = 0
        
    return 1 / (kernel_params['sigma'] * numpy.sqrt(2 * numpy.pi )) * numpy.exp(-.5 * numpy.power(numpy.linalg.norm(x0 - x1, axis = axis)/ kernel_params['sigma'], 2))


class online_meanshift:
    modes = np.array([]) # List of modes: KxD
    assigments = np.array([]) # Indices into modes

    def init_modes(self, xtrain):
        
    def mean_shift(self, xt, xprev):
        
        xt.mode = mode
        if mode not in self.modes:
            self.modes.append(mode)
        else:
            shift_mode(mode)

    def shift_mode(self, mode, x):
        N = assignments.length()
        new_mode = ((N-1) * mode + x) / N
        return new_mode


def main():
    seq1 = '09'
    seq2 = '10'
    ca01_feats_orig = numpy.load('feats/ca{}_no_label_bov.npz'.format(seq1))['arr_0'][15:]
    ca02_feats_orig = numpy.load('feats/ca{}_no_label_bov.npz'.format(seq2))['arr_0'][15:]

    ca01_labels = numpy.load('data/ca{}_no_label_bov_labels.npz'.format(seq1))['arr_0'][15:]
    ca02_labels = numpy.load('data/ca{}_no_label_bov_labels.npz'.format(seq2))['arr_0'][15:]

    train_test_idx = 480
    n_duplicates = 0
    stack_next_seq = True

    le_weird = ca01_feats_orig[train_test_idx:, :]
    le_normal = ca01_feats_orig[:train_test_idx, :]
    le_normal_labels = ca01_labels[:train_test_idx, :]
    le_weird_labels = ca01_labels[train_test_idx:, :]

    for x in range(n_duplicates):
        ca01_feats_orig = numpy.vstack((ca01_feats_orig, le_normal))
        ca01_labels = numpy.vstack((ca01_labels, le_normal_labels))
   
    shuffle_inds = numpy.asarray(range(ca01_feats_orig.shape[0]))
    ca01_feats = ca01_feats_orig[shuffle_inds, :]

    online_meanshift oms
    oms.InitModes()
    oms.Test()
    oms.Evaluate()

    print "Done!"


if __name__ == '__main__':
    #pdbw.pdbwrap(main)()
    main()    
