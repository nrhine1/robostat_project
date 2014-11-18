#!/usr/bin/env python

# Mean shift implementation largely provided by:
# http://sociograph.blogspot.com/2011/11/scalable-mean-shift-clustering-in-few.html?view=classic

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy
import numpy as np

import sklearn.metrics
from sklearn.neighbors import BallTree
from sklearn.utils import extmath
from sklearn.metrics.pairwise import euclidean_distances

import pdb
#import gtkutils.pdbwrap as pdbw
#import gtkutils.img_util as iu

 
def gaussian_kernel_update(x, points, bandwidth):
    distances = euclidean_distances(points, x)
    weights = np.exp(-1 * (distances ** 2 / bandwidth ** 2))
    return np.sum(points * weights, axis=0) / np.sum(weights)
 
def flat_kernel_update(x, points, bandwidth):
    return np.mean(points, axis=0)

def gaussian_kernel_func(x0, x1, sigma=0.005):
    if x0.ndim == 2 or x1.ndim == 2:
        axis = 1
    else:
        axis = 0
    return 1 / (sigma * numpy.sqrt(2 * numpy.pi )) * numpy.exp(-.5 * numpy.power(numpy.linalg.norm(x0 - x1, axis = axis)/ sigma, 2))


class online_meanshift:
    modes = np.array([]) # List of modes: KxD
    assigments = np.array([]) # Indices into modes

    def __init__(self):
        pass
    
#    def InitModes(self, xtrain):
    def MeanShift(self, X, bandwidth=0.01, seeds=None, kernel_update_function=gaussian_kernel_update, max_iterations=300):
        n_points, n_features = X.shape
        stop_thresh = 1e-3 * bandwidth  # when mean has converged
        cluster_centers = []
        ball_tree = BallTree(X)  # to efficiently look up nearby points

        if seeds==None:
            seeds=X.copy()
        
        # For each seed, climb gradient until convergence or max_iterations
        
        j = 1
        for weighted_mean in seeds:
             completed_iterations = 0
             while True:
                 points_within = X[ball_tree.query_radius([weighted_mean], bandwidth*3)[0]]
                 old_mean = weighted_mean  # save the old mean
                 weighted_mean = kernel_update_function(old_mean, points_within, bandwidth)
                 converged = extmath.norm(weighted_mean - old_mean) < stop_thresh
                 if converged or completed_iterations == max_iterations:
                     cluster_centers.append(weighted_mean)
                     break
                 completed_iterations += 1
             j += 1
             print "Mode for {} computed; converged={}".format(j,converged)
        return cluster_centers
    
    def OnlineUpdate(self, xt):
        
        xt.mode = mode
        if mode not in self.modes:
            self.modes.append(mode)
        else:
            shift_mode(mode)
        return

    def ShiftMode(self, mode, x):
        N = assignments.length()
        new_mode = ((N-1) * mode + x) / N
        return new_mode
    
    def Test(self):
        pass

    def Evaluate(self):
        pass


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
    
    oms = online_meanshift()
    oms.MeanShift(ca01_feats[:100,:])
    oms.Test()
    oms.Evaluate()

    print "Done!"

if __name__ == '__main__':
    #pdbw.pdbwrap(main)()
    main()    
