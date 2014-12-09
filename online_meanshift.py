#!/usr/bin/env python

# Mean shift implementation largely provided by:
# http://sociograph.blogspot.com/2011/11/scalable-mean-shift-clustering-in-few.html?view=classic

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy
import numpy as np
import util
import Image
import os

import sklearn.metrics
from sklearn.neighbors import BallTree
from sklearn.utils import extmath
from sklearn.metrics.pairwise import euclidean_distances

import pdb
import gtkutils.pdbwrap as pdbw
import gtkutils.vis_util as vu
import gtkutils.img_util as iu

 
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

# larger sigmasq means more lenient merging and larger kernel
class online_meanshift3:
    def __init__(self, feature_size, max_nbr_samples = 200, batch_size=100, sigmasq = 0.05):
        self.samples = np.zeros((max_nbr_samples, feature_size), dtype=np.float64)
        self.max_nbr_samples = max_nbr_samples
        self.feature_size = feature_size
        self.n_samples = 0
        self.t = 1.0

        self.batch_size = batch_size
        self.n_buffered = 0
        self.sample_buffer = np.zeros((batch_size, feature_size), dtype=np.float64)

        self.sigmasquare = sigmasq

        self.modes = []
        self.mode_weights = []

    def compute_weight(self, x1, x2):
        # x1 : (m,d)
        # x2 : (n,d)
        # ret: (m,n) of dists 
        return np.exp(-(np.sum((x1[:,np.newaxis,:] - x2)**2, axis=2) / self.sigmasquare))

    def compute_dist(self, x1, x2):
        assert(x1.shape[0] == 1)
        # x1 : (1, d)
        # x2 : (n, d)
        return numpy.linalg.norm(x1 - x2, axis = 1)

    def fit(self, x):
        # use mean shift on x with the samples to find its mode.
        
        self.sample_buffer[self.n_buffered] = x
        self.n_buffered += 1
        self.shift_indicators = numpy.zeros((len(self.modes), 1), dtype = numpy.bool)

        n_modes_added = 0
        orig_num_modes = len(self.modes)



        if self.batch_size == self.n_buffered:
            print "doing mean shift"
            # do mean shift on both stored samples, and buffered samples. 
            seeds = self.sample_buffer.copy()

            normality_scores = numpy.zeros((self.batch_size))
            mode_assignments = numpy.zeros((self.batch_size), numpy.int32)

            # mean shift
            for i in range(200):
                w1 = self.compute_weight(seeds, self.samples[:self.n_samples])
                w2 = self.compute_weight(seeds, self.sample_buffer)
                seeds = (np.dot(w1, self.samples[:self.n_samples]) + 
                         np.dot(w2, self.sample_buffer)) / (np.sum(w1, axis=1) + np.sum(w2, axis=1))[:,np.newaxis]

            # for each seed determine whether it is new.
            for (s_idx, seed) in enumerate(seeds):
                has_found = False

                # when we add modes, future seeds will be compared to them because enumerate() is a generator
                d = 0

                for mode_i, mode in enumerate(self.modes):
                    d = np.linalg.norm(mode - seed)**2
                    # print d
                    # print '{} < {}'.format(d, .25 * self.sigmasquare)
                    if d < .5 *  self.sigmasquare:
                        #duplicate node
                        has_found = True
                        mode_assignment = mode_i
                        break
                    

                if not has_found:
                    self.modes.append(seed)
                    self.mode_weights.append(1.0)
                    n_modes_added += 1
                    self.shift_indicators = numpy.vstack((self.shift_indicators, False))
                    mode_assignment = len(self.modes) - 1
                    # definitely abnormal TODO
                else:                     
                    #if has_found
                    self.modes[mode_i] = (self.modes[mode_i] * self.mode_weights[mode_i] + seed) / (self.mode_weights[mode_i] + 1)
                    self.mode_weights[mode_i] += 1
                    
                    # an old mode has shifted (don't mark new modes as shifted)
                    if mode_i < orig_num_modes:
                        self.shift_indicators[mode_i] = True
                    # compute abnormal scale TODO

                dist_to_modes = self.compute_dist(seed[np.newaxis, ], numpy.asarray(self.modes))
                weighted_dists = self.mode_weights * dist_to_modes
                # normality_scores[s_idx] = weighted_dists[mode_assignment] / (numpy.sum(weighted_dists) + 
                #                                                              numpy.finfo(numpy.float64).
                mode_assignments[s_idx] = mode_assignment


                normality_scores[s_idx] = 1 - \
                                          self.mode_weights[mode_assignment] / \
                                          (numpy.sum(self.mode_weights) + numpy.finfo(numpy.float64).eps)

            # for s_idx in range(seeds.shape[0]):

            #     normality_scores[s_idx] = 1 - self.mode_weights[mode_assignments[s_idx]] / (numpy.sum(self.mode_weights) + numpy.finfo(numpy.float64).eps)
            #     print normality_scores[s_idx]

            # update saved samples
            for spl in self.sample_buffer:
                self.update_samples(spl)
                self.t += 1.

            # clear buffer
            self.n_buffered = 0
            print "added {} modes, {} total".format(n_modes_added, len(self.modes))
            return normality_scores, mode_assignments
        else:
            print "waiting for batch to arrive... {}/{}".format(self.n_buffered, self.batch_size)
            return None, None

    def update_samples(self, x):
        if self.n_samples < self.max_nbr_samples:
            self.samples[self.n_samples] = x
            self.n_samples += 1
        else:
            # this is weird... we care about newer points less?? maybe make it likelier to 
            # record them if they're far
            # away from other points in the cluster, so later points are less likely to be "novel"
            r_num = np.random.uniform(0,1) * self.t
            if r_num < 1:
                # record the sample with prob.
                idx = int(r_num * self.max_nbr_samples)
                self.samples[idx] = x     

class Mode(object):
    def __init__(self, mean, lam=1e-3):
        self.sum1 = mean
        self.sum2 = np.outer(mean, mean) + lam * np.eye(mean.shape[0])
        self.n_points = 1.

    def mean(self): self.sum1 / self.n_points

    def cov(self):
        return self.sum2 / self.n_points

    def update(self, x):
        self.sum1 += x
        self.sum2 += np.outer(x, x)
        self.n_points += 1

    def weight_pt(self, x):
        return self.n_points * scipy.stats.multivariate_normal(x, self.mean(), self.cov())
        

class online_meanshift2(object):
    def __init__(self):
        self.modes = []
        self.t = 1.0

    def mean_shift(self, x):
        weights = [ mode.weight_pt(x) for mode in self.modes]
        default_weight = meow(self.t)
        weights.append(default_weight)
        
        max_mode_i = np.argmax(weights)
        # need to also consider the spawning case: 
        if  default_weight < weights[max_mode_i]:
            max_mode = self.modes(max_mode_i)
            max_mode.update(x)
        else:
            # spawn new mode
            self.modes.append(Mode(x))
        
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

def compute_pairwise_dists(X1, X2):
    pd = numpy.zeros((X1.shape[0], X2.shape[0]))
    for (x1i, x1) in enumerate(X1):
        for (x2i, x2) in enumerate(X2):
            d = numpy.linalg.norm(x1 - x2) ** 2
            pd[x1i, x2i] = d

    return pd

def main():
    seq1 = '99'
    seq2 = '10'
    key = 'feats'

    ca01_feats_orig = numpy.load('feats/ca{}_no_label_bov.npz'.format(seq1))[key][15:]
    ca02_feats_orig = numpy.load('feats/ca{}_no_label_bov.npz'.format(seq2))['arr_0'][15:]

    # ca01_labels = numpy.load('data/ca{}_no_label_bov_labels.npz'.format(seq1))['arr_0'][15:]
    ca02_labels = numpy.load('data/ca{}_no_label_bov_labels.npz'.format(seq2))['arr_0'][15:]

    train_test_idx = 480
    n_duplicates = 0
    stack_next_seq = True

    le_weird = ca01_feats_orig[train_test_idx:, :]
    le_normal = ca01_feats_orig[:train_test_idx, :]
    # le_normal_labels = ca01_labels[:train_test_idx, :]
    # le_weird_labels = ca01_labels[train_test_idx:, :]

    for x in range(n_duplicates):
        ca01_feats_orig = numpy.vstack((ca01_feats_orig, le_normal))
        # ca01_labels = numpy.vstack((ca01_labels, le_normal_labels))
   
    shuffle_inds = numpy.asarray(range(ca01_feats_orig.shape[0]))
    ca01_feats = ca01_feats_orig[shuffle_inds, :]
    
    # oms = online_meanshift()
    # oms.MeanShift(ca01_feats[:100,:])
    # oms.Test()
    # oms.Evaluate()
    
    startup = 100
    class_thresh = startup * 2
    alarm_thresh = .5
    oms = online_meanshift3(feature_size = ca01_feats_orig.shape[1],
                            max_nbr_samples = startup,
                            batch_size = 10,
                            sigmasq = 100)
                            # sigmasq = 9.5e-6)

    all_normality_scores = numpy.zeros((ca01_feats_orig.shape[0]))
    all_mode_assignments = numpy.zeros((ca01_feats_orig.shape[0]))

    batch_idx = 0
    x_vals = range(ca01_feats_orig.shape[0])

    fig = plt.figure()
    blend_dir = 'ca{}_anomaly_video'.format(seq1)

    blend_fn = '{}/image_list.txt'.format(blend_dir)

    if not os.path.isdir(blend_dir):
        os.mkdir(blend_dir)

    blend_fh = open(blend_fn, 'w')
    

    for i in range(ca01_feats_orig.shape[0]):
        normality_scores, mode_assignments = oms.fit(ca01_feats_orig[i, :])

        frame_fn = 'data/ca{}_frames/ca{}_{:05d}.png'.format(seq1, seq1, i + 15)
        frame = iu.o(frame_fn)

        if normality_scores is not None:
            all_normality_scores[batch_idx * oms.batch_size : 
                                 (batch_idx + 1) * oms.batch_size] = normality_scores

            all_mode_assignments[batch_idx * oms.batch_size : 
                                 (batch_idx + 1) * oms.batch_size] = mode_assignments

            batch_idx += 1
            sample_idx = batch_idx * oms.batch_size
            print "on sample {}".format(sample_idx)
                
            plt.cla()
            ax = plt.plot(x_vals, all_normality_scores)
            if i > class_thresh:
                plt.fill_between(x_vals, all_normality_scores, 
                                 where=numpy.logical_and(all_normality_scores >= alarm_thresh,
                                                         numpy.asarray(x_vals) > class_thresh),
                                color='red')


            # plots cluster assignments
            # plt.scatter(x_vals,
            #             all_mode_assignments / 100.0 + 1/100.,
            #             linewidths = 1.0)


            plt.axis([0, ca01_feats_orig.shape[0], 0, 1])
            plt.xlabel('Frame Index')
            plt.ylabel('Anomaly Score')
            plt.title('Sequence {}'.format(seq1))

            plt.draw()
            plt.show(block = False)
            os.fsync(blend_fh)

        plt.draw()
        plt.show(block = False)
        ii = vu.fig2img(fig)

        ii.save('tmp/image_{}.png'.format(i))

        blended = util.blend_plot_and_frame(ii, frame, frame_weight = .7)

        save_bn = '{:05d}_blended.png'.format(i)
        Image.fromarray(blended).save('{}/{}'.format(blend_dir, save_bn))
        blend_fh.write("file '{}'\n".format(save_bn))

    blend_fh.close()

    os.chdir(blend_dir)
    ffmpeg_cmd = 'ffmpeg  -i %05d_blended.png -c:v libx264 -r 30 blended.avi'
    os.system(ffmpeg_cmd)
    print "Done!"

    pdb.set_trace()


if __name__ == '__main__':
    pdbw.pdbwrap(main)()
    # main()    
