#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
import numpy as np

import sklearn.metrics

import pdb
import gtkutils.pdbwrap as pdbw
import gtkutils.img_util as iu

class online_one_class_svm(object):
    def __init__(self, nu, feature_size, phi=None):
        self.feature_size = feature_size
        self.nu = nu
        self.nu_inv = 1.0 / nu
        self.sqrt_inv_nu = numpy.sqrt(self.nu_inv)
        self.rho = 1.0
        self.w = np.zeros((self.feature_size,), dtype=np.float64)

        if phi is None:
            self.phi = lambda x:x
        self.t = 1.0
        
    def predict(self, x):
        return np.dot(self.w, self.phi(x)) - self.rho

    def fit(self, x):

        score = self.predict(x)
        grad_w = self.w
        grad_rho = -1.0
        if score <= 0:
            grad_w -= self.phi(x) * self.nu_inv
            grad_rho += self.nu_inv
        
        w_eta =  self.nu * 1.0 / self.t # TODO learning rate
        rho_eta = w_eta * 5

        self.w -= w_eta * grad_w
        self.rho -= rho_eta * grad_rho
        
        wn = numpy.linalg.norm(self.w)
        if wn > self.sqrt_inv_nu:
            self.w = self.w / wn * self.nu_inv
        # TODO projection
        self.t += 1

    def evaluate(self, X, Y):

        Y_p = numpy.zeros_like(Y)
        for (xi, x) in enumerate(X):
            y_p = self.predict(x)
            Y_p[xi] = y_p

            self.fit(x)
        return Y_p

def poly_kernel_func(x0, x1, kernel_params):
    return numpy.power((numpy.dot(x0, x1) + kernel_params['c']), kernel_params['d'])

def gaussian_kernel_func(x0, x1, kernel_params):
    if x0.ndim == 2 or x1.ndim == 2:
        axis = 1
    else:
        axis = 0
        
    return 1 / (kernel_params['sigma'] * numpy.sqrt(2 * numpy.pi )) * numpy.exp(-.5 * numpy.power(numpy.linalg.norm(x0 - x1, axis = axis)/ kernel_params['sigma'], 2))

def uniform_kernel_func(x0, x1, kernel_params):
    if x0.ndim == 2 or x1.ndim == 2:
        axis = 1
    else:
        axis = 0

    distance = numpy.linalg.norm(x0 - x1, axis = axis)
    return .5 * ( distance < float(kernel_params['bandwidth']))

class one_class_norma(object):
    def __init__(self, nu, lam, eta, kernel_type = 'gaussian', kernel_params = {'c': 1, 'd' : 2}):
        self.nu = nu
        self.lam = lam
        self.eta = eta

        self.t = 1
        self.beta_counter = 1
        self.tau = 10000
        


        self.rho = 1
        self.betas = [numpy.power(1 - self.eta * self.lam, i) for i in range(self.tau + 1)]
        self.alphas = []

        self.xs = []

        self.bt_offsets = []

        self.kernel_params = kernel_params
        self.kernel_type = kernel_type

        if self.kernel_type == 'poly':
            self.kernel_func = poly_kernel_func
        elif self.kernel_type == 'gaussian':
            self.kernel_func = gaussian_kernel_func
        elif self.kernel_type == 'uniform':
            self.kernel_func = uniform_kernel_func
        else:
            raise RuntimeError("unknown kernel: {}".format(kernel_type))

    def predict(self, x):
        pred = 0
        if len(self.alphas) == 0:
            return pred

        # print "pred"
        irange = range(max(0, self.t - self.tau - 1), self.t - 1)
        
        # ke = self.kernel_func(x, 
        #                       numpy.asarray(self.xs)[irange, :],
        #                       self.kernel_params)
        

        for i in irange:
            beta_idx = self.beta_counter - i - 2
            # print "beta: ", self.betas[beta_idx]
            # print "ke: ", ke
            # print "alpha: {}".format(self.alphas[i])

            if self.alphas[i] == 0:
                continue

            ke = self.kernel_func(x, 
                                  self.xs[i],
                                  self.kernel_params)

            pred += self.alphas[i] * self.betas[beta_idx] * ke
            # pred += self.alphas[i] * self.betas[self.beta_counter - (i + self.bt_offsets[beta_idx]) - 2] * ke#[i]
            # print "pred: {}".format(pred)

        # print "pred final: {}\n\n".format(pred)
        return pred
        
    def fit(self, x, pred = None):
        if pred is None:
            pred = self.predict(x)
        
        # for (ai, a) in enumerate(self.alphas):
        #     self.alphas[ai] = (1 - self.eta)*a
        self.alphas = list((1 - self.eta) * numpy.asarray(self.alphas))

        # abnormal... rho gets bigger
        if pred < self.rho:
            self.alphas.append(self.eta)
            self.rho += self.eta * (1 - self.nu)

        # normal... rho decreases
        else:
            self.alphas.append(0)
            self.rho += - self.eta * self.nu

        
        self.xs.append(x)
        self.t += 1

        self.beta_counter += 1
        self.bt_offsets.append(self.beta_counter - self.t)

        assert(len(self.xs) == len(self.alphas))

        assert(self.t - 1 == len(self.xs))

            
    def evaluate(self, X, Y, stop_idx = None):

        Y_p = numpy.zeros_like(Y)
        rhos = numpy.zeros_like(Y)
        classification = numpy.zeros_like(Y)
        
        consec_anomaly_thresh = 4

        consec_anomaly = 0
        min_non_anomaly_frames = stop_idx

        for (xi, x) in enumerate(X):
            if xi % 20:
                print xi

            y_p = self.predict(x)
            Y_p[xi] = y_p
            rhos[xi] = self.rho

            if y_p < self.rho and xi > stop_idx:
                consec_anomaly += 1
            else:
                consec_anomaly = 0

            if consec_anomaly >= consec_anomaly_thresh:
                for i in range(consec_anomaly_thresh):
                    classification[xi - i] = 1

            # TODO SUPER HACK
            if stop_idx is not None and xi < stop_idx:
                self.fit(x, pred = y_p)
        return Y_p, rhos, classification
        
            
def main():
    nu = 0.1

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

    ca01_feats_orig = le_normal
    ca01_labels = le_normal_labels

    for x in range(n_duplicates):
        ca01_feats_orig = numpy.vstack((ca01_feats_orig, le_normal))
        ca01_labels = numpy.vstack((ca01_labels, le_normal_labels))

    stop_training_idx = ca01_feats_orig.shape[0]
    ca01_feats_orig = numpy.vstack((ca01_feats_orig, le_weird))
    ca01_labels = numpy.vstack((ca01_labels, le_weird_labels))

    all_labels = ca01_labels
    if stack_next_seq:
        ca01_feats_orig = numpy.vstack((ca01_feats_orig, ca02_feats_orig))
        all_labels = numpy.vstack((ca01_labels, ca02_labels))

    # ca01_feats = numpy.hstack((ca01_feats, numpy.ones((ca01_feats.shape[0], 1))))

    # oocs = online_one_class_svm(nu, ca01_feats.shape[1])

    # y_p = oocs.evaluate(ca01_feats, numpy.zeros((ca01_feats.shape[0], 1)))
   
    shuffle_inds = numpy.asarray(range(ca01_feats_orig.shape[0]))

    # numpy.random.shuffle(shuffle_inds)

    ca01_feats = ca01_feats_orig[shuffle_inds, :]

    kernel_params = {'sigma' : 0.005, 'bandwidth' : 0.01}
    kernel_type = 'gaussian'
    ocn = one_class_norma(nu = 1e-2, lam = .3, eta = 1e-3, 
                          kernel_type = kernel_type,
                          kernel_params = kernel_params)

    # kernel_surface = numpy.zeros((ca01_feats.shape[0], ca01_feats.shape[0]))
    # print "computing kernel surface"
    # for point_idx in range(ca01_feats.shape[0]):
    #     point = ca01_feats[point_idx, :]
    #     kernel_eval = gaussian_kernel_func(point, ca01_feats, kernel_params)
    #     kernel_surface[point_idx, :] = kernel_eval

    # normalized_surface = kernel_surface / (kernel_surface.max() - kernel_surface.min())
    # # xmesh, ymesh = numpy.meshgrid(range(ca01_feats.shape[0]), range(ca01_feats.shape[0]))
    # iu.v(normalized_surface)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')
    # ax.plot_surface(xmesh, ymesh, kernel_surface, cmap = cm.coolwarm)
    
    # ### params

    # pdb.set_trace()


    # synth_data = numpy.vstack((numpy.arange(9, 10, .1)[:,numpy.newaxis], 
    #                            numpy.arange(19, 20, .1)[:, numpy.newaxis]))
                       
    # y_p,rhos = ocn.evaluate(synth_data,
    #                         numpy.zeros((synth_data.shape[0], 1)))
    y_p, rhos, classification = ocn.evaluate(ca01_feats, numpy.zeros((ca01_feats.shape[0], 1)),
                                             stop_idx = stop_training_idx)
    test_classification = classification[stop_training_idx:]

    y_p_fixed = numpy.zeros_like(y_p)
    y_p_fixed[shuffle_inds] = y_p

    rhos_fixed = numpy.zeros_like(rhos)
    rhos_fixed[shuffle_inds] = rhos

    # y_p_2, rhos_2 = ocn.evaluate(ca02_feats_orig, numpy.zeros((ca02_feats_orig.shape[0], 1)),
    #                              stop_idx = 0)
    # plt.plot(range(y_p_2.shape[0]), y_p_2)
    # plt.plot(range(rhos_2.shape[0]), rhos_2)

    y_p_training = y_p[:stop_training_idx]
    y_p_testing = y_p[stop_training_idx:]

    plt.plot(range(y_p_training.shape[0]), y_p_training, 'g', label='f(x) training')
    plt.plot(range(y_p_training.shape[0], y_p.shape[0]), y_p_testing, 'b', label='f(x) testing')
    plt.plot(range(rhos.shape[0]), rhos, 'k', label=u'\u03C1')

    plt.plot(range(all_labels.shape[0]), all_labels, 'rx', label='GT label')


    plt.title('Sequence Frame vs. NORMA One Class SVM Training and Testing Predictions')

    # plt.plot(range(stop_training_idx, classification.shape[0]), 
    #          test_classification, 'rx', label='classification')

    
    plt.ylim(ymin = min(-0.05, plt.ylim()[0]))

    plt.xlabel('Frame Number')
    plt.ylabel('Score')
    plt.legend(fontsize = 10)

    
    plt.show(block = False)
    
    cm = sklearn.metrics.confusion_matrix(classification[stop_training_idx:],
                                         all_labels[stop_training_idx:])

    accuracy = numpy.diag(cm).sum() / float(cm.sum())
    recall = cm[1][1] / float(cm[:, 1].sum())
    precision = cm[0][0] / float(cm[:, 0].sum())

    print "p r a ntrain ntest trainanomalies test normalities :\n{:.3f} & {:.3f} & {:.3f} & {}& {} & {} & {} ".format(precision, recall, accuracy, \
                                                                                                                   all_labels[:stop_training_idx].shape[0],
                                                                                                                   all_labels[stop_training_idx:].shape[0],
                                                                                                                   (all_labels[stop_training_idx:] == 1).sum(),
                                                                                                                   (all_labels[stop_training_idx:] == 0).sum())
                                                                                                                   

        # 01 and 02 480 : 0.841 & 0.995 & 0.875
    # 01 480: 0.103 & 1.000 & 0.783
    # 01 300: 0.732 & 1.000 & 0.813 
    # 01 100: 0.653 & 1.000 & 0.716 

    pdb.set_trace()
    
if __name__ == '__main__':
    pdbw.pdbwrap(main)()
        

    
