#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy
import numpy as np
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
        self.tau = 1000


        self.rho = 1
        self.betas = [numpy.power(1 - self.eta * self.lam, i) for i in range(self.tau + 1)]
        self.alphas = []

        self.xs = []

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
            beta_idx = self.t - i - 2
            # print "beta: ", self.betas[beta_idx]
            # print "ke: ", ke
            # print "alpha: {}".format(self.alphas[i])

            ke = self.kernel_func(x, 
                                  self.xs[i],
                                  self.kernel_params)

            pred += self.alphas[i] * self.betas[beta_idx] * ke#[i]
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

        #todo... really store this x?

        self.xs.append(x)
        self.t += 1

        assert(len(self.xs) == len(self.alphas))

        assert(self.t - 1 == len(self.xs))

            
    def evaluate(self, X, Y):

        Y_p = numpy.zeros_like(Y)
        rhos = numpy.zeros_like(Y)


        for (xi, x) in enumerate(X):
            if xi % 20:
                print xi

            y_p = self.predict(x)
            Y_p[xi] = y_p
            rhos[xi] = self.rho

            # TODO SUPER HACK
            if xi < 500:
                self.fit(x, pred = y_p)
        return Y_p, rhos
        
            
def main():
    nu = 0.1

    ca01_feats_orig = numpy.load('feats/ca01_no_label_bov.npz')['arr_0'][15:]

    # le_weird = ca01_feats[242:270, :]
    # for x in range(20):
    #     ca01_feats = numpy.vstack((ca01_feats, le_weird))
    # ca01_feats = numpy.vstack((ca01_feats, ca01_feats))

    # ca01_feats = numpy.hstack((ca01_feats, numpy.ones((ca01_feats.shape[0], 1))))

    # oocs = online_one_class_svm(nu, ca01_feats.shape[1])

    # y_p = oocs.evaluate(ca01_feats, numpy.zeros((ca01_feats.shape[0], 1)))
   
    shuffle_inds = numpy.asarray(range(ca01_feats_orig.shape[0]))

    # numpy.random.shuffle(shuffle_inds)

    ca01_feats = ca01_feats_orig[shuffle_inds, :]

    kernel_params = {'sigma' : 0.005, 'bandwidth' : 0.005}
    ocn = one_class_norma(nu = 1e-2, lam = .3, eta = 5e-4, kernel_params = kernel_params)

    # kernel_surface = numpy.zeros((ca01_feats.shape[0], ca01_feats.shape[0]))
    # print "computing kernel surface"
    # for point_idx in range(ca01_feats.shape[0]):
    #     point = ca01_feats[point_idx, :]
    #     kernel_eval = uniform_kernel_func(point, ca01_feats, kernel_params)
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
    y_p, rhos = ocn.evaluate(ca01_feats, numpy.zeros((ca01_feats.shape[0], 1)))

    y_p_fixed = numpy.zeros_like(y_p)
    y_p_fixed[shuffle_inds] = y_p

    rhos_fixed = numpy.zeros_like(rhos)
    rhos_fixed[shuffle_inds] = rhos

    plt.plot(range(y_p.shape[0]), y_p_fixed)
    plt.plot(range(rhos.shape[0]), rhos_fixed)
    plt.show(block = False)
    pdb.set_trace()
    
if __name__ == '__main__':
    pdbw.pdbwrap(main)()
        

    
