#!/usr/bin/env python
import scipy.io
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pdb
#import gtkutils.pdbwrap as pdbw

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
    return 1 / (kernel_params['sigma'] * numpy.sqrt(2 * numpy.pi )) * numpy.exp(-.5 * numpy.power(numpy.linalg.norm(x0 - x1)/ kernel_params['sigma'], 2))

class one_class_norma(object):
    def __init__(self, nu, lam, eta, kernel_type = 'gaussian', kernel_params = {'c': 1, 'd' : 2}):
        self.nu = nu
        self.lam = lam
        self.eta = eta

        self.t = 1
        self.tau = 100


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

        else:
            raise RuntimeError("unknown kernel: {}".format(kernel_type))

    def predict(self, x):
        pred = 0
        if len(self.alphas) == 0:
            return pred

        # print "pred"
        for i in range(max(0, self.t - self.tau - 1), self.t - 1):
            beta_idx = self.t - i - 2
            # print "beta: ", self.betas[beta_idx]
            ke = self.kernel_func(x, 
                                  self.xs[i],
                                  self.kernel_params)
            # print "ke: ", ke
            # print "alpha: {}".format(self.alphas[i])

            pred += self.alphas[i] * self.betas[beta_idx] * self.kernel_func(x, 
                                                                             self.xs[i],
                                                                             self.kernel_params)
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

        print "."
        for (xi, x) in enumerate(X):
            #if xi % 20:
            #    print xi

            y_p = self.predict(x)
            Y_p[xi] = y_p
            rhos[xi] = self.rho

            self.fit(x, pred = y_p)
        return Y_p, rhos
        
            
def main():
    nu = 0.1

    ca01_feats_orig = numpy.load('feats/ca01_no_label_bov.npz')['arr_0'][15:]
    y = scipy.io.loadmat('feats/ca01_no_label_bov_XY.mat')['Y'].ravel()[15:]

    # le_weird = ca01_feats[242:270, :]
    # for x in range(20):
    #     ca01_feats = numpy.vstack((ca01_feats, le_weird))
    # ca01_feats = numpy.vstack((ca01_feats, ca01_feats))

    # ca01_feats = numpy.hstack((ca01_feats, numpy.ones((ca01_feats.shape[0], 1))))

    # oocs = online_one_class_svm(nu, ca01_feats.shape[1])

    # y_p = oocs.evaluate(ca01_feats, numpy.zeros((ca01_feats.shape[0], 1)))
   
    shuffle_inds = numpy.asarray(range(ca01_feats_orig.shape[0]))
    numpy.random.shuffle(shuffle_inds)
    ca01_feats = ca01_feats_orig[shuffle_inds, :]
    

    best_accu = 0
    best_prec = 0
    best_recall = 0
    best_f1 = 0
    best_accu_choice = [0,0,0,0]
    best_prec_choice = [0,0,0,0]
    best_reca_choice = [0,0,0,0]
    best_f1_choice = [0,0,0,0]

    stepsize = 5e-3
    for nu in np.arange(stepsize, 1, stepsize):
        for eta in np.arange(stepsize, 1, stepsize):
            for sigma in [1e-3, 3e-3, 6e-3, 1e-2, 2e-2, 4e-2, 6e-2, 8e-2, 1e-1]:
                for lam in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
                    ocn = one_class_norma(nu = nu, lam = lam, eta = eta, kernel_params = {'sigma' : sigma})
                    y_p, rhos = ocn.evaluate(ca01_feats, numpy.zeros((ca01_feats.shape[0], 1)))

                    y_p_fixed = numpy.zeros_like(y_p)
                    y_p_fixed[shuffle_inds] = y_p

                    rhos_fixed = numpy.zeros_like(rhos)
                    rhos_fixed[shuffle_inds] = rhos

                    y_hat = (y_p_fixed <= rhos_fixed).ravel()
                    
                    accu = np.sum(y_hat == y) / np.float64(y.shape[0])
                    tp = np.sum((y_hat==1) * (y==1))
                    fp = np.sum((y_hat==1) * (y==0))
                    tn = np.sum((y_hat==0) * (y==0))
                    fn = np.sum((y_hat==0) * (y==1))
                    prec = tp / np.float(tp + fp) 
                    recall = tp / np.float(tp + fn)
                    f1 = 2* prec * recall / (prec + recall)
                    if accu > best_accu:
                        best_accu = accu
                        best_accu_choice = [nu, eta, sigma, lam]
                        print "accu {} [ {} {} {} {} ]".format(accu, nu, eta, sigma, lam)
                    if prec > best_prec:
                        best_prec = prec
                        best_prec_choice = [nu, eta, sigma, lam]
                        print "prec {} [ {} {} {} {} ]".format(prec, nu, eta, sigma, lam)
                    if recall > best_recall:
                        best_recall = recall
                        best_reca_choice = [nu, eta, sigma, lam]
                        print "recall {} [ {} {} {} {} ]".format(recall, nu, eta, sigma, lam)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_f1_choice = [nu, eta, sigma, lam]
                        print "f1 {} [ {} {} {} {} ]".format(f1, nu, eta, sigma, lam)

    # synth_data = numpy.vstack((numpy.arange(9, 10, .1)[:,numpy.newaxis], 
    #                            numpy.arange(19, 20, .1)[:, numpy.newaxis]))
                       
    # y_p,rhos = ocn.evaluate(synth_data,
    #                         numpy.zeros((synth_data.shape[0], 1)))
    plt.plot(range(y_p.shape[0]), y_p_fixed)
    plt.plot(range(rhos.shape[0]), rhos_fixed)
    plt.show(block = False)
    pdb.set_trace()
    
if __name__ == '__main__':
    #pdbw.pdbwrap(main)()
    main()
        

    
