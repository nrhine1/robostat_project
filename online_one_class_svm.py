#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pdb

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
        
        eta =  self.nu * 1.0 / self.t # TODO learning rate

        self.w -= eta * grad_w
        self.rho -= eta * grad_rho
        
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
            
def main():
    nu = 0.1

    ca01_feats = numpy.load('feats/ca05_no_label_bov.npz')['arr_0'][15:]

    # ca01_feats = numpy.vstack((ca01_feats, ca01_feats))
    oocs = online_one_class_svm(nu, ca01_feats.shape[1])

    y_p = oocs.evaluate(ca01_feats, numpy.zeros((ca01_feats.shape[0], 1)))

    plt.plot(range(y_p.shape[0]), y_p)
    plt.show(block = False)
    pdb.set_trace()
    
if __name__ == '__main__':
    main()
        

    
