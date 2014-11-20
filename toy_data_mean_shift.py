import numpy as np
import pdb
import numpy.random as rnd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from online_meanshift import online_meanshift3

#def DisplayClrPts(fout, locs, yhat, node_labels, classes):
#    clrs = np.array(('g','k','b','brown','y'))
#
#    K = classes.shape[0]
#
#    fig = plt.figure()
#    ax = plt.subplot(121, projection='3d')
#    for ci in range(K):
#        idxs = node_labels==classes[ci]
#        xs = locs[idxs,0]
#        ys = locs[idxs,1]
#        zs = locs[idxs,2]
#        ax.scatter(xs, ys, zs, c=clrs[ci], marker='o', edgecolors='none', label='hi!')
#    ax.set_xlabel('X axis')
#    ax.set_ylabel('Y axis')
#    ax.set_zlabel('Z axis')
#    ax.view_init(azim=180)
#    plt.title('Title')
#    
#    ax.view_init(azim=0)
#    plt.savefig(fout)

def Display3DPts(ax, locs,clr='b'):
    xs = locs[:,0]
    ys = locs[:,1]
    zs = locs[:,2]
    path_collection = ax.scatter(xs, ys, zs, c=clr, marker='o', edgecolors='none')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(azim=180)
#     plt.title('Title')
    
    ax.view_init(azim=0)
    return ax, path_collection


mu1 = np.array([1, 0, 0])
Sigma1 = np.eye(mu1.shape[0],mu1.shape[0]) * 0.01
N1 = 100
mu2 = np.array([0, 1, 0])
Sigma2 = np.eye(mu1.shape[0],mu1.shape[0]) * 0.01
N2 = 100
mu3 = np.array([0, 0, 1])
Sigma3 = np.eye(mu1.shape[0],mu1.shape[0]) * 0.01
N3 = 20

class1 = rnd.multivariate_normal(mu1, Sigma1, N1) # Normal class seen 1st
class2 = rnd.multivariate_normal(mu2, Sigma2, N2) # Normal class seen 2nd
class3 = rnd.multivariate_normal(mu3, Sigma3, N3) # Anomaly class

toy_lbls = np.vstack((1*np.ones((N1/2,1)), 
                      3*np.ones((1,1)), 
                      2*np.ones((N2/2,1)), 
                      1*np.ones((N1/2,1)), 
                      2*np.ones((N2/2,1)), 
                      3*np.ones((N3-1,1)))).astype(int) - 1

toy_feats = np.vstack((class1[:(N1/2),:], 
                       class3[1,:], 
                       class2[:(N2/2),:], 
                       class1[(N1/2):,:], 
                       class2[(N2/2):,:], 
                       class3[1:,:]))

print toy_lbls.shape, toy_feats.shape
assert(toy_lbls.shape[0] == toy_feats.shape[0])

oms = online_meanshift3(3, 25, 10)


fig = plt.figure()
ax = plt.subplot(111, projection='3d')



clrs = np.array(('g','b','r','k','brown','y'))
N = toy_feats.shape[0]
for i in range(N):
    print "fitting iteration {}/{}".format(i,N)
    oms.fit(toy_feats[i])

    ax, pc = Display3DPts(ax, toy_feats[np.newaxis,i,:], clrs[toy_lbls[i]])
    
    print "number of modes: {}".format(len(oms.modes))
    for mode in oms.modes:
        ax, pc = Display3DPts(ax, mode[np.newaxis, :], clrs[3])

    plt.savefig('figs/{}.jpg'.format(i))

    pc.remove()
    plt.draw()

plt.show()

print "Done!"
