
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import inv, det
from numpy import exp, pi, sqrt
from scipy.stats import norm


def make_data():
    data = np.random.normal(loc=[0,6], scale=[2,1], size=(6,2))
    x = data.T.ravel()[:-2]
    x -= x.min()-1
    y = np.array([0]*6 + [1]*4, dtype="uint8")
    return(x,y)


def visualize_points_and_their_gaussians(x,y):
    for k in sorted(set(y)):
        color = ["g","orange"]
        mu,sd = x[y==k].mean(), x[y==k].std(ddof=0)
        pdf = norm(mu,sd).pdf
        xx = np.linspace(x[y==k].min()-10, x[y==k].max()+10, 100)
        yy = pdf(xx)
        mask = yy > 0.001
        xx,yy = (a[mask] for a in (xx,yy))
        plt.plot(xx,yy, color='#999999')
        plt.fill(xx,yy, alpha=0.5, color=color[k])
        plt.plot(x[y==k], [0]*len(x[y==k]), 'o', color=color[k], mec='#999999', ms=8)
    return(plt.gca())
    
##############################################################################


x,y = make_data()
sp = visualize_points_and_their_gaussians(x,y)

