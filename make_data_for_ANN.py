#CONTINUE

"""
function that creates multidimensional data for ANN
(the multidimensional space is filled with data-points and split; border-line data-points are removed 
  (or rather moved to the center of the respective class))

"""



#CONTINUE
"""
make synthetic dataset function for NN
"""

import matplotlib.pyplot as plt

import sklearn
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

bunch = load_digits()
X = bunch.data
y = bunch.target

X = MinMaxScaler().fit_transform(X)

md = MLPClassifier((128,64,64))
md.fit(X,y)

acc = md.score(X,y)
print(acc)

plt.hist(md.coefs_[-1].ravel())



#############
import numpy as np, matplotlib.pyplot as plt
m = 1000
a1 = np.random.normal(loc=0, scale=0.15, size=m)
a2 = np.random.uniform(-1,1, size=m)
plt.hist((a1+a2)/2)

###################
#prefer lower entropy (in each observation's target)

def entropy(p):
    from numpy import log2 as log, dot
    assert sum(p)==1.0,"bad probabilities"
    entropy = -dot(p, log(p))
    return entropy


def avg_entropy(p):
    from numpy import log2 as log, dot, array, float128, allclose
    nd = array(p, dtype=float128)
    print(nd, nd.sum())
    #assert allclose(nd, 1.0),"bad probabilities"
    return -dot(nd, log(nd)) / log(len(p))



p = (.12, .18, .28, .42)

p = (.25, .25, .25, .25)
p = (.5, .5)
p = (1/3, 1/3, 1/3)
p = (.1, .1, .1, .1, .1, .1, .1, .1, .1, .1)

from decimal import Decimal
d = Decimal('0.1')
p = [d,]*10

d = Decimal('0.33333333333333333333333333333333333333333333333')
p = [d,]*3


#H = entropy(p)
#print(H)

p = (.12, .18, .28, .42)
p = (.1, .2, .3, .4)
p = (.1, .1, .1, .1, .6)

ae = avg_entropy(p)
print(ae)





