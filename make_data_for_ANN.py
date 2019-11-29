
#IN PROGRESS
import matplotlib.pyplot as plt
"""
function generating synthetic multidimensional data for ANN by forward propagation

the multidimensional space is (linearly) filled with data-points
the weights-matreces are filled with values from standard normal distribution 
the data is forward-propagated through the weights

"""



import numpy as np

def avg_entropy(p): #prefer lower entropy
    from numpy import log2 as log, dot
    return -dot(p, log(p)) / log(len(p))



#########################################################




m = 10000
n = 2
K = 10
L = 3
u = 32
gmm = True
seed = 42


shapes = (n,) + (u,)*L + (K,)

#np.random.seed(seed)

#create weights
g = zip(shapes[1:], shapes[:-1])  #shape-tuples
WW = [np.random.normal(loc=0, scale=1, size=shape) for shape in g]



#create data
X = np.random.uniform(-1,1, size=(m,n))





#forward propagation
def softmax(Z):
    return np.exp(Z) / np.exp(Z).sum(0, keepdims=True)

A = X.T
for l,W in enumerate(WW):
    Z = np.matmul(W,A) 
    A = (np.tanh if l<len(WW)-1 else softmax)(Z)
P = A.T    
y = P.argmax(1)
print(np.bincount(y))


if n == 2:
    plt.figure()
    plt.scatter(*X.T, c=y, s=5, cmap='rainbow')


#GMM (the separability is usually better with GMM)
if gmm:
    print("WITH GMM")
    default_n_gaussians = 25
    n_gaussians = ((abs(int(gmm))-1) or default_n_gaussians-1)+1
    classes = sorted(set(y))
    from sklearn.mixture import GaussianMixture
    MD = [GaussianMixture(n_components=n_gaussians, covariance_type='spherical').fit(X[y==c]) for c in classes]
    mm = [X[y==c].shape[0] for c in classes]
    y = sum([[k]*m for k,m in zip(range(len(classes)),mm)], [])
    X = np.vstack([md.sample(m)[0] for md,m in zip(MD,mm)])


print(X.shape, "<<<SHAPE")



if n == 2:
    plt.figure()
    plt.scatter(*X.T, c=y, s=5, cmap='rainbow')



from sklearn.neural_network import MLPClassifier
md = MLPClassifier((64,64,64))
md.fit(X,y)
acc = md.score(X,y)
print(acc)



