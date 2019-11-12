
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sklearn.neighbors

path = r"~/Downloads/germany.csv"
df = pd.read_csv(path)                   # lat, lon

fig = plt.figure(figsize=(10,5))
bm = Basemap(projection="cyl", resolution='l',
             llcrnrlat=47, llcrnrlon=5,
             urcrnrlat=56, urcrnrlon=16)
bm.drawcoastlines()
bm.drawcountries(linewidth=2)

#this is for the city name allignement
x1min = ((bm.xmax-bm.xmin) / (bm.boundarylonmax - bm.boundarylonmin)) / 60
y1min = ((bm.ymax-bm.ymin) / (bm.boundarylats[1]-bm.boundarylats[0])) / 60
markersize = 2

for town,lat,lon,pop in df.iloc[:, [0,1,2,7]].values:
    if np.isnan(np.float(pop)):     
        bm.plot(lon,lat, 'b.', markersize=markersize)
    else: 
        plt.text(lon+6*x1min, lat-y1min, town, fontsize=6, va='center', ha='left')
        bm.plot(lon,lat, 'b.', markersize=np.log10(pop)*(markersize/2), alpha=0.6)


"""2D haversine KDE"""
#θ = sklearn.neighbors.DistanceMetric.get_metric("haversine").pairwise(mx)[0,1]
#data must be in the form of [latitude, longitude] and both inputs and outputs are in units of radians
#fit    
md = sklearn.neighbors.KernelDensity(bandwidth=0.005, kernel="gaussian", metric="haversine")
X = np.radians(df.iloc[:, [1,2]].values)   # lat,lon
md.fit(X)

#grid and land-mask
rx, ry = np.linspace(bm.xmin, bm.xmax, 100), np.linspace(bm.ymin, bm.ymax, 100)
XX,YY = np.meshgrid(rx,ry)
Xtest = np.c_[YY.ravel(), XX.ravel()]
mask = np.array([bm.is_land(x,y) for y,x in Xtest])

#predict
logs = md.score_samples(np.radians(Xtest[mask]))
ypred = np.exp(logs)
Z = np.full(shape=len(mask), fill_value=0.0)
Z[mask] = ypred
ZZ = Z.reshape(XX.shape)

levels = np.linspace(0, ZZ.max(), 50)
bm.contourf(XX,YY,ZZ, cmap=plt.cm.Purples)
