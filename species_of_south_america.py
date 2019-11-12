
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import sklearn.datasets
import sklearn.neighbors

#get data
bunch = sklearn.datasets.fetch_species_distributions()
nd = bunch.train   # structured array
tp = nd.dtype      # dtype([('species', 'S22'), ('dd long', '<f4'), ('dd lat', '<f4')])

#arrange data
latitudes = nd["dd lat"]
longitudes = nd["dd long"]
X = np.vstack([latitudes, longitudes]).T
y = np.array([bt.startswith(b'micro') for bt in nd['species']], dtype='int')  # species: {0,1}

#figure
fig, ax = plt.subplots(1,3, figsize=(10,5))
fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98)

"""PLOT 1"""
xgrid, ygrid = sklearn.datasets.species_distributions.construct_grids(bunch)  # xrange, yrange
sp = ax[0]
m = Basemap(projection='cyl', resolution='c', ax=sp,
            llcrnrlat=ygrid.min(), llcrnrlon=xgrid.min(),
            urcrnrlat=ygrid.max(), urcrnrlon=xgrid.max())
m.drawmapboundary(fill_color="#DDEEFF")
m.fillcontinents(color='#DDEEFF')
m.drawcoastlines(color='gray', zorder=2)
m.drawcountries(color='gray', zorder=2)
m.scatter(X[:,1], X[:,0], c=y, lw=0.1, cmap=plt.cm.RdBu_r, marker='.', zorder=3, latlon=True)
sp.plot(0,0, color="navy", marker='.', linestyle='none', label="Bradypus Variegatus")
sp.plot(0,0, color="darkred", marker='.', linestyle='none', label="Microryzomys Minutus")
sp.legend(loc=4, fontsize="x-small", frameon=False, framealpha=0.8)


"""PLOT 2 & 3"""
#grid and mask
mx = bunch.coverages[6][::5, ::5]  # 319x243 mx of target-data (feature 6) only used here for land reference
mask = (mx > -9999).ravel()        # land mask (-9999 indicates ocean)
XX,YY = np.meshgrid(xgrid[::5], ygrid[::5][::-1])
Xtest = np.c_[YY.ravel(), XX.ravel()]
Xtest = np.radians(Xtest[mask])

species_names = ['Bradypus Variegatus', 'Microryzomys Minutus']
colours = ['Purples', 'Reds']

X = np.radians(X)

for i,sp in enumerate(ax[1:]):
    sp = ax[i+1]
    sp.set_title(species_names[i])
    
    m = Basemap(projection='cyl', resolution='c', ax=sp,
                llcrnrlat=YY.min(), llcrnrlon=XX.min(),
                urcrnrlat=YY.max(), urcrnrlon=XX.max())
    m.drawmapboundary(fill_color='#DDEEFF')
    m.drawcoastlines()
    m.drawcountries()
    
    #2d KDE (spherical i.e. haversine)
    md = sklearn.neighbors.KernelDensity(bandwidth=0.03, metric='haversine')
    md.fit(X[y==i])
    
    #
    logs = md.score_samples(Xtest)
    ypred = np.exp(logs)
    Z = np.full(shape=mask.shape[0], fill_value=-9999.0)
    Z[mask] = ypred
    ZZ = Z.reshape(XX.shape)
    
    #plot
    levels = np.linspace(0, ZZ.max(), 25)
    m.contourf(XX,YY,ZZ, levels=levels, cmap=colours[i])



