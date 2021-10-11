import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from eofs.xarray import Eof
from matplotlib import colors
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs


#### Functions #########################################################################################################
def xrFieldTimeDetrend_sst(xrda, dim, deg=1):
    # detrend along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg)
    trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])
    dt = xrda - trend
    return dt

def LinearReg(xrda, dim, deg=1):
    # liner reg along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg, skipna=True)
    return aux


def plotEofs(data, dim_reg, data_corr, dataset,
             title='', save=False, nombre_fig='fig'):
    # linear regression and %variance
    reg = LinearReg(xrda=data, dim=dim_reg)
    data_plot = reg.polyfit_coefficients[0]
    data_var_plot = xr.corr(data, data_corr, dim=dim_reg) ** 2

    # cmap
    cmap = colors.ListedColormap(['deepskyblue', 'white', 'yellow'
                                     , 'gold', 'orange', 'orangered'])
    cmap.set_over('magenta')
    cmap.set_under('blue')
    cmap.set_bad(color='black')

    fig = plt.figure(figsize=(6, 3), dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([120, 300, -15, 15], crs=crs_latlon)

    im = ax.contourf(data_plot.lon, data_plot.lat, data_plot.values, levels=np.linspace(-.3, .9, 7),
                     transform=crs_latlon, cmap=cmap, extend='both')
    ax.contour(data_plot.lon, data_plot.lat, data_plot.values, levels=np.linspace(-.3, .9, 7),
               transform=crs_latlon, colors='w', linewidths=.5)

    anom = ax.contour(data_var_plot.lon, data_var_plot.lat, data_var_plot.values * 100,
                      colors='black', levels=np.linspace(20, 80, 4),
                      linewidths=.7, transform=crs_latlon)

    plt.colorbar(im, orientation="horizontal", pad=0.2, shrink=0.60)
    ax.clabel(anom, inline=1, fontsize=8, fmt='%1.0f')
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    # ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(120, 330, 30), crs=crs_latlon)
    ax.set_yticks(np.arange(-15, 25, 5), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=6)
    plt.title(title + '-' + dataset)
    plt.tight_layout()
    if save:
        plt.savefig(nombre_fig + dataset + '.jpg')
        plt.close()
    else:
        plt.show()


########################################################################################################################
dataset = ['ERSSTv5','HadISST1.1']
v = dataset[0]

#baseline period
year0 = str(1854)
year1 = str(2021)
#------------------------------------------------------------------------------#
linear_detrend = True ## WTF!!!
# I think I am misunderstanding something.
# By removing the linear trend of the whole period with respect to its mean
# and then calculating the monthly anomalies the PC1 has the opposite sign with
# respect to when the trend is not filtered out. But PC2 is still similar,
# this means that when calculating the E and C index they give
# opposite patterns to the ones they should!

# Without removing trend, the patterns found are very similar
# to those of Takahashi et al 2011.
# (they do not clarify whether they detrend or not).
#------------------------------------------------------------------------------#

if v == dataset[0]:
    v = str('ERSSTv5')
    sst = xr.open_dataset('sst.mnmean.nc')
    # ftp://ftp2.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc


else:
    v = 'HadISST1.1'
    sst = xr.open_dataset('HadISST_sst.nc')
    #

    #rename dims
    sst = sst.rename({'latitude': 'lat'})
    sst = sst.rename({'longitude': 'lon'})

    # lons --> 0-360
    # stackoverflow:
    # Adjust lon values to make sure they are within (-180, 180)
    sst['_longitude_adjusted'] = xr.where(
        sst['lon'] < 0,
        sst['lon'] + 360,
        sst['lon'])
    # reassign the new coords to as the main lon coords
    # and sort DataArray using new coordinate values
    sst = (
        sst
            .swap_dims({'lon': '_longitude_adjusted'})
            .sel(**{'_longitude_adjusted': sorted(sst._longitude_adjusted)})
            .drop('lon'))

    sst = sst.rename({'_longitude_adjusted': 'lon'})


# change dims order for Eof function
sst = xr.DataArray(sst.sst, coords=[sst.time.values, sst.lat.values, sst.lon.values],
                   dims=['time', 'lat', 'lon'])
# domain
#sst = sst.sel(time=slice('1920-01-01', '2020-12-31'))
sst = sst.sel(lat=slice(15, -15), lon=slice(150, 280))

#linear detrend
if linear_detrend:
    sst = xrFieldTimeDetrend_sst(xrda=sst, dim='time', deg=1)
    #climatology and month anomaly
    sst_clim = sst.sel(time=slice(year0 + '-01-01', year1 + '-12-31'))
    sst = sst.groupby('time.month') - sst.groupby('time.month').mean('time', skipna=True)
else:
    # climatology and month anomaly
    sst_clim = sst.sel(time=slice(year0 + '-01-01', year1 + '-12-31'))
    sst = sst.groupby('time.month') - sst_clim.groupby('time.month').mean('time', skipna=True)

# lat weights
coslat = np.cos(np.deg2rad(sst.coords['lat'].values))
wgts = np.sqrt(coslat)[..., np.newaxis]

# EOF
if xr.__version__ == '0.19.0':
    # due to xrda.groupby('time.month') can lead
    # to problems with solver.pcs() and others
    solver = Eof(sst.drop('month'), weights=wgts, center=False)
else:
    solver = Eof(sst, weights=wgts, center=False)

eof = solver.eofs(neofs=2)

pcs = solver.pcs(npcs=2, pcscaling=0)


# % of variance
var1, var2 = np.around(solver.varianceFraction(neigs=2).values*100,1)

# projecting over the entire dataset, (following Takahashi et al 2011).
# and
#  pcs from the projected sst fields onto the EOF 1 and 2
sst = xr.DataArray(sst.values, coords=[sst.time.values, sst.lat.values, sst.lon.values],
                   dims=['time', 'lat', 'lon'])
pcs_proj = solver.projectField(sst, neofs=2, weighted=False)

# "The PCs were normalized by the standard
# deviation for the base period and smoothed with a 1‐2‐1 filter."
pcs_proj_clim = pcs_proj.sel(time=slice(year0 + '-01-01', year1 + '-12-31'))
aux = (pcs_proj - pcs_proj_clim.mean(dim='time')) / pcs_proj_clim.std('time')


# filter 1-2-1. #### binomial filter ??? ###
def filter(x):
    aux = np.convolve(x, [.25, .5, .25], mode='same')
    return aux
aux = xr.apply_ufunc(filter, aux, input_core_dims=[['time']],
                     vectorize=True, output_dtypes=[object])

# pseudo PCs (== pcs[0] [0])
pc1 = aux.values[0]
pc2 = aux.values[1]


# E-index
E = (pc1 - pc2) / np.sqrt(2)
E_aux = xr.DataArray(sst.values,
                     coords={'lat': sst.lat.values, 'lon': sst.lon.values, 'E': E},
                     dims=['E', 'lat', 'lon'])
E_aux2 = xr.DataArray(E, [('E', E)])


# C-index
C = (pc1 + pc2) / np.sqrt(2)
C_aux = xr.DataArray(sst.values,
                     coords={'lat': sst.lat.values, 'lon': sst.lon.values, 'C': C},
                     dims=['C', 'lat', 'lon'])
C_aux2 = xr.DataArray(C, [('C', C)])


# PC1
PC1_aux = xr.DataArray(sst.values,
                       coords={'lat': sst.lat.values, 'lon': sst.lon.values, 'PC': pc1},
                       dims=['PC', 'lat', 'lon'])
PC1_aux2 = xr.DataArray(pc1, [('PC', pc1)])


# PC2
PC2_aux = xr.DataArray(sst.values,
                       coords={'lat': sst.lat.values, 'lon': sst.lon.values, 'PC': pc2},
                       dims=['PC', 'lat', 'lon'])
PC2_aux2 = xr.DataArray(pc2, [('PC', pc2)])


# linear reg and plot.
plotEofs(data=E_aux, dim_reg='E', data_corr=E_aux2, save=False,
         nombre_fig='E-index' + year0 + '_' + year1,
         title='E-pattern - ' + year0 + '_' + year1, dataset=v)

plotEofs(data=C_aux, dim_reg='C', data_corr=C_aux2, save=True,
         nombre_fig='C-index' + year0 + '_' + year1,
         title='C-pattern - ' + year0 + '_' + year1, dataset=v)

plotEofs(data=PC1_aux, dim_reg='PC', data_corr=PC1_aux2, save=False,
         nombre_fig='eof1' + year0 + '_' + year1,
         title='EOF1 '  + year0 + '_' + year1+ ' - '+ str(var1) + '%', dataset=v)

plotEofs(data=PC2_aux, dim_reg='PC', data_corr=PC2_aux2, save=False,
         nombre_fig='eof2' + year0 + '_' + year1,
         title='EOF2 '  + year0 + '_' + year1+ ' - '+ str(var2) + '%', dataset=v)

#\m/
