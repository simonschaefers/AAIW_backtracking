########################################################################################################
# Author: Simon Schäfers
# Contact: sschaefers@geomar.de
# Date: 2025-12-19

########################################################################################################

# This script contains helper functions for plotting the results of particle tracking experiments.
# The plotting functions can be found in the plotter.py script.
#=======================================================================================================

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from scipy.optimize import leastsq
import cartopy.crs as ccrs
import matplotlib.path as mpath
from matplotlib.colors import LogNorm,Normalize
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerBase

# plot presets
class PlotPreset:
    '''
    Container for plot presets
    '''
    def __init__(self):
        # figure size
        self.fs = (20, 7)
        # font sizes
        self.fd = {
            'header': {'fontsize': 20, 'fontweight': 'bold'},
            'ticks': {'fontsize': 20, 'fontweight': 'bold'},
            'legend': 20,
            'title': {'fontsize': 24, 'fontweight': 'bold'},
            'suptitle': {'fontsize': 30, 'fontweight': 'bold'},
            'abc':20,
            }

class Experiment:
    '''
    Class to store experiment metadata and make them accessible for plotting
    Method set_default() sets the metadata for the ED04 experiment

    ## Attributes
    - name: str, name of the experiment
    - pfile: str, path to the particle data file (.csv)
    - zfile: str, path to the file containing the trajectory bins (.zarr)
    - sfile: str, path to the start conditions file (.csv)
    - color: str, color for the experiment
    - ls: str, linestyle for the plot
    - lw: int, linewidth for the plot
    - cmap: str, colormap for the plot
    '''
    def __init__(self, name = None, pfile = None,zfile = None,sfile = None,**kwargs):
        self.name = name
        self.pfile = pfile
        self.zfile = zfile
        self.sfile = sfile
        self.velfiles = [None,None,None]
        self.color = kwargs.get('color', 'tab:blue')
        self.ls = kwargs.get('ls', '-')
        self.lw = kwargs.get('lw', 2)
        self.cmap = kwargs.get('cmap', 'cmo.deep')
        self.date = kwargs.get('date','0815')

    def set_default(self):
        self.name = 'ED04'
        folder = 'zenodo_folder/metadata/'
        self.pfile = folder+'EDDY_subduction.csv'
        self.zfile = folder+'EDDY_pathways.zarr'
        self.sfile = folder+'start_conditions.csv'

def set_share_axes(axs, target=None, sharex=False, sharey=False,own = False):
    ''' 
    Subroutine required for the oririn_plain rotine to split axes
    '''
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_axes['x'].join(target, ax)
        if sharey:
            target._shared_axes['y'].join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)

def plot_fronts(ax,**kwargs):
    '''
    Subroutine to plot the fronts in the Southern Ocean
    used in the polish function

    ! does not work without local files, see below !

    '''
    main_folder = '/work/uo0780/u301534/Masterarbeit/'
    # these files are not contained in the zenodo upload due to size restrictions
    fronts = xr.open_dataset(main_folder+'POP_TEMP_2004.nc')
    grid = xr.open_dataset(main_folder+'Data_velocities/10y_0101_u.nc')
    ax.contour(grid.XU[:1800],grid.YU[:800],fronts.TEMP[13,:800,:1800],levels = [2,5],**kwargs)
    ax.contour(grid.XU[1800:]-360,grid.YU[:800],fronts.TEMP[13,:800,1800:],levels = [2,5],**kwargs)

def plot_ventilations(df,ax,mask,weights,
                          bin_size=1,cmap = 'viridis',**kwargs):
    '''
    Subroutine to plot the ventilation histogram, used in the origin_plain routine
    ## Parameters:
    - df: pandas DataFrame, particle data
    - ax: matplotlib axis, axis to plot on
    - mask: array-like, boolean mask to select particles
    - weights: array-like, weights for the histogram
    '''

    vent_bins = [np.arange(0,360+bin_size,bin_size),np.arange(-75,-20,bin_size)]
    # plot lon, lat of ventilations
    hist_values = np.histogram2d(df['lon'][mask],df['lat'][mask],bins = vent_bins,weights=weights,density=False)
    hist_values[0][hist_values[0]<=2*(bin_size**2)] = np.nan
    hv = hist_values[0].T/(bin_size**2)/np.nansum(hist_values[0])
    cmap = plt.get_cmap(cmap, 6)
    cmap.set_over('black')
    norm = Normalize(vmax =1e-3,vmin = 0)
    kwargs['transform'] = ccrs.PlateCarree()
    img = ax.pcolor(hist_values[1][1:]+hist_values[1][-1]-1,hist_values[2][1:]-1,hv,
            cmap = cmap,norm = norm,**kwargs)

    return img


def get_boxes(version = 'DAIP'):
    ''' 
    box selector
    '''

    if version in ['DAIPA',True]:
        box1 = [-90,-40,-65,-40]
        box2 = [-55,0,-40,-20]
        box3 = [60,150,-50,-20]
        box4 = [150,270,-55,-35]
        box5 = [0,30,-50,-30]
        boxes = [box1,box2,box3,box4,box5]
        names = ['Drake Passage','South Atlantic','South Indian Ocean','South Pacific','Agulhas Retroflection']
        short_names = ['DP','SA','SI','SP','AR']
    elif version == 'DAIP':
        boxes = [
        [250,320,-70,-45],        
        [-60,30,-45,-20],
        [40,180,-55,-20],
        [180,240,-70,-35],
        ]
        names = ['Drake Passage','South Atlantic','South Indian Ocean','South Pacific']
        short_names = ['DP','SA','SI','SP']

    elif version == 'DAIP_new':
        boxes = [
        [-90,-40,-65,-45],
        [-60,30,-45,-20],
        [60,150,-50,-20],
        [150,270,-55,-35],
        ]
        names = ['Drake Passage','South Atlantic','South Indian Ocean','South Pacific']
        short_names = ['DP','SA','SI','SP']

    elif version == 'c/w':
        t = 400
        boxes = [[t],
                 [-t],
                 [-t,t]]
        names = ['Cold Route','Warm Route','Local Waters']
        short_names = ['cold','warm','local']
    else:
        raise ValueError('version not known')
    return pd.DataFrame({'box':boxes,'name':names,'sname':short_names},index = np.arange(len(boxes)))

def plot_box(box,ax,text = '',**kwargs):
    '''
    box plotter
    '''
    if len(box)!= 4:
        return 'Routes not yet plottable'
    if box[0] > box[1]:
        print('splitting box')
        lon1 = np.arange(box[0],361,1)
        lon2 = np.arange(0,box[1]+1,1)
        
        ax.plot(lon1,np.ones_like(lon1)*box[2],**kwargs)
        ax.plot(lon1,np.ones_like(lon1)*box[3],**kwargs)
        ax.plot(lon2,np.ones_like(lon2)*box[2],**kwargs)
        ax.plot(lon2,np.ones_like(lon2)*box[3],**kwargs)
        box[0] -= 360
    else:
        lon = np.arange(box[0],box[1]+1,1)
        ax.plot(lon,np.ones_like(lon)*box[2],**kwargs)
        ax.plot(lon,np.ones_like(lon)*box[3],**kwargs)
    ax.plot([box[0],box[0]],[box[2],box[3]],**kwargs)
    ax.plot([box[1],box[1]],[box[2],box[3]],**kwargs)
    kwargs.pop('linewidth',None)
    ax.text((np.mean(box[0:2])-15+360),box[3]+8,text,va = 'center',ha = 'center',fontsize = 17,fontweight = 'bold',**kwargs)

def boxmask(df,box):
    '''
    Selector for particles based on the chosen boxes
    '''

    if box['sname'] in ['cold','warm','local']:
        thresh = 2000
        home = ((df['lon'] > 300) | (df['lon'] < 30)) & (df['lat'] > -45) & (abs(df['dU']*5*24*3600*1e-3)< thresh) 
        c = (df['dU']*5*24*3600*1e-3>500) & ~home
        w = (df['dU']*5*24*3600*1e-3<-500) & ~home
        if box['sname'] == 'cold':
            return c
        elif box['sname'] == 'warm':
            return  w
        elif box['sname'] == 'local':
            return ~(c|w) 
        
    if type(box) != list:
        box = box['box']
    for i in [0,1]:
        box[i] += 360 if box[i] < 0 else 0
    if box[0]>box[1]:
        lon_mask = (df['lon'] > box[0]) | (df['lon'] < box[1])
    else:
        lon_mask = (df['lon'] > box[0]) & (df['lon'] < box[1])
    
    return lon_mask & (df['lat'] > box[2]) & (df['lat'] < box[3])


def check_exp(exp):
    '''
    Check if exp is a list of Experiment objects
    '''
    if type(exp) != list:
        exp = [exp]
    for e in exp:
        assert type(e) == Experiment, 'all exps must be of type Experiment'
    return exp

def get_mask(sdf,mask_kw = 'density>27'):
    '''
    Subselector of the particle set.
    For the publication, 'density>27' is used
    '''

    mask = np.ones(len(sdf),dtype = bool)
    if mask_kw == None: return mask
    if type(mask_kw) == str: mask_kw = [mask_kw]

    for kw in mask_kw:
        if kw == 'lat<-35': mask &= (sdf['lat']<-35)
        elif kw == 'density>27': mask &= (sdf['density']>27)
        elif kw == 'density<27': mask &= (sdf['density']<27)
        else: raise ValueError('mask_kw not recognized')
    return mask


def polish(ax,polar = True,yl = -19.5,p = PlotPreset(),fronts = True):
    '''
    polish function to set up the map axes, which are in SouthPolarStereo projection.

    ## Parameters:
    - ax: matplotlib axis, axis to polish
    - polar: bool, if True, the plot is in polar projection
    - yl: float, latitude limit for the y-axis
    - p: PlotPreset, plot preset to use
    - fronts: bool, if True, plot the fronts, requires local files

    '''

    ax.coastlines()
    ax.set_extent([-180, 180, -90 if polar else -70, yl], crs=ccrs.PlateCarree())
    if polar:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                linewidth=2, color='gray', alpha=1, linestyle='--',
                rotate_labels=False,ylocs = [-60,-40,-20])
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                linewidth=2, color='gray', alpha=1, linestyle='--',
                ylocs = [-60,-20],xlocs = [])
        
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        for lon in range(-180, 180, 120):
            x, y = ax.projection.transform_point(lon, yl+3.5, ccrs.PlateCarree())
            dd = '  '+str(abs(lon)) + '°' + ['W', 'E'][lon >= 0]
            if lon in [-180, 0]:
                dd = '  '+str(abs(lon)) + '°'
            ax.text(x, y,dd, transform=ccrs.SouthPolarStereo(),
                    ha='center', va='center', fontdict=p.fd['ticks'],
                    rotation=-lon if lon != -180 else 0)
    
    else:
        gl = ax.gridlines(draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--',
                        ylocs = [-60,-40,-20],xlocs = np.arange(-120,121,60))
        gl.top_labels,gl.right_labels = False,False
    if fronts:   
        plot_fronts(ax,transform = ccrs.PlateCarree(),colors = 'r',alpha = .8,linewidths = 1.5)
    gl.ylabel_style = {'size': p.fd['ticks']['fontsize'],'weight':p.fd['ticks']['fontweight']}
    return gl


def load_velocity(velfiles,depth = 500,frac = None):
    '''
    Load velocity data from netcdf files
    
    ! local files are required, which are not in the zenodo upload due to size restrictions !
    ## Parameters:
    - velfiles: list of str, paths to the velocity files [u_file,v_file,w_file]
    - depth: float, depth to load the velocities at
    - frac: int, if not None, downsample the data by this factor
    ## Returns:
    - u: xarray DataArray, zonal velocity at the specified depth
    - v: xarray DataArray, meridional velocity at the specified depth
    - w: xarray DataArray, vertical velocity at the specified depth
    - grid: xarray DataArray, grid cell area
    '''
    u = xr.open_dataset(velfiles[0])['UVEL'][0]
    v = xr.open_dataset(velfiles[1])['VVEL'][0]

    d = np.argmin(np.abs(u['DEPTH_T'].values-depth))
    print(u.DEPTH_T.values[d])
    u = u.isel(DEPTH_T = d)
    v = v.isel(DEPTH_T = d)
    w = xr.open_dataset(velfiles[2])['WVEL'][0]
    w = w.isel(W_DEP= d)
    #w = w.isel(W_DEP= [d,d+1]).mean('W_DEP')

    grid = xr.open_dataset('/work/uo0780/u301534/Masterarbeit/gridarea.nc')['cell_area']

    latmin = -70    
    latmax = -20
    lmi = np.argmin(np.abs(u['YU'].values-latmin))
    lma = np.argmin(np.abs(u['YU'].values-latmax))
    u = u.isel(YU = slice(lmi,lma))/100
    v = v.isel(YU = slice(lmi,lma))/100
    w = w.isel(YU = slice(lmi,lma))/100
    grid = grid.isel(YU = slice(lmi,lma))

    if frac:
        u = u[::frac,::frac]
        v = v[::frac,::frac]
        w = w[::frac,::frac]
        grid = grid[::frac,::frac]
    return u,v,w,grid
        

def find_coord(ds,keys = ['XU','YU','depth_t'],lon = None,lat = None,depth = None):
    '''
    helper function to find the nearest index for a given lon, lat, depth in a xarray dataset
    '''


    tmp = [None,None,None]
    if lon is not None:
        # get lon range
        if ds[keys[0]].values[0] <-150:
            if lon > 180:
                lon = lon - 360
        elif ds[keys[0]].values[0] >=0:
            if lon < 0:
                lon = lon + 360
        tmp[0] =  np.argmin(abs(ds[keys[0]].values - lon))
    if lat is not None:
        tmp[1] =  np.argmin(abs(ds[keys[1]].values - lat))
    if depth is not None:
        tmp[2] =  np.argmin(abs(ds[keys[2]].values - depth))
    return tmp

########### Support classes for legend handling ##############
class AnyObjectHandler(HandlerBase):
    def __init__(self, linestyle,color, **kw):
        HandlerBase.__init__(self, **kw)
        self.ls = linestyle
        self.color = color

    def create_artists(self, legend, orig_handle,
                    x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], 
            color=self.color[0], linestyle=self.ls[0])
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], 
            color=self.color[1], linestyle=self.ls[1])
        return [l1, l2] 

class SingleObjectHandler(HandlerBase):
    def __init__(self, linestyle,color, **kw):
        HandlerBase.__init__(self, **kw)
        self.ls = linestyle
        self.color = color

    def create_artists(self, legend, orig_handle,
                    x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], 
            color=self.color, linestyle=self.ls,lw = 2)

        return [l1] 

def gaussian_fit(xdata,ydata,ax,mass = 1,vals = 1,dt = 1,free = False,**kwargs):
    '''
    Inverse Gaussian fit to a histogram. 

    ## Parameters:
    - xdata: array-like, bin edges
    - ydata: array-like, histogram values
    - ax: matplotlib axis, axis to plot on
    - mass: float, mass parameter for the fit
    - vals: float, initial guess for the fit
    - dt: float, time step for the fit
    - free: bool, if True, mass is a free parameter

    ## Returns:

    
    '''
    if free:
        fitfunc = lambda p, x: p[2]*dt* np.sqrt(p[0]**3/(4*np.pi*p[1]**2*x**3))*np.exp(-p[0]*(x-p[0])**2/(4*x*p[1]**2)) 
    else:
        fitfunc = lambda p, x: mass*dt* np.sqrt(p[0]**3/(4*np.pi*p[1]**2*x**3))*np.exp(-p[0]*(x-p[0])**2/(4*x*p[1]**2)) 
    errfunc  = lambda p, x, y: (y - fitfunc(p, x))
    
    c = leastsq(errfunc,  [vals,5,mass], args=(xdata[:-1], ydata),maxfev=100000)[0]

    label = '$\Gamma$  = %.1f y\n$\Delta/\Gamma$ = %.3f'%(c[0],c[1]/c[0])
    lines, = ax.plot(xdata[:-1],fitfunc(c,xdata[:-1]),'-',label = label,**kwargs)
    return lines,label


def get_age_grid(exp,bs = 1,mask_kw = 'densoty>27'):
    '''
    Support function to get the age grid for the experiments
    used in the age_map routine

    ## Parameters:
    - exp: list of Experiment objects, experiments to process
    - bs: int, bin size in degrees
    - mask_kw: str, keyword for the mask to apply to the particles

    ## Returns:
    - exp_grid: list of arrays, age grids for each experiment
    - lons: array-like, longitude bin edges
    - lats: array-like, latitude bin edges

    '''

    exp_grid = []
    for e in exp:
        df = pd.read_csv(e.pfile)
        sdf = pd.read_csv(e.sfile)
        Mask = (df['transit_time']>0) & get_mask(sdf,mask_kw)

        lons = np.arange(0,361,bs)
        lats = np.arange(-70,-21,bs)
        meangrid = np.zeros((len(lons)-1,len(lats)-1))
        stdgrid = np.zeros((len(lons)-1,len(lats)-1))
        Nogrid = np.zeros((len(lons)-1,len(lats)-1))
        for i in range(len(lons)-1):
            for j in range(len(lats)-1):
                mask = Mask & (df['lon'] >= lons[i]) & (df['lon'] < lons[i+1]) & (df['lat'] >= lats[j]) & (df['lat'] < lats[j+1])
                if mask.sum() <2*bs**2:
                    meangrid[i,j] = np.nan
                    stdgrid[i,j] = np.nan
                    Nogrid[i,j] = np.nan
                else:
                    meangrid[i,j] = (df['transit_time'][mask]/365-sdf['start_time'][mask]/365).mean()
                    stdgrid[i,j] = (df['transit_time'][mask]/365-sdf['start_time'][mask]/365).std()
                    Nogrid[i,j] = mask.sum()
        exp_grid.append([meangrid,stdgrid,Nogrid])
    return exp_grid,lons,lats

