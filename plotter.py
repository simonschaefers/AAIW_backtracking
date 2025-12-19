########################################################################################################
# Author: Simon Schäfers
# Contact: sschaefers@geomar.de
# Date: 2025-12-08

########################################################################################################

# This script contains functions to visualize various aspects of particle advection experiments, as seen
# in the manuscript 
# "Mesoscale Eddies Enhance the Ventilation of Antarctic Intermediate Water in the South Atlantic"


import string
import zarr
import gsw
from glob import glob
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import cmocean as cmo
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from _plotter_helper import *
main_folder = '/work/uo0780/u301534/Masterarbeit/'

def AAIW(exp,p = PlotPreset(),mask_kw = 'density>27',lon = 335,complete = True,**kwargs):
    '''
    Plots a comparison of salinity and temperature from the 2004 POP simulation and
    the WAGHC climatology along a zonal section at given longitude.
    
    ! files required for plotting are not contained in the zenodo upload due to size limitations !

    ## Parameters:
    - exp: Experiment object, containing information about the experiment (not used here)
    - p: PlotPreset object, containing plot presets
    - mask_kw: str or list of str, keywords to define the density range for AAIW
    - lon: int, longitude of the zonal section (in degrees east)
    - complete: bool, if True, plots also the incomplete AAIW density contour

    ## Returns:
    - fig: matplotlib Figure object
    - AX: array of matplotlib Axes objects
    '''


    if type(mask_kw) != list:
        mask_kw = [mask_kw]
    plats = [-40,-30]
    dense = [27,27.43] if 'density>27' in mask_kw else [26.82,27.43]
    if 'density>27' not in mask_kw: complete = False
   
    grid = xr.open_dataset(main_folder+'gridarea.nc')
    pop_s = xr.open_dataset(main_folder+'POP_SALT_2004.nc')['SALT']
    pop_t = xr.open_dataset(main_folder+'POP_TEMP_2004.nc')['TEMP']
    woce = xr.open_dataset(main_folder+'WAGHC_BAR_ALL.nc')
    
    fig,AX= plt.subplots(2,2,figsize = (20,8),constrained_layout=True,sharex = True,sharey = True)

    # get limits and coordnate indices
    lims = (lon,[-70,-10],2200)
    pop_lim = find_coord(grid,lon = lims[0],lat = lims[1][1])
    pop_lim[2] = find_coord(pop_s,depth = lims[2])[2]
    woce_lim = find_coord(woce,keys = ['longitude','latitude','depth'],
                          lon = lims[0],lat = lims[1][1],depth = lims[2])

    min_lat_pop = find_coord(grid,lat = lims[1][0])[1]
    min_lat_woce = find_coord(woce,keys = ['','latitude',''],lat = lims[1][0])[1]

    # extract data
        # WOCE
    woce_s = woce['salinity'][0,:woce_lim[2],min_lat_woce:woce_lim[1],woce_lim[0]]
    woce_t = woce['temperature'][0,:woce_lim[2],min_lat_woce:woce_lim[1],woce_lim[0]]
    woce_dens = gsw.sigma0(woce_s,woce_t)
    woce_dims = [woce_s['latitude'],woce_s['depth']]
        # POP
    pop_s = pop_s[:pop_lim[2],min_lat_pop:pop_lim[1],pop_lim[0]]*1000
    pop_t = pop_t[:pop_lim[2],min_lat_pop:pop_lim[1],pop_lim[0]]
    pop_dens = gsw.sigma0(pop_s,pop_t)
    pop_dims = [grid['YU'][min_lat_pop:pop_lim[1]],pop_s['depth_t']]

    dims = [pop_dims,woce_dims]
    dens = [pop_dens,woce_dens]
    vals = [[pop_s,pop_t],[woce_s,woce_t]]
    
    # set plotting parameters
    cmaps = ['cmo.haline','cmo.thermal']
    lvlss = [np.linspace(34,36,11),np.linspace(0,20,11)]

    # plot
    for i in range(2):
        for j in range(2):
            ax = AX[i,j]
            ax.set_ylim([1700,0])
            ax.set_yticks([0,400,800,1200,1600]) 
            ax.set_yticklabels([0,400,800,1200,1600],fontdict = p.fd['ticks'])
            ax.set_xticks([-60,-50,-40,-30,-20])
            ax.set_xticklabels(['60°S','50°S','40°S','30°S','20°S'],
                                fontdict = p.fd['ticks'])
            img = ax.contourf(dims[i][0],dims[i][1],vals[i][j],
                                cmap = cmaps[j],levels = lvlss[j],
                                extend = 'both')
            CS = ax.contour(dims[i][0],dims[i][1],dens[i],
                                colors = 'w',levels = dense,
                                linewidths =3) 
            ax.clabel(CS,inline=1, fontsize=20)
            ax.grid(alpha = 1)
            
            ax.set_title(['2004 POP','1985-2016 WAGHC'][i],# + ' ' + ['Salinity','Temperature'][j],
                         fontdict = p.fd['title'])
            if i == 0:
                cbar = fig.colorbar(img, ax = AX[:,j],fraction = 0.075,pad = 0.02,orientation = 'horizontal')
                cbar.set_label(['S [g/kg]','T [°C]'][j],fontdict = p.fd['header'])
                cbar.ax.tick_params(labelsize = p.fd['ticks']['fontsize'])
                for d in dims[i][1]:
                    ul = 600 if 'density>27' in mask_kw else 500
                    if d <1200 and d > ul:
                        ax.plot(plats,[d,d],'r',linewidth = 3)
                    elif d < 600 and d > 500 and complete:
                        ax.plot(plats,[d,d],'r--',linewidth = 3)

    for ax in AX.flatten():
        import string 
        ax.set_title(string.ascii_lowercase[AX.flatten().tolist().index(ax)],fontsize = p.fd['abc'], fontweight='bold',loc = 'left')
        ax.text(-40 if 'density>27' in mask_kw else -38 ,1080,'AAIW',fontsize = 20,fontweight = 'bold',color = 'w')
    return fig,AX
    

def MLD(exp,p = PlotPreset(),date = '08',**kwargs):
    '''
    Plots a comparison of Mixed Layer Depth from the 2004 POP simulation and
    the WAGHC climatology for a given month, as well as the annual cycle for chosen locations.

    ! files required for plotting are not contained in the zenodo upload due to size limitations !
    
    ## Parameters:
    - exp: Experiment object, containing information about the experiment (not used here)
    - p: PlotPreset object, containing plot presets
    - date: str, month to plot (from '01' to '12')

    ## Returns:
    - fig: matplotlib Figure object
    - AX: array of matplotlib Axes objects
    '''

    # setup figure
    fig = plt.figure(figsize = (20,8),constrained_layout = True)
    gs = GridSpec(1,3,width_ratios = [1,1,.7],figure=fig)
    AX = []
    AX.append(fig.add_subplot(gs[:,0],projection = ccrs.SouthPolarStereo()))
    AX.append(fig.add_subplot(gs[:,1],projection = ccrs.SouthPolarStereo()))
    AX.append(fig.add_subplot(gs[:,2]))
    levels = np.arange(50,800,50)
    norm = Normalize(0,800)

    Date = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'][int(date)-1] + ' '

    # load data and plot maps
        # POP
    mld = xr.open_dataset(main_folder+'HMXL_2004%.2d_rm.nc'%int(date))['HMXL_ll']    
    AX[0].contourf(mld.XU,mld.YU,mld,cmap = cmo.cm.deep,levels =levels,norm = norm,
                    transform = ccrs.PlateCarree(),extend = 'max')
        # WAGHC
    woce = xr.open_dataset('/work/uo0780/u301534/Masterarbeit/WAGHC_ml_bd_UHAM-ICDC_v1_0_1.nc')['mixedlayerdepth_temperature'][int(date)-1]
    img = AX[1].contourf(woce.longitude,woce.latitude,woce.values,levels = levels,
                    cmap = cmo.cm.deep,norm = norm,transform = ccrs.PlateCarree(),extend = 'max')
    
    # plot boxes
    AX[0].set_title('%s POP'%Date,fontdict = p.fd['title'])
    AX[1].set_title('%s WAGHC'%Date,fontdict = p.fd['title'])
    cbar = fig.colorbar(img,ax = np.array(AX[:2]).ravel().tolist(),orientation = 'horizontal',pad = 0.1,fraction = 0.075)
    cbar.set_label('Mixed Layer Depth [m]',fontdict = p.fd['ticks'])
    cbar.ax.tick_params(labelsize = p.fd['ticks']['fontsize'])

    # polish maps, see subroutine in _plotter_helper.py
    polish(AX[1],polar = True)
    polish(AX[0],polar = True)

    # get 3 points and plot the mld over the year
    point1 = [-80,-55]
    point2 = [-10,-40]
    point3 = [130,-45]
    points = [point1,point2,point3]
    pcolors = ['red','orange','magenta']

    # load annual cycle data
        # POP
    from datetime import datetime,timedelta
    def add_time_dim(xda):
        xda = xda.expand_dims(time = [datetime.now()])
        return xda
    time = [datetime(2004,1,1) + timedelta(days = i) for i in range(366)]
    month_time = [t for t in time if t.day == 15]
    mlds = xr.open_mfdataset(sorted(glob(main_folder+'HMXL_2004*_rm.nc')),preprocess=add_time_dim)['HMXL_ll']
        # WAGHC
    woces = xr.open_dataset(main_folder+'WAGHC_ml_bd_UHAM-ICDC_v1_0_1.nc')['mixedlayerdepth_temperature']
    
    # plot annual cycle and markers on maps
    labels,objs = [],[]
    handler_map = {}
    for i,point in enumerate(points):
        label = '%d'%(abs(point[0])) + ('°E' if point[0] > 0 else '°W' )+ ', ' + '%d'%(abs(point[1])) + '°S'
        lon,lat = find_coord(woces,keys = ['longitude','latitude',''],lon = point[0],lat = point[1])[:2]
        print('lon,lat',lon,lat)
        AX[0].scatter(point[0],point[1],transform = ccrs.PlateCarree(),color = pcolors[i],s = 150,marker = 'd',edgecolor = 'k')
        AX[1].scatter(point[0],point[1],transform = ccrs.PlateCarree(),color = pcolors[i],s = 150,marker = 'd',edgecolor = 'k')
        AX[2].plot(month_time,mlds[:,lat,lon].values,color = pcolors[i],lw = 3)
        AX[2].plot(month_time,woces[:,lat,lon].values,color = pcolors[i],ls = '--',lw = 3)
        labels.append(label)
        globals()[string.ascii_lowercase[i]] = object()
        objs.append(globals()[string.ascii_lowercase[i]])
        handler_map[globals()[string.ascii_lowercase[i]]] = AnyObjectHandler(color = [pcolors[i],pcolors[i]],linestyle=['-','--'])
    
    AX[2].legend(objs,labels,handler_map = handler_map,fontsize = 17,loc = 'lower left')
    AX[2].set_title('Annual Cycle',fontdict = p.fd['title'])
    AX[2].set_ylabel('Depth [m]',fontdict = p.fd['ticks'])
    AX[2].set_ylim(800,0)
    AX[2].set_yticks([0,200,400,600,800])
    AX[2].grid(alpha = 1)
    AX[2].set_yticklabels([0,200,400,600,800],fontdict = p.fd['ticks'])
    AX[2].set_xticks(month_time)
    AX[2].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],fontdict = p.fd['ticks'])

    for ax in AX:
        ax.set_title(string.ascii_lowercase[AX.index(ax)],fontsize = p.fd['abc'], fontweight='bold',loc = 'left')

    return fig,AX

def velocities(exp,p = PlotPreset(),bs = 10,**kwargs):
    '''
    Method to plot horizontal velocities at a given depth for different experiments.
    Also plots histograms of the u, v and w velocity components.

    ## Parameters:
    - exp: list of Experiment objects or single Experiment object
    - p: PlotPreset object, containing plot presets
    - bs: int, resolution factor for velocity fields

    ## Returns:
    - fig: matplotlib Figure object
    - AX: array of matplotlib Axes objects
    '''

    #  check input, set up figure
    exp = check_exp(exp)
    fig = plt.figure(figsize=(20,10))
    gs = GridSpec(2,len(exp),figure=fig,height_ratios= [1,.5])
    lax,rax = [],[] # lax = map axes, rax = histogram axes
    for i in range(len(exp)):
            lax.append(fig.add_subplot(gs[0,i],projection = ccrs.SouthPolarStereo()))   
            rax.append(fig.add_subplot(gs[1,i]))  

    # plot velocities
    depth = 570
    objs = []
    handler_map = {}
    for i,e in enumerate(exp):
            # for each experiment, load velocity files
            u,v,w,g = load_velocity(e.velfiles,depth = depth,frac=int(bs*2))
            lax[i].set_title(e.name,fontdict = p.fd['title'])
            # load velocity files and plot absolute horizontal velocities
            kelevels = np.linspace(0,.1*1e4,21)
            norm = mcolors.Normalize(vmin=0, vmax=.1*1e4)
            img = lax[i].contourf(u['XU'],u['YU'],.5*(u**2+v**2)*1e4,
                                    cmap = e.cmap,norm = norm,levels = kelevels,extend = 'max',
                                    transform = ccrs.PlateCarree())
        
            plot_box([-60,20,-40,-30],
                 lax[i],color = 'xkcd:forest green',linewidth = 3,transform = ccrs.PlateCarree())

            polish(lax[i],fronts = False)
            fac = 1.5
            ubins = np.linspace(-10*fac,10*fac,100)
            vbins = np.linspace(-10*fac,10*fac,100)
            wbins = np.linspace(-2e-3*fac,2e-3*fac,100)
            rax[0].hist(u.values.flatten()*100,weights =g.values.flatten(), density=True,
                    bins = ubins,color = e.color,histtype = 'step',lw =2)
            rax[1].hist(v.values.flatten()*100,weights = g.values.flatten(),density=True,
                    bins = vbins,color = e.color,histtype = 'step',lw = 2)
            rax[2].hist(w.values.flatten()*100,weights = g.values.flatten(),density=True,
                    bins = wbins,color = e.color,histtype = 'step',lw = 2)
            
            g.values[np.isnan(u)] = 0
            u.values[np.isnan(u)] = 0
            v.values[np.isnan(v)] = 0
            w.values[np.isnan(w)] = 0
            lax[i].set_title(r'%.1f'%np.average((u.values**2+v.values**2)*.5*1e4,weights = g.values),
                                fontdict = p.fd['header'],loc = 'right')

            globals()[string.ascii_lowercase[i]] = object()
            objs.append(globals()[string.ascii_lowercase[i]])
            handler_map[globals()[string.ascii_lowercase[i]]] = SingleObjectHandler(color = e.color,linestyle='-')
        

    # figure settings
    clb = fig.colorbar(img, ax=lax,extend = 'max',fraction = 0.046,
                    pad = 0.03,orientation = 'vertical')
    
    clb.ax.set_yticks(np.arange(0,.1*1e4,.02*1e4))
    clb.ax.tick_params(labelsize=p.fd['ticks']['fontsize'])
    clb.set_label('Kinetic Energy [cm²/s²]',fontdict = p.fd['ticks'])

    rax[0].set_xlim(-10*fac,10*fac)
    rax[1].set_xlim(-10*fac,10*fac)
    rax[2].set_xlim(-.002*fac,.002*fac)
    rax[2].xaxis.get_offset_text().set_size(p.fd['ticks']['fontsize'])
    rax[0].set_title('u [cm/s]',fontdict = p.fd['ticks'])
    rax[1].set_title('v [cm/s]',fontdict = p.fd['ticks'])
    rax[2].set_title('w [cm/s]',fontdict = p.fd['ticks'])

    rax[1].legend(objs,[e.name for e in exp],handler_map = handler_map,
            fontsize = p.fd['legend'],ncol = 3, bbox_to_anchor=(0.5, -0.15),loc = 'upper center')   # move below
    for i,ax in enumerate(lax+rax):
            ax.set_title(string.ascii_lowercase[i],fontsize = p.fd['abc'], fontweight='bold',loc = 'left')    
    
    for ax in rax:
            ax.tick_params(labelsize=p.fd['ticks']['fontsize'])
            ax.grid(alpha = .8)

    return fig,[lax,rax]


def origin_distr(exp,p = PlotPreset(),
           mask_kw = 'density>27',**kwargs):
    '''
    Method to plot the origin of particles in a given experiment. This plot consists of:
    - histograms showing the distribution of particle origin in longitude, latitude,depth 
    and season of subduction

    ## Parameters:
    - exp: list of Experiment objects or single Experiment object
    - p: PlotPreset object, containing plot presets
    - mask_kw: str, keyword to define the mask for particles to be considered

    ## Returns:
    - fig: matplotlib Figure object
    - rax: array of matplotlib Axes objects
    '''

    # check input
    exp = check_exp(exp)

    # create figure structure
    fig = plt.figure(figsize=(20,8),constrained_layout = True)
    gs = GridSpec(2,3,figure=fig)
    rax = [] 
    rax.append(fig.add_subplot(gs[0,:]))
    rax.append(fig.add_subplot(gs[1,0]))
    rax.append(fig.add_subplot(gs[1,1]))
    rax.append(fig.add_subplot(gs[1,2]))
    
    # plot mapped distribution of particle origin
    objs, handler_map = [],{}
    for i,e in enumerate(exp):

        globals()[string.ascii_lowercase[i]] = object()
        objs.append(globals()[string.ascii_lowercase[i]])
        handler_map[globals()[string.ascii_lowercase[i]]] = SingleObjectHandler(color = e.color,linestyle='-')

        # load data
        df = pd.read_csv(e.pfile)
        sdf = pd.read_csv(e.sfile)  
        Mask = (df['transit_time']>0) & get_mask(sdf,mask_kw)
        weights = sdf['Volume'][Mask]/np.average(sdf['Volume'][Mask])

        hist_kw = {'weights': weights,
                   'color': e.color,
                   'histtype': 'step',
                   'lw': 2,
                   }
        f_kw = hist_kw.copy()
        f_kw['histtype'] = 'stepfilled'
        f_kw['alpha'] = 0.2
        lons = df['lon'][Mask]
        lons[lons>180] = lons[lons>180]-360
        rax[0].hist(lons,bins = np.arange(-180,180+5,5),
                        density = True,**hist_kw)
        
        rax[1].hist(df['lat'][Mask],bins = np.arange(-80,0+2,2),
                            density = True,
                        orientation = 'horizontal',**hist_kw)
        if not i:
            rax[0].hist(lons,bins = np.arange(-180,180+5,5),density = True,
                        **f_kw)
            rax[1].hist(df['lat'][Mask],bins = np.arange(-80,0+2,2),density = True,
                            orientation = 'horizontal',**f_kw)
            rax[3].hist(365-df['transit_time'][Mask]%365,
                bins = np.arange(0,366,15),
                density=True,**f_kw)  
        rax[2].hist(df['depth'][Mask],bins = np.arange(0,800,25),density = True,
                            orientation = 'horizontal',**hist_kw)
        if not i:
            rax[2].hist(df['depth'][Mask],bins = np.arange(0,800,25),density = True,
                                orientation = 'horizontal',**f_kw)
      
        rax[3].hist(365-df['transit_time'][Mask]%365,
            bins = np.arange(0,366,15),
            density=True,**hist_kw)        
    
    for ax in rax:
        ax.grid(alpha = 0.5)

    # latitude
    for ax in [rax[1]]:
        ax.set_ylim(-70,-20)
        ax.set_yticks(np.arange(-60,0,20))
        ax.set_yticklabels([])
        ax.tick_params(axis='x', which='both', labelsize=p.fd['ticks']['fontsize'])
    rax[1].set_yticklabels(['60°S','40°S','20°S'],fontdict = p.fd['ticks'])
    # longitude
    for ax in [rax[0]]:    
        ax.set_xlim(-180,180)
        ax.set_xticks(np.arange(-120,121,60 if polar else 120))
        ax.set_xticklabels([])
        ax.tick_params(axis='y', which='both', labelsize=p.fd['ticks']['fontsize'])
    rax[0].set_xticklabels(['120°W','60°W','0','60°E','120°E'] if polar else ['120°W','0','120°E'],
                                   fontdict = p.fd['ticks'])
    # depth
    for ax in [rax[2]]:
        ax.set_ylim(850,0)
        ax.set_yticks(np.arange(0,800,200))
        ax.set_yticklabels([])   
        ax.tick_params(axis='x', which='both', labelsize=p.fd['ticks']['fontsize'])
    rax[2].set_yticklabels(['0m','200m','400m','600m'],fontdict = p.fd['ticks'])

    rax[0].legend(objs,[e.name for e in exp],handler_map = handler_map,
            fontsize = p.fd['legend'],loc = 'upper right') 
    rax[0].set_title('Longitude',fontdict = p.fd['title'])
    rax[1].set_title('Latitude',fontdict = p.fd['title'])
    rax[2].set_title('Depth',fontdict = p.fd['title'])
    rax[3].set_title('Season of Subduction',fontdict = p.fd['title'])
    
    for ax in [rax[3]]:
        ax.tick_params(axis='y', which='both', labelsize=p.fd['ticks']['fontsize'])
        ax.set_xlim(-1,365)
        ax.set_xticks([0,31,59,90,120,151,181,212,243,273,304,334])
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'],fontdict = p.fd['ticks'])  

    for i,ax in enumerate(rax):            
        ax.set_title(string.ascii_lowercase[i], fontsize = p.fd['abc'], fontweight='bold',loc = 'left')

    return fig,rax

def origin_plain(exp,p = PlotPreset(),bs = 10,
           boxes = 'DAIP_new',
           mask_kw = 'density>27',**kwargs):
    '''
    Method to plot the origin of particles in a given experiment. This plot consists of:
    - a map showing the horizontal distribution of particle origin

    ## Parameters:
    - exp: list of Experiment objects or single Experiment object
    - p: PlotPreset object, containing plot presets
    - bs: int, bin size for histograms in degrees (horizontally) or x10 meters (vertically)
    - boxes: str, version of boxes to use for plotting
    - mask_kw: str, keyword to define the mask for particles to be considered
    
    ## Returns:
    - fig: matplotlib Figure object
    - lax: array of matplotlib Axes objects

    '''

    # check input
    exp = check_exp(exp)
    boxes = get_boxes(version=boxes)

    # create figure structure
    fig,lax = plt.subplots(1,3,figsize = (20,9),#constrained_layout=True,
                        sharex=True,sharey=True,
                        subplot_kw={'projection':ccrs.SouthPolarStereo()},
                        )

    # plot mapped distribution of particle origin
    objs, handler_map = [],{}
    Masks = []

    for i,e in enumerate(exp):
        globals()[string.ascii_lowercase[i]] = object()
        objs.append(globals()[string.ascii_lowercase[i]])
        handler_map[globals()[string.ascii_lowercase[i]]] = SingleObjectHandler(color = e.color,linestyle='-')

        # load data
        df = pd.read_csv(e.pfile)
        sdf = pd.read_csv(e.sfile)  
        Mask = (df['transit_time']>0) & get_mask(sdf,mask_kw)
        weights = sdf['Volume'][Mask]/np.average(sdf['Volume'][Mask])

        Masks.append([sdf['Volume'][Mask & boxmask(df,boxes.loc[j])].sum()/sdf['Volume'][Mask].sum()*100 for j in range(len(boxes))])
        # plot horizontal distribution of particle origin
        img = plot_ventilations(df,lax[i],Mask,weights,bin_size=bs,
                                cmap= e.cmap)
        
        lax[i].set_title(e.name,fontdict = p.fd['title']) 

    clb = fig.colorbar(img, ax=lax,extend = 'max',fraction = 0.015,
                                pad = 0.04,
                                orientation = 'vertical')
        
    clb.ax.tick_params(labelsize=p.fd['ticks']['fontsize'])
    clb.ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    clb.ax.yaxis.get_offset_text().set_size(p.fd['ticks']['fontsize'])
    
    # adapt plot layout    
    for i,ax in enumerate(lax):
        for (box,text,mask) in zip(boxes['box'],boxes['sname'],Masks[i]):
            text += '\n%.0f%%'%(mask)
            plot_box(box,ax,text,transform = ccrs.PlateCarree(),color = 'k',linewidth=3,alpha = .8)
        gl = polish(ax)
        ax.set_title(string.ascii_lowercase[i],fontsize = p.fd['abc'], fontweight='bold',loc = 'left')
    
    return fig,lax


def age_map(exp,p = PlotPreset(),bs = 10,mask_kw = 'density>27',std = False,boxes = 'DAIP_new',**kwargs):
    '''
    Method to map the mean transit time (and standard deviation) of particles in a given experiment.

    ## Parameters:
    - exp: list of Experiment objects or single Experiment object
    - polar: bool, if True, the plot will be in polar projection
    - p: PlotPreset object, containing plot presets
    - bs: int, bin size for histograms in degrees
    - mask_kw: str, keyword to define the mask for particles to be considered
    - std: bool, if True, also plots the standard deviation of particle ages
    - boxes: str, version of boxes to use for plotting

    ## Returns:
    - fig: matplotlib Figure object
    - AX: array of matplotlib Axes objects
    '''

    # set up figure
    exp = check_exp(exp)
    exp_grid,lons,lats = get_age_grid(exp,bs = bs,mask_kw = mask_kw)
    I = 2 if std else 1
    boxes = get_boxes(version=boxes)

    fig,AX = plt.subplots(I,3,figsize = (20,8),
                        sharex=True,sharey=True,constrained_layout = True,
                        subplot_kw={'projection':ccrs.SouthPolarStereo()},
                        squeeze=False)

    for i in range(len(exp_grid)):
        ax = AX[:,i] 
        meangrid,stdgrid,_ = exp_grid[i]
        name = exp[i].name

        # plot mean age per bin
        ax[0].set_title(name,fontdict = p.fd['title'])
        norm2 = plt.Normalize(vmin = 10,vmax = 50)
        cmap2 = plt.cm.get_cmap('jet',8)
        cmap2.set_over('xkcd:dried blood')
        ax[0].pcolormesh(lons,lats,meangrid.T,norm = norm2,cmap = cmap2,transform = ccrs.PlateCarree())
   
        # plot std per bin  
        if std:
            norm3 = plt.Normalize(vmin = 10,vmax = 30)
            cmap3 = plt.cm.get_cmap('cool',8)
            cmap3.set_over('xkcd:neon pink')
            cmap3.set_under('xkcd:bright cyan')
            ax[1].pcolormesh(lons,lats,stdgrid.T,norm = norm3,cmap = cmap3,transform = ccrs.PlateCarree())
        
        # plot boxes
        for axx in ax:
            for box,text in zip(boxes['box'],boxes['sname']):
                plot_box(box,axx,text,transform = ccrs.PlateCarree(),color = 'xkcd:indigo',linewidth=3)

            gl = polish(axx)

        # colorbars
        if i == len(exp_grid)-1:
            clb = fig.colorbar(plt.cm.ScalarMappable(norm = norm2,cmap = cmap2),ax = ax[0],
                        label = 'Years',extend = 'max',fraction = 0.046, pad = 0.04)
            clb.ax.tick_params(labelsize=p.fd['ticks']['fontsize'])
            clb.set_label('Particle Age [Years]',fontdict = p.fd['ticks'])
            if std:
                clb = fig.colorbar(plt.cm.ScalarMappable(norm = norm3,cmap = cmap3),ax = ax[1],
                            label = 'Years',extend = 'both',fraction = 0.046,pad = 0.04)
                clb.ax.tick_params(labelsize=p.fd['ticks']['fontsize'])
                clb.set_label('Standard Deviation [Years]',fontdict = p.fd['ticks'])

    for i,ax in enumerate(AX.flatten()):
        ax.set_title(string.ascii_lowercase[i],fontsize = p.fd['abc'], fontweight='bold',loc = 'left')
    
    return fig,AX

def transit_times(exp,dt = 1, mask_kw = 'density>27',
                  freefit = True,p = PlotPreset(),
                  boxes = False,pc = False,**kwargs):
    '''
    Method to plot histograms of particle transit times for different experiments,
    optionally separated by source region boxes.

    ## Parameters:
    - exp: list of Experiment objects or single Experiment object
    - dt: float, bin width for histograms in years
    - mask_kw: str, keyword to define the mask for particles to be considered
    - freefit: bool, if True, allows the Gaussian fit to adjust all parameters freely
    - p: PlotPreset object, containing plot presets
    - boxes: str or bool, version of boxes to use for plotting, if False no boxes are used
    - pc: bool, if True, shows the percentage of particles from each box
    
    ## Returns:
    - fig: matplotlib Figure object
    - AX: array of matplotlib Axes objects
    '''

    exp = check_exp(exp)    
    # case if boxes are used, multiple subplots
    if boxes: 
        boxes = get_boxes(version = boxes)
        I = len(boxes)
    fig,AX = plt.subplots(I,len(exp),figsize = (20,5*I+3),
                          sharex = True,sharey = 'row', constrained_layout = True,
                          squeeze=False)

    # collect and plot data
    for i,e in enumerate(exp):
        df = pd.read_csv(e.pfile)
        sdf = pd.read_csv(e.sfile)  
        MASK = (df['transit_time']>0) & get_mask(sdf,mask_kw)

        for j in range(I):
            ax = AX[j,i]
            if type(boxes) == pd.DataFrame:
                Mask= MASK & boxmask(df,boxes.loc[j])
                if i == 0:
                    ax.set_ylabel(boxes['name'][j],fontdict = p.fd['header'])
                if pc:
                    ax.set_title('%.1f%%'%(sdf['Volume'][Mask].sum()/sdf['Volume'][MASK].sum()*100),fontdict = p.fd['header'],loc = 'right')
            else: Mask = MASK

            weights = sdf['Volume'][Mask]/np.average(sdf['Volume'][Mask])

            hist = np.histogram(df['transit_time'][Mask]/365-sdf['start_time'][Mask]/365,
                                bins = np.arange(0,120+dt,dt),weights = weights,density=False)
            h = hist[0] /np.sum(MASK)/dt
            bars = ax.bar(hist[1][:-1],h,width = dt,alpha = 0.5,color = e.color,align = 'edge',edgecolor='k',lw=1)
        
            lines,label = gaussian_fit(hist[1]+dt/2,h,ax,
                mass = np.sum(hist[0]),dt = dt,vals = np.average(hist[1][:-1]+dt/2,weights=hist[0]),
                color = 'k',lw = 3, free = freefit,eps = True,dg=True)
            handles = [bars, lines]
            labels = [r'$\varnothing$ = %.1f y'%np.average(df['transit_time'][Mask]/365-sdf['start_time'][Mask]/365,weights=weights), label]

            ax.legend(handles, labels,fontsize = p.fd['legend']+3) 
            if j ==0: ax.set_title(e.name,fontdict = p.fd['title'])

    for ax in AX[:,0]:
        ax.tick_params(axis='y', which='both', labelsize=p.fd['ticks']['fontsize'])

    for ax in AX[-1,:]:
        ax.set_xlim(-1,120)
        ax.set_xticks(np.arange(0,121,10))
        xtl = [str(i) if i%20 == 0 else '' for i in np.arange(0,111,10)] + ['']
        ax.set_xticklabels(xtl,fontdict = p.fd['ticks'])
        ax.set_xlabel(r'$\tau$ [years]',fontdict = p.fd['ticks'])

    for i,ax in enumerate(AX.flatten()):
        ax.grid(alpha = 0.5)
        ax.set_title(string.ascii_lowercase[i],fontsize = p.fd['abc'], fontweight='bold',loc = 'left',pad=-10)
    
    return fig,AX

def pathways(exp,mask_kw = None,p = PlotPreset(),boxes = False,quick = False,diff = True,**kwargs):
    '''
    Method to plot the particle pathways for different experiments.
    Optionally, the difference between pathways of different experiments can be shown.
    Another option is to show the pathways for different source region boxes.

    ## Parameters:
    - exp: list of Experiment objects or single Experiment object
    - mask_kw: str, keyword to define the mask for particles to be considered
    - p: PlotPreset object, containing plot presets
    - boxes: str or bool, version of boxes to use for plotting, if False no boxes are used
    - quick: int or bool, if int, only uses a subset of particles for plotting
    - diff: bool, if True, plots the difference between pathways of different experiments

    ## Returns:
    - fig: matplotlib Figure object
    - AX: array of matplotlib Axes objects

    '''
    
    exp = check_exp(exp)
    I = 2 if diff else 1
    # case if boxes are used, multiple subplots
    if boxes:  
        diff = False
        print('Difference is not possible with boxes')
        boxes = get_boxes(version = boxes)
        I = len(boxes)
    fig,AX = plt.subplots(I,len(exp),figsize = (7*len(exp),6*I+2),
                          constrained_layout = True,
                          subplot_kw={'projection':ccrs.SouthPolarStereo() if polar else ccrs.PlateCarree()},
                          squeeze=False)
    
    # set mapping parameters
    nlvl = 7
    path_kw = {'levels':np.logspace(-3.5,-0.5,nlvl),
               'norm': mcolors.LogNorm(vmin=1e-4, vmax=10**(-.5)),
               'cmap': plt.cm.get_cmap(cmc.navia_r,nlvl+2),
               'extend':'both',
               'transform':ccrs.PlateCarree()}

    Paths = []
    for i,e in enumerate(exp):
        df = pd.read_csv(e.pfile)
        sdf = pd.read_csv(e.sfile)
        zf = zarr.open(e.zfile,mode = 'r')  
        MASK = (df['transit_time']>0) & get_mask(sdf,mask_kw)
        if quick: 
            MASK[quick:] = False 
            tvol = sdf['Volume'][get_mask(sdf,mask_kw)][:quick].sum()
        else: 
            tvol = sdf['Volume'][get_mask(sdf,mask_kw)].sum()
 
        for j in range(I):
            ax = AX[j,i]    
            if type(boxes) == pd.DataFrame:
                print(boxes.loc[j]) 
                Mask= MASK & boxmask(df,boxes.loc[j])
                if i == 0:
                    ax.text(
                        -0.1, 0.5,             
                        boxes['name'][j],
                        va='center', ha='right',
                        transform=ax.transAxes,  
                        fontdict = p.fd['header'],
                        rotation=90             
                    )
                plot_box(boxes['box'].loc[j],ax,color = 'k',linewidth = 3,transform = ccrs.PlateCarree())

            else: 
                Mask = MASK
                if j ==1:
                    continue

            weights = sdf['Volume'][Mask]/np.average(sdf['Volume'][Mask])
            path = np.average(zf[list(Mask)],weights = weights,axis = 0).T*sdf['Volume'][Mask].sum()/tvol
            path = np.concatenate([path,path],axis = 1)[:,:181]
            Paths.append(path)  
            path[path==0] = np.nan
            img = ax.contourf(np.arange(0,362,2)+1,np.arange(-80,-2,2)+1,path,**path_kw)
            path[np.isnan(path)] = 0
            polish(ax)

            if j == 0: ax.set_title(e.name,fontdict = p.fd['title'])
    
            if i == len(exp)-1:
                cbar = fig.colorbar(img, ax=ax,fraction = .046, pad = .01)
                cbar.set_ticks([1e-3,1e-2,1e-1])
                cbar.set_label('rel. particle pathways',fontdict = p.fd['ticks'])
                cbar.set_ticklabels([r'10$^{-3}$',r'$10^{-2}$',r'$10^{-1}$'],fontsize = p.fd['ticks']['fontsize'])

    if diff:
        diff_kw = {'levels':np.concatenate([-np.logspace(-3.5,-.5,nlvl)[::-1],[0],np.logspace(-3.5,-.5,nlvl)]),
                   'norm': mcolors.SymLogNorm(linthresh=5e-3, linscale=1,vmin=-1, vmax=1),
                   'cmap': cmc.vik_r,
                   'extend':'both',
                   'transform':ccrs.PlateCarree()}

        diffs = [Paths[0]-Paths[1],Paths[0]-Paths[2],Paths[1]-Paths[2]]
        titles = ['%s - %s'%(exp[0].name,exp[1].name),
                    '%s - %s'%(exp[0].name,exp[2].name),
                    '%s - %s'%(exp[1].name,exp[2].name)]
        for i in range(len(exp)):
            ax = AX[1,i]
            diff = diffs[i]
            diff[(diff<1e-5) & (diff>-1e-5)] = np.nan
            img = ax.contourf(np.arange(0,362,2)+1,np.arange(-80,-2,2)+1,diffs[i],**diff_kw)
            polish(ax)
            if i == len(exp)-1:
                cbar = fig.colorbar(img, ax=ax,fraction = .046, pad = .01)
                cbar.set_ticks([-1e-1,-1e-2,-1e-3,0,1e-3,1e-2,1e-1])
                cbar.set_label('rel. particle pathways',fontdict = p.fd['ticks'])
                cbar.set_ticklabels([r'$-10^{-1}$',r'$-10^{-2}$',r'$-10^{-3}$',0,r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'],
                                    fontsize = p.fd['ticks']['fontsize'])
            ax.set_title(titles[i],fontdict = p.fd['title'])

    for i,ax in enumerate(AX.flatten()):
        ax.set_title(string.ascii_lowercase[i],fontsize = p.fd['abc'], fontweight='bold',loc = 'left',pad=-10)

    else:   return fig,AX


def properties(exp,p=PlotPreset(),mask_kw = 'density>27',
               boxes = 'DAIP_new',**kwargs):
    '''
    Method to plot the temperature-salinity diagram and density histogram
    of particles in different experiments.
    
    ## Parameters:
    - exp: list of Experiment objects or single Experiment object
    - p: PlotPreset object, containing plot presets
    - mask_kw: str, keyword to define the mask for particles to be considered
    - boxes: str or bool, version of boxes to use for plotting, if False no boxes are used
    
    ## Returns:
    - fig: matplotlib Figure object
    - AX: array of matplotlib Axes objects
    '''

    exp = check_exp(exp)
    grid_temp = np.arange(0,21,.1)
    grid_sal = np.arange(33.8,36.2,.01)
    grid_temp,grid_sal = np.meshgrid(grid_temp,grid_sal)
    sigma_theta = gsw.sigma0(grid_sal,grid_temp)

    if boxes:
        boxes = get_boxes(version = boxes)
        bcs = ['tab:blue','tab:orange','tab:green','tab:red','tab:pink']
    fig,AX = plt.subplots(2,len(exp),figsize = (7*len(exp),5*I+2),
                          sharex = 'row',sharey = 'row', constrained_layout = True,
                          squeeze=False)

    for i,e in enumerate(exp):
        # ---------- T-S diagram ------------
        ax = AX[0,i]
        df = pd.read_csv(e.pfile)
        sdf = pd.read_csv(e.sfile)  
        MASK = (df['transit_time']>0) & get_mask(sdf,mask_kw)

        if type(boxes) == pd.DataFrame:
            for j in range(len(boxes)):
                Mask = MASK & boxmask(df,boxes.loc[j]) 
                ax.plot(df['salt'][Mask],df['temp'][Mask],'.',markersize = .4,alpha =.8,color = bcs[j])
                if i==0: ax.scatter([-10],[-10],label = boxes['sname'].loc[j],color = bcs[j],s =50)
        else:
            ax.plot(df['salt'][MASK],df['temp'][MASK],'.',markersize = .4,alpha =.7,color = e.color)
        ax.set_xlim(33.8,36.2)
        ax.set_ylim(0,21) 
        CS = ax.contour(grid_sal,grid_temp,sigma_theta,colors='black',alpha=0.5,levels = [23,24,25,26,28,29])
        CSS = ax.contour(grid_sal,grid_temp,sigma_theta,colors='red',alpha=1,levels = [27,27.43],linewidths = 2.5)
        ax.clabel(CS, inline=2, fontsize=p.fd['legend'])
        ax.clabel(CSS, inline=2, fontsize=p.fd['legend'])

        if i == 1: ax.set_xlabel('Salinity [g\kg]',fontdict = p.fd['ticks'])

        ax.set_title(e.name,fontdict = p.fd['title'])
        if i ==0:
            ax.set_ylabel('Temperature [°C]',fontdict = p.fd['ticks'])
            ax.set_yticks(np.arange(0,24,5))
            ax.set_yticklabels(np.arange(0,24,5),fontdict = p.fd['ticks'])
            ax.legend(loc = 'lower right',fontsize = p.fd['legend'])

        ax.set_xticks(np.arange(34,36.1,.5))
        ax.set_xticklabels(np.arange(34,36.1,.5),fontdict = p.fd['ticks'])
        
        # ---------- density histogram ------------
        ax = AX[1,i]
        weights = sdf['Volume'][MASK]/np.average(sdf['Volume'][MASK])
        diapycnity = (df['density'][MASK]-sdf['density'][MASK])*weights
        ax.hist(df['density'][MASK],bins = np.arange(25,28,.05),
                   label = r'$\Delta\sigma_{0}$'+ r': %.2f'%(-np.nanmean(diapycnity)),
                       weights = weights/np.sum(MASK),
                       histtype = 'step',color = 'gray' if type(boxes) == pd.DataFrame else e.color,
                       lw = 4,alpha = .5 if type(boxes) == pd.DataFrame else 1) 

        if type(boxes) == pd.DataFrame:
            for j in range(len(boxes)):
                Mask = MASK & boxmask(df,boxes.loc[j])
                weights = sdf['Volume'][Mask]/np.average(sdf['Volume'][Mask])
                diapycnity = (df['density'][Mask]-sdf['density'][Mask])

                ax.hist(df['density'][Mask],bins = np.arange(25,28,.05),
                   label = boxes['sname'][j]+ r': %.2f'%(-np.nanmean(diapycnity)),
                       weights =weights/np.sum(MASK),
                       histtype = 'step',color = bcs[j],lw = 2)    
        
        ax.axvline(27.43,color = 'r',linestyle = '--',lw = 2.5)
        ax.axvline(27,color = 'r',linestyle = '--',lw = 2.5)

        ax.legend(fontsize = p.fd['legend'],loc = 'upper left')
        ax.tick_params(axis='y', which='both', labelsize=p.fd['ticks']['fontsize'])     
        ax.grid(alpha = 0.3)
        if i ==1: ax.set_xlabel(r'$\sigma_{0}$ [kg/m$^3$]',fontdict = p.fd['ticks'])
        
        ax.set_xlim(25.3,27.7)
        ax.set_xticks(np.arange(25.5,28,.5))
        ax.set_xticklabels(np.arange(25.5,28,.5),fontdict = p.fd['ticks'])

    for i,ax in enumerate(AX.flatten()):
        ax.set_title(string.ascii_lowercase[i],fontsize = p.fd['abc'], fontweight='bold',loc = 'left')
    
    return fig,AX


#######  ANALYSIS FUNCTION   #########
def get_share(exp,mask_kw = None):
    '''
    Method to calculate and print the share of subducted, not subducted,
    and trapped or left domain particles for different experiments.
    ## Parameters:
    - exp: list of Experiment objects or single Experiment object
    - mask_kw: str, keyword to define the mask for particles to be considered
    '''

    exp = check_exp(exp)
    for e in exp:
        sdf = pd.read_csv(e.sfile)
        Mask = get_mask(sdf,mask_kw)
        print('Total:', np.sum(Mask))
        df = pd.read_csv(e.pfile)
        submask = (df['transit_time']>0)
        nanmask = (df['stuck']==1)

        print('Subducted:', np.sum(submask & Mask),end = ' ')
        print('%.2f%%:'%(np.sum(sdf['Volume'][submask & Mask])/np.sum(sdf['Volume'][Mask])*100))

        print('Not subducted:', np.sum(~submask & ~nanmask & Mask),end = ' ')
        print('%.2f%%:'%(np.sum(sdf['Volume'][~submask & ~nanmask & Mask])/np.sum(sdf['Volume'][Mask])*100))

        print('trapped or left domain:', np.sum(~submask &nanmask & Mask),end = ' ')
        print('%.2f%%:'%(np.sum(sdf['Volume'][~submask &nanmask & Mask])/np.sum(sdf['Volume'][Mask])*100))
