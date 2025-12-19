########################################################################################################
# Author: Simon Schäfers
# Contact: sschaefers@geomar.de
# Date: 2025-12-08

########################################################################################################

# This script contains functions to produce metadata files for particle advection experiments. For 
# subduction, the functions calculate the transit time until subduction, the location of subduction, and
# the properties at subduction. It further contains a function to identify removed particles.
# Additionally, it includes a function to bin particle pathways into a 2D histogram/map.

import zarr
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob


def get_subduction(set):
    '''
    Function to calculate the transit time until subduction for each particle in the experiment. The 
    function iterates through all sorted zarr files, checking when each particle crosses the subduction
    threshold (ds.up == 1). The transit time is calculated based on the time step derived from the length
    of the zarr files. The results are saved in a CSV file.
    
    ## Parameters:
    - set: An experiment setting object containing attributes like save_folder, name, and YEARS_PER_ITERATION.
    The set object is created using the `_settings.py` script. 
    '''

    # Get sorted files
    files = sorted(glob(set.save_folder+set.name+'_*_sorted.zarr'),key = lambda x: int(x.split('_')[-2]))
    
    # Check file order
    year = int(files[0].split('_')[-2])
    assert year == 0, 'Files do not start at year 0'
    for file in files[1:]:
        assert int(year) + int(set.YEARS_PER_ITERATION) == int(file.split('_')[-2]), 'Files do not follow the correct order'
        year = int(file.split('_')[-2])
    print('Files are in correct order up to year',year)

    # open first file to get time step, check for `up` variable and determine time step
    zfile = zarr.open(files[0],mode='r')
    print(zfile.up.shape)
    dt = set.YEARS_PER_ITERATION*365//(zfile.up.shape[1]-1)
    print('Time step is',dt,'days')

    # Initialize subduction array and iterate through files
    i = 0
    sub = np.zeros((zfile.up.shape[0]),dtype = int)
    for file in tqdm(files):
        ds = zarr.open(file,mode='r')
        for j in range(ds.up.shape[1]):
            sub_mask = (sub == 0)
            cross_mask = (ds.up[:,j]==1)
            sub[sub_mask&cross_mask] = i*dt
            i+=1  
        # correct for extra increment at the end of each file
        i-=1

    # Save subduction array
    df = pd.DataFrame({'transit_time':sub})
    df.to_csv(set.main_folder + 'metadata/' + set.name+'_subduction.csv',index = False)


def get_subduction_vals(set,loc = True, props = False):
    '''
    Function to determine the geographical location (longitude, latitude, depth) of subduction for each particle
    and/or the properties (temperature, salinity, density) at the time of subduction. The function reads the
    transit times from a previously generated CSV file and iterates through all sorted zarr files to extract
    the required data at the time of subduction. The results are saved in a CSV file.
    location and properties can be toggled using the `loc` and `props` parameters. In matters of computational
    efficiency, it can be sensible to only enable one of the two options at a time.
    
    ## Parameters:
    - set: An experiment setting object containing attributes like save_folder, name, and YEARS_PER_ITERATION.
    The set object is created using the `_settings.py` script.
    - loc (bool): If True, the geographical location at subduction is calculated and saved.
    - props (bool): If True, the properties at subduction are calculated and saved.
    
    '''

    # check if subduction file exists, if so load it
    assert os.path.exists(set.main_folder + 'metadata/' + set.name+'_subduction.csv'),'Subduction file does not exist. Run get_subduction first.'
    df = pd.read_csv(set.main_folder + 'metadata/' + set.name+'_subduction.csv')
    
    # get zarr files and determine time step
    files = sorted(glob(set.save_folder+set.name+'_*_sorted.zarr'),key = lambda x: int(x.split('_')[-2]))
    zfile = zarr.open(files[0],mode='r')
    dt = set.YEARS_PER_ITERATION*365//(zfile.up.shape[1]-1)

    # get transit time iterate through files
    sub = df['transit_time']
    i = 0
    for file in tqdm(files):
        ds = zarr.open(file,mode='r')
        if i == 0:
            # initialize arrays
            if loc:
                lon,lat,depth = np.zeros((ds.z.shape[0])),np.zeros((ds.z.shape[0])),np.zeros((ds.z.shape[0]))
            if props:
                temp,salt = np.zeros((ds.z.shape[0])),np.zeros((ds.z.shape[0]))

        for j in range(ds.up.shape[1]):
            sub_mask = (sub//dt == i)

            if sum(sub_mask) >0 and i!=0:
                if loc:
                    lon[sub_mask] = ds.lon[:,j][sub_mask]
                    lat[sub_mask] = ds.lat[:,j][sub_mask]
                    depth[sub_mask] = ds.z[:,j][sub_mask]
                if props:
                    temp[sub_mask] = ds.TEMP[:,j][sub_mask]
                    salt[sub_mask] = ds.SALT[:,j][sub_mask]*1e3
            i+=1  
        # correct for extra increment at the end of each file
        i-=1

    # save to dataframe
    if loc:
        df['lon'],df['lat'],df['depth'] = lon,lat,depth
    if props:
        # calculate density using gsw
        import gsw
        density = gsw.sigma0(salt,temp)
        df['temp'],df['salt'],df['density'] = temp,salt,density
    df.to_csv(set.main_folder + 'metadata/' + set.name+'_subduction.csv',index = False)
 
 
def get_stuck(set):
    '''
    Function to identify particles that got stuck during advection (i.e., particles that were deleted).
    The function reads the last sorted zarr file to check for NaN values in the longitude array,
    indicating that the particle was removed. The results are saved in a CSV file.

    ## Parameters:
    - set: An experiment setting object containing attributes like save_folder and name.
    The set object is created using the `_settings.py` script.
    
    '''
    # check if subduction file exists, if so load it
    assert os.path.exists(set.main_folder + 'metadata/' + set.name+'_subduction.csv'),'Subduction file does not exist. Run get_subduction first.'
    df = pd.read_csv(set.main_folder + 'metadata/' + set.name+'_subduction.csv')
    
    # get zarr files and determine time step
    files = sorted(glob(set.save_folder+set.name+'_*_sorted.zarr'),key = lambda x: int(x.split('_')[-2]))
    print(set.save_folder,files)
    ds = zarr.open(files[-1],mode='r')

    # find out which particles were deleted during advection
    stuck_mask = np.isnan(ds.lon[:,-1])

    # save to dataframe
    df['stuck'] = stuck_mask.astype(int)
    df.to_csv(set.main_folder + 'metadata/' + set.name+'_subduction.csv',index=False)


def bin_pathways(set,bs =2):
    '''
    Function to bin particle pathways into a 2D histogram/map. The function reads the `*_complete.zarr` file
    containing the complete trajectories of particles and bins their longitude and latitude positions until 
    subduction into specified bins. The binned data is saved in a zarr file of shape 
    (num_particles, num_lon_bins, num_lat_bins).

    ## Parameters:
    - set: An experiment setting object containing attributes like save_folder and name.
    The set object is created using the `_settings.py` script.
    - bs (int): Bin size in degrees for both longitude and latitude. Default is 2 degrees.
    '''

    # load complete zarr file and subduction file
    z_path_complete = set.save_folder+set.name+ '_complete.zarr'
    pfile = set.main_folder + 'metadata/' + set.name+'_subduction.csv'
    assert os.path.exists(z_path_complete), 'Complete zarr file does not exist. Run `_zarr_routines.chunk` first.'
    ds = zarr.open(z_path_complete, mode = 'r')
    tt = pd.read_csv(pfile)['transit_time']

    # get zarr files and determine time step
    files = sorted(glob(set.save_folder+set.name+'_*_sorted.zarr'),key = lambda x: int(x.split('_')[-2]))
    zfile = zarr.open(files[0],mode='r')
    dt = set.YEARS_PER_ITERATION*365//(zfile.up.shape[1]-1)

    # define bins and initialize output zarr file 
    lon_bins = np.arange(0,360+bs,bs)
    lat_bins = np.arange(-80,0,bs)
    bin_file = set.main_folder + 'metadata/' + set.name + '_pathways.zarr'
    ds_bin = zarr.open(bin_file, mode = 'w',shape = (ds.z.shape[0],len(lon_bins)-1,len(lat_bins)-1),
                        chunks = (100,len(lon_bins)-1,len(lat_bins)-1),dtype = bool)

    # iterate through particles and bin pathways
    for i in tqdm(range(ds.z.shape[0])):
        # determine time until subduction
        if tt[i] > 0:
            hist,_,_= np.histogram2d(ds.lon[i,:int(tt[i]//dt)],ds.lat[i,:int(tt[i]//dt)],bins = [lon_bins,lat_bins])
        # if not subducted, take full trajectory
        else: 
            hist,_,_ = np.histogram2d(ds.lon[i,:],ds.lat[i,:],bins = [lon_bins,lat_bins])
        # save histogram as boolean array
        ds_bin[i] = hist.astype(bool)


def get_zonal_displacement(set):
    '''
    Function to calculate the zonal displacement of particles until subduction.
    The function reads the `*_complete.zarr` file containing the complete trajectories of particles
    and calculates the zonal displacement for each particle until subduction. The results are saved
    in the subduction CSV file.

    ## Parameters:
    - set: An experiment setting object containing attributes like save_folder and name.
    The set object is created using the `_settings.py` script.

    '''

    # load complete zarr file and subduction file
    z_path_complete = set.save_folder+set.name+ '_complete.zarr'
    pfile = set.main_folder + 'metadata/' + set.name+'_subduction.csv'
    assert os.path.exists(z_path_complete), 'Complete zarr file does not exist. Run `_zarr_routines.chunk` first.'    
    ds = zarr.open(z_path_complete, mode = 'r')
    df = pd.read_csv(pfile)['transit_time']
    tt = df['transit_time']

    # //hardcoded// set time step in days to 5
    dt = 5

    # collect displacements per particle
    displ=[]
    dlon = []
    for i in tqdm(range(ds.lon.shape[0])):
        
        # determine time step until subduction
        if tt[i] >0:
            # calculate zonal displacement as difference in longitude
            ud = np.diff(ds.lon[i,:int(tt[i]//dt)])
            
            # correct for dateline crossing
            ud[ud>180] = ud[ud>180]-360
            ud[ud<-180] = ud[ud<-180]+360

            # calculate total zonal displacement in meters, accounting for latitude dependence
            dlon.append(np.nansum(-ud))
            displ.append(np.nansum(-ud*np.cos(np.deg2rad(ds.lat[i,:int(tt[i]/dt)-1]))/(5*24*3600)*6371e3*2*np.pi/360,axis = 1))
        
        # if not subducted, take full trajectory
        else: 
            ud = np.diff(ds.lon[i,:])
            ud[ud>180] = ud[ud>180]-360
            ud[ud<-180] = ud[ud<-180]+360
            dlon.append(np.nansum(-ud))
            displ.append(np.nansum(-ud*np.cos(np.deg2rad(ds.lat[i,:-1]))/(5*24*3600)*6371e3*2*np.pi/360,axis = 1))
    
    df['dlon'] = dlon
    df['displacement'] = displ
    df.to_csv(pfile,index = False)