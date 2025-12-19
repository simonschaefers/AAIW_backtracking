########################################################################################################
# Author: Simon Schäfers
# Date: 2025-01-27

########################################################################################################

# This script contains routines to make the particle output (zarr files) usable for the analysis.

import os
import zarr
import numpy as np
import pandas as pd
from src.advection_routine._settings import ReadSettings
from glob import glob
from tqdm import tqdm

def time_lag_correction(set):
    '''
    The `time_lag_correction` function corrects the time lag of the first particle data based on the time 
    of release. This is necessary to allow a further advection of the particles based on the last entry of 
    the zarr file. The corrected zarr file replaces the original zarr file, the original zarr file is 
    renamed to an uncorrected zarr file.
    This method currently only works if particles are loaded from a csv file.
    This method is only necessary if particles are released at different time steps.
    
    ## Parameters:
    - set: ReadSettings, settings for the advection routine. Important attribute is `particle_file`,   
    `save_folder`, `name`, and `TS_SAVE`.
    '''

    # check input
    assert type(set) == ReadSettings, 'Input must be of type ReadSettings'

    # load start particle file and zarr file (first output)
    pfile = pd.read_csv(set.particle_file)
    zfile = zarr.open(set.save_folder + set.name + '_0.zarr',mode = 'r')
    
    # check if number of particles match
    assert len(pfile) == zfile['z'].shape[0], 'Number of particles in pfile and zfile do not match'

    # get time lag of each particle, group them to reduce computation time
    dtime = pd.to_datetime(pfile['time']).max() - pd.to_datetime(pfile['time'])
    dtime = (dtime.dt.days+int(set.TS_SAVE.days)-1)//int(set.TS_SAVE.days)
    same_dt = dtime.groupby(dtime)
    dtt = np.unique(dtime)

    # open new zarr file, correct time lag of each particle and save
    zfile_cor = zarr.open(set.save_folder + set.name + '_0_corrected.zarr',mode = 'w')
    for key in zfile.keys():
        if key in ['obs','trajectory']:
            zfile_cor[key] = zfile[key]
        else:
            zfile_cor[key] = zarr.array(np.zeros_like(zfile[key]))
            for i in range(len(dtt)):
                Is = same_dt.indices[dtt[i]]
                zfile_cor[key][Is] = np.roll(zfile[key][Is],dtt[i].astype(int),axis = 1)
        print('Corrected',key)

    # rename original zarr file to uncorrected zarr file
    file_orig = set.save_folder + set.name + '_0.zarr'
    file_cor = set.save_folder + set.name + '_0_corrected.zarr'
    file_uc = set.save_folder + set.name + '_0_uncorrected.zarr'
    os.system('mv %s %s'%(file_orig,file_uc))
    os.system('mv %s %s'%(file_cor,file_orig))


def sort_zarr(set,year):
    '''
    Method to sort the zarr file based on the trajectory of the particles. This is necessary to maintain
    consistent trajectories of the particles over the years. The sorted zarr file is saved as a new zarr.
    Utilizes the `get_traj` method to get the order of the trajectories.

    ## Parameters:
    - set: ReadSettings, settings for the advection routine. Important attribute is `save_folder` and `name`.
    - year: int, year of the zarr file to sort.

    '''
    if os.path.exists(set.save_folder+ set.name+'_%s_sorted.zarr'%year):
        return 'File already exists'

    # open file
    zfile = zarr.open(set.save_folder + set.name+'_%s.zarr'%year, mode='r')
    
    # get order of trajectories based on the order of the starting particles
    T = get_traj(set,year)

    # write file with sorted trajectories

    zfile_sorted = zarr.open(set.save_folder+ set.name+'_%s_sorted.zarr'%year, mode='w')
    for key in zfile.keys():
        if key in ['obs','trajectory','isStuck']:
            zfile_sorted[key] = zfile[key]
            continue
        zfile_sorted[key] = zfile[key][T]
        print('Sorted', key)
        

def get_traj(set,year):
    '''
    Supplementary method to get the order of the trajectories based on the order of the starting particles.
    This method is used in the `sort_zarr` method.
    Order of trajectories is maintained by iteratively synchronizing the indices of the trajectories of the
    previous years.

    ## Parameters:
    - set: ReadSettings, settings for the advection routine. Important attribute is `save_folder`, `name`, and 
    `YEARS_PER_ITERATION`.
    - year: int, year of the zarr file to sort.
    '''
    # load initial file, get indices of trajectories
    zfile = zarr.open(set.save_folder + set.name+'_%s.zarr'%year, mode='r')
    t = zfile.trajectory[:]

    # iterate down to year 0 
    while year >0:
        year -= set.YEARS_PER_ITERATION
        zfile = zarr.open(set.save_folder + set.name+'_%s.zarr'%year, mode='r')

        # adapt indices of trajectories
        t = zfile.trajectory[t]
    
    # invert indices to apply to the file
    TT = np.array([t,np.arange(len(t))])
    TT = TT.T
    TT = TT[TT[:,0].argsort()]
    T = TT[:,1]
    return T


def chunk(set):
    '''
    Method to combine all sorted zarr files into a single zarr file. This is necessary for the analysis 
    of trajectories. Here the chunks are chosen for easy loading single trajectories rather than time steps.

    ## Parameters:
    - set: ReadSettings, settings for the advection routine. Important attribute is `save_folder` and `name`.    
    '''

    # load sorted files
    files = sorted(glob(set.save_folder+set.name+'_*_sorted.zarr'),key = lambda x: int(x.split('_')[-2]))
    # create new zarr file
    ds = zarr.open(set.save_folder+set.name+'_complete.zarr',mode = 'w')
    
    # choose vars that are relevant for analysis
    vars = ['z','lon','lat','HMXL','SALT','TEMP']
    
    # concatenate variables over time and save to new zarr file
    for var in tqdm(vars):
        print(var,end='... ')
        DS = []
        for file in files: 
            ds_new = zarr.open(file,mode = 'r')
            DS.append(ds_new[var])

        DS = np.concatenate(DS,axis = 1)
        
        # save to new zarr file with chunks optimized for trajectory loading
        ds[var] = zarr.array(DS,chunks = (100,None))
        print('done')


def clear_zarr(zarr_file):
    '''
    Maintenance to remove artificial time steps from zarr files that were created due to
    restarting the advection routine. The cleaned zarr file is saved as a new zarr file
    Should only be applied to *_complete.zarr files and is hard coded to a run time 
    of 8760 days (120 years with 5 day time steps).

    ## Parameters:
    - zarr_file: str, path to the zarr file to clean.

    '''

    # open zarr file and check length of time dimension
    ds = zarr.open(zarr_file, mode='r')
    l = len(ds.z[0])
    if l == 8760:
        print('nothing to do')
        return
    else:
        # remove artificial time steps, which are at the interface of restarts
        # (120//ll gives the runtime in years for each restart, 73 is the number of time steps per year)
        ll = l-8760
        print('removing %i points'%ll)
        c_mask = np.ones_like(ds.z[0],dtype = bool)
        c_mask[::120//ll*73+1] = False 
    print(len(ds.z[0][c_mask]))

    # save cleaned zarr file
    ds_new = zarr.open(zarr_file[:-5]+'2.zarr', mode='w')
    for var in ds:
        print(var)
        ds_new[var] = ds[var][:][:,c_mask]
