########################################################################################################
# Author: Simon Schäfers
# Contact: sschaefers@geomar.de
# Date: 2025-01-21

########################################################################################################

# This script contains the main function to execute the advection routine. The function creates the 
# fieldset and particleset, defines the kernels, and executes the advection routine. The output is a 
# zarr file. It contains the main execution function and additional functions to create the fieldset
# and particleset, and to update the settings file after the advection routine.


import zarr
import numpy as np
from datetime import datetime
from parcels import FieldSet,Field,ParticleSet,AdvectionRK4_3D
import src.advection_routine._settings as setts
from src.advection_routine._kernels import *


def execution(set = setts.Settings(),name = None,execution = False):
    '''
    Main function to execute the advection routine. This function creates the fieldset and particleset,
    defines the kernels, and executes the advection routine. The output is stored in a zarr file.

    ## Parameters:
    - set: Settings object or str, the settings for the advection routine. If a string is provided, 
    the function will read the settings from a .txt file.
    - name: str, the name of the run. If None, the name will be set to the current date and time.
    - execution: bool, if True, the advection routine will be executed, otherwise only the fieldset and
    particleset will be created.
    '''

    # check wether set is a Settings object or a path to a .txt file
    if type(set) == str: 
        if set[-4:] == '.txt':  
            set = setts.ReadSettings(set)
        else: raise ValueError('set must be a Settings object or a path to a .txt file')
    assert type(set) in [setts.Settings,setts.ReadSettings], 'set must be a Settings object'

    # set name of the run
    set.name = 'test_'+datetime.now().strftime('%Y%m%d%H%M%S') if name is None else name
    print('Run name: %s'%set.name)

    # ---------------MAIN SETUP-------------------------------------------------------------------------
    # create fieldset and particleset, and define kernels
    fs = create_fieldset(set)
    ps = create_particleset(fs,set)
    kernels = [globals()[kernel] for kernel in set.kernels]
    # --------------------------------------------------------------------------------------------------

    

    # ---------------EXECUTION--------------------------------------------------------------------------
    if execution:
        # create output file, currently stored in memory
        output_memorystore = zarr.storage.MemoryStore()
        output_file = ps.ParticleFile(name=output_memorystore, outputdt=set.TS_SAVE)

        # execute advection routine
        ps.execute(kernels, runtime=set.runtime, dt=set.TS_COMP, output_file = output_file)

        # save output to zarr file
        output_dirstore_name = set.save_folder+"/%s_%i.zarr"%(set.name, set.N_YEARS)
        output_dirstore = zarr.storage.DirectoryStore(output_dirstore_name)
        zarr.convenience.copy_store(output_memorystore, output_dirstore)
        output_dirstore.close()

        # update current simulation year
        if hasattr(set,'filename'):
            update_setting_file(set, output_dirstore_name)
    
    else: return set,fs,ps,kernels

# ========================================================================================================
# ===================== fieldset and particleset creation ================================================

def create_fieldset(set):
    '''
    Function to create the fieldset for the advection routine. The function reads the field data based on 
    the provided settings and creates a fieldset object.

    ## Parameters:
    - set: Settings object, the settings for the advection routine.

    ## Returns:
    - fieldset: FieldSet object, the fieldset for the advection routine.
    '''

    # create fieldset for the velocities (currently only B-grid)
    fieldset = FieldSet.from_b_grid_dataset(
        filenames = {key: set.filenames[key] for key in set.vels},
        variables = {key: set.field_variables[key] for key in set.vels},
        dimensions = {key: set.field_dimensions[key] for key in set.vels},
        timestamps = set.timestamps['U'],
        time_periodic=set.fieldtime['U'],
        chunksize = set.chunking['U'],
    )

    # add fields for the other variables
    for var in set.vars:
        if var not in ['U','V','W']:
            print(var)
            globals()[var] = Field.from_netcdf(
                        filenames=set.filenames[var],
                        variable={var:var if var != 'HMXL_mean' else 'HMXL'},
                        dimensions=set.field_dimensions[var],
                        timestamps = set.timestamps[var],
                        time_periodic=set.fieldtime[var],
                        chunksize  = set.chunking[var]
                    )
            fieldset.add_field(globals()[var])

    # add scaling factors to the fieldset (POP uses cm/s for velocities and cm for HMXL)
    for var in set.scaling.keys():
        if set.scaling[var] is not None: fieldset.__dict__[var].set_scaling_factor(set.scaling[var])
        print('scaling factor for %s: %s'%(var,set.scaling[var]))

    # create halo for periodic boundary conditions
    fieldset.add_constant("halo_west", fieldset.U.grid.lon[0])                            
    fieldset.add_constant("halo_east", fieldset.U.grid.lon[-1])
    fieldset.add_periodic_halo(zonal=True) 

    return fieldset

# --------------------------------------------------------------------------------------------------------

def create_particleset(fieldset,set):
    '''
    Function that creates the particleset for the advection routine. The function reads the particle data 
    from a start condition (if starting) or from a previous particle .zarr file (if continuing) and 
    creates a particleset object.

    ## Parameters:
    - fieldset: FieldSet object, the fieldset for the advection routine.
    - set: Settings object, the settings for the advection routine.

    ## Returns:
    - particleset: ParticleSet object, the particleset for the advection routine.
    '''

    # if no previous particles are available, create new particles, or load from source
    if set.N_YEARS == 0:
        import src.advection_routine._start_conditions as sc
        # define the start condition in the set object
        particle_data = sc.start_condition(set, plot=False)

    # if previous particles are available, load their last position from the previous .zarr file
    else: 
        ds = zarr.open(set.old_file, mode='r')
        i = np.arange(ds['lon'].shape[0],dtype = int)
        t = np.zeros_like(i,dtype = int)
        t[:] = -1

        # -------------------------------------------------------------------------------------
        # if particles have not been started at the same time, use this for the first reload:
        # corrected for in the _zarr_routines.time_lag_correction function
        # if set.N_YEARS == set.YEARS_PER_ITERATION:
        #     ii,tt = np.where(ds['time'][:] == np.nanmin(ds['time'][0]))
        #     t[ii] = tt
        # -------------------------------------------------------------------------------------

        particle_data = {k: np.array(ds[k][i, t]) for k in ds.keys() if k not in ['trajectory','obs']}
        # filter out stuck particles by following criteria and move them to a new location
        particle_data = delete_stuck_particles(particle_data,ds,set)

        # change 'depth' to 'z' and remove 'time' to avoid conflicts with the particle class
        particle_data['depth']=particle_data['z']
        particle_data.pop('z')
        particle_data.pop('time')

    return ParticleSet(fieldset=fieldset,pclass=set.Particle,**particle_data)

# ========================================================================================================
# ===================== additional functions =============================================================

def update_setting_file(set,output_dirstore_name):
    '''
    Function to update the settings file with the new year of advection and the name of the output file.
    
    ## Parameters:
    - set: Settings object, the settings for the advection routine.
    - output_dirstore_name: str, the name of the output file.

    '''

    year_done,file_done = False,False
    # read file and change the line that starts with 'N_YEARS'
    with open(set.filename, 'r') as file:
        lines = file.readlines()
    with open(set.filename, 'w') as file:
        for line in lines:
            if line.startswith('N_YEARS'):
                file.write('N_YEARS : %i\n'%(set.N_YEARS+set.YEARS_PER_ITERATION))
                year_done = True
            elif line.startswith('old_file'):
                file.write('old_file : "%s"\n'%output_dirstore_name)
                file_done = True
            else:
                file.write(line)
        if not year_done: file.write('\nN_YEARS : %i\n'%(set.N_YEARS+set.YEARS_PER_ITERATION))
        if not file_done: file.write('old_file : "%s"\n'%output_dirstore_name)


# --------------------------------------------------------------------------------------------------------
def delete_stuck_particles(transferred_data,ds,set):
    '''
    Function to detect and remove stuck particles from the particle data. The function checks for 
    particles that are stuck in the horizontal or vertical direction, too far north, or too close to 
    the surface.

    ## Parameters:
    - transferred_data: dict, the particle data from the previous year.
    - ds: zarr object, the particle data from the previous year.
    - set: Settings object, the settings for the advection routine.

    ## Returns:
    - transferred_data: dict, the particle data with the stuck particles removed.
    '''

    print('[INFO] detecting & (re)moving stuck particles')
    stuck_mask = np.zeros_like(transferred_data['lon'],dtype = bool)

    # define the criteria for stuck particles
    stuck_particles =  {'horiz_stuck':(np.std(np.array(ds.lon[:,-set.stuck_timer:]),axis = 1) == 0) |
                          (np.std(np.array(ds.lon[:,-set.stuck_timer:]),axis = 1) == 0),
                        'vertic_stuck':  (np.std(np.array(ds.z[:,-set.stuck_timer:]),axis = 1) == 0),
                        'too_far_north':      (transferred_data['lat'] > set.north_limit),
                        'close to surface':   (transferred_data['z'] < 0.5),
                        'deleted':  np.isnan(transferred_data['lon']),
    }   

    # print the status of the particles and combine the criteria
    print('[INFO] particle status:')
    for key in stuck_particles.keys():
        print(key,': ',sum(stuck_particles[key]))
        stuck_mask |= stuck_particles[key]

    for key in transferred_data.keys():
        transferred_data[key][stuck_mask] = 0
    transferred_data['isStuck'][stuck_mask] = 1
    transferred_data['lon'][stuck_mask] = 300
    transferred_data['lat'][stuck_mask] = -10
    transferred_data['z'][stuck_mask] = 1000
    print('[INFO] detected %i new stuck particles'
        %(sum(stuck_mask)-sum(stuck_particles['deleted']))) 
    
    return transferred_data
