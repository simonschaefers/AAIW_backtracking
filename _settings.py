########################################################################################################
# Author: Simon Schäfers
# Contact: sschaefers@geomar.de
# Date: 2025-01-21

########################################################################################################

# This script contains the settings for the advection routine. The `Settings` class contains the default 
# settings for the advection routine. The `ReadSettings` class reads the settings from a file and updates
# the default settings accordingly. The `calc_settings` method calculates additional settings based on the
# provided settings.

import numpy as np
from parcels import JITParticle, Variable
from datetime import timedelta
from glob import glob


class Settings():
    '''
    Settings class for the advection routine. Contains the default settings for the advection routine.
    
    ## Methods:
    - __init__: initializes the settings.
    - calc_settings: calculates additional settings based on the provided settings.

    ## Parameters:
    - kwargs: dict, import adiitional settings for the advection routine. 
    '''
    def __init__(self, **kwargs):
        # -----------------------------------------------------------------------------------------------
        # default fieldset settings
        self.year = 2000
        self.vels = ['U','V','W']
        self.vars = ['HMXL','TEMP','SALT']

        self.main_folder = '/work/uo0780/u301534/Masterarbeit/'
        self.save_folder = self.main_folder + 'zarr_storage/'
        self.folders = {'U':    self.main_folder     + 'Data_velocities/',
                        'V':    self.main_folder     + 'Data_velocities/',
                        'W':    self.main_folder     + 'Data_velocities/',
                        'HMXL': self.main_folder     + 'Data_HMXL/',
                        'HMXL_mean': self.main_folder+ 'Data_HMXL/',
                        'TEMP': self.main_folder     + 'Data_TS/',
                        'SALT': self.main_folder     + 'Data_TS/'
                        }
        self.time = [0,365]
        self.field_variables =  {"U": 'UVEL',"V": 'VVEL',"W": 'WVEL'}       

        self.chunking = {'U':None,'V':None,'W':None,'HMXL':None,'HMXL_mean':None,'TEMP':None,'SALT':None}
        self.TS_VARS = {'U': 1,'V': 1,'W': 1,'HMXL': 1,'HMXL_mean': 1,'TEMP': 5,'SALT': 5}

        # -----------------------------------------------------------------------------------------------
        # default particleset and kernel settings
        self.N_YEARS = 0  

        self.start_particles = 'from_csv'#'from_csv'
        self.particle_file = self.main_folder + '/metadata/Global_AAIW_particles-POP%s.csv'%self.year

        self.lon_lims = [0,360]
        self.lat_lims = [-60,-20]
        self.depth_lims = [0,2000]
        self.definition = None

        self.stuck_timer = 30
        self.north_limit = -5

        self.kernels = ['periodic_bc','AdvectionRK4_3D']
        self.bouy = False

        # -----------------------------------------------------------------------------------------------
        # default execution settings
        self.YEARS_PER_ITERATION = 2
        self.isBacktrack = True
        self.TS_SAVE = 5
        self.TS_COMP = 1
 
        # -----------------------------------------------------------------------------------------------
        # update settings from kwargs
        self.__dict__.update(kwargs)
        # -----------------------------------------------------------------------------------------------
        # calculate additional settings
        self.calc_settings()

    def calc_settings(self):
        '''
        calc_settings method calculates additional settings based on the provided settings.
        '''
        # -----------------------------------------------------------------------------------------------
        # final fieldset settings
        self.vars = self.vels + self.vars 
        if not hasattr(self, 'files'):
            self.files = {k: sorted(glob(self.folders[k]+str(self.year)+'*%s.nc'%k)) for k in self.vars}
        self.field_dimensions = {k: {"lon": "XU", "lat": "YU", "depth": "W_DEP"} for k in self.vars}
        self.filenames = {k: 
                           {'lon':self.files['U'][0],
                            'lat':self.files['U'][0],
                            'depth':self.files['W'][0],
                            'data':self.files[k]
                            } for k in self.vars
                        }
        for k in ['HMXL','HMXL_mean']:
            if k in self.vars:
                self.filenames[k].pop('depth')
                self.field_dimensions[k].pop('depth')

        dates = [np.timedelta64(i,'D') for i in range(self.time[0],self.time[1])]
        timestamps = np.expand_dims(np.datetime64('1970-01-01')+dates,axis = 1)
        self.timestamps = {k: timestamps[::self.TS_VARS[k]] for k in self.vars}


        self.fieldtime = {k: timedelta(days = (self.time[1]-self.time[0])) for k in self.vars}

        self.scaling = {k: 1/100 for k in self.vars if k in ['U','V','HMXL','HMXL_mean']}
        self.scaling['W'] = -1/100

        # -----------------------------------------------------------------------------------------------
        # final particleset and kernel settings
        particle_variables = [Variable("isStuck",initial = 0),Variable('up',initial = 0)]
        for variable in ['HMXL','HMXL_mean','TEMP','SALT']:
            if variable in self.vars:
                particle_variables.append(Variable(variable,initial = 0))

        if self.bouy:
            self.kernels.append('buoy')
            self.kernels.remove('AdvectionRK4_3D')
            particle_variables.append(Variable('uvel',initial = 0))
            particle_variables.append(Variable('vvel',initial = 0))
            particle_variables.append(Variable('wvel',initial = 0))

        if all([k in self.vars for k in ['HMXL','SALT','TEMP']]):
            self.kernels.append('sample_all_props')    
        elif 'HMXL' in self.vars:
            self.kernels.append('sample_props')
        elif 'HMXL_mean' in self.vars:
            self.kernels.append('sample_props_mean')
                                              
        self.kernels.append('cope_errors')
        self.Particle = JITParticle.add_variables(particle_variables)
        
        # -----------------------------------------------------------------------------------------------
        # final execution settings
        if type(self.TS_SAVE) == int:
            self.stuck_timer = self.stuck_timer//self.TS_SAVE
            self.TS_SAVE = timedelta(days=self.TS_SAVE) 
        if type(self.TS_COMP) == int:
            self.TS_COMP = (-1)**self.isBacktrack * timedelta(days=self.TS_COMP)
        self.runtime = timedelta(days=(self.time[1]-self.time[0])*self.YEARS_PER_ITERATION+1)

#========================================================================================================
class ReadSettings(Settings):
    '''
    ReadSettings class reads the settings from a file and updates the default settings accordingly, by
    adding read attributes to the kwargs dictionary.
    '''
    def __init__(self,filename, **kwargs):
        self.filename = filename
        if kwargs == {}:
            kwargs = {}
        with open(filename) as f:
            for line in f:
                if line[0] == '#': continue
                try: 
                    key,value = line.split(':')
                    key,value = key.strip(),value.strip()
                    kwargs[key] = eval(value)
                except:
                    kwargs[line.strip()] = True
        super().__init__(**kwargs)
