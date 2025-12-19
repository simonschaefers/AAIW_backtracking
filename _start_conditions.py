########################################################################################################
# Author: Simon Schäfers
# Date: 2025-01-27

########################################################################################################

# This script contains start condition methods for the particles. The `start_condition` function 
# initializes a chosen start condition method. The `from_csv` method reads the particle data from a 
# csv file and returns the particle data as a dictionary, containing lon, lat, depth, and optionally 
# time of release.

import pandas as pd

def start_condition(set,plot = False):
    '''
    Main function to initialize the start conditions for the particles.
    Initializes the chosen start condition method based on the settings.

    ## Parameters:
    - set: Settings, settings for the advection routine. Important attribute is `start_particles`.
    
    ## Returns:
    - dict, dictionary containing the particle data (lon, lat, depth, and optionally time)
    '''
    return globals()[set.start_particles](set,plot = plot)

def from_csv(set,plot = False):
    '''
    Reads the particle data from a csv file and returns the particle data as a dictionary, containing
    lon, lat, depth, and optionally time of release. Longitude is converted to the range [0,360] if the 
    settings contain `lon_lims` with negative values.

    ## Parameters:
    - set: Settings, settings for the advection routine. Important attribute is `particle_file`. 
    Further, the settings can contain `lon_lims`, `lat_lims`, and `depth_lims` to mask the particle data.

    ## Returns:
    - dict, dictionary containing the particle data (lon, lat, depth, and optionally time)

    '''

    particle_data = pd.read_csv(set.particle_file)

    # mask particle data based on lon, lat, and depth limits
    if hasattr(set,'lon_lims'):
        lon_lims = [lon_lim if lon_lim > 0 else lon_lim+360 for lon_lim in set.lon_lims]
        if lon_lims[0] < lon_lims[1]:
            lon_mask = (particle_data['lon'] >= lon_lims[0]) & (particle_data['lon'] <= lon_lims[1])
        else:
            lon_mask = (particle_data['lon'] >= lon_lims[0]) | (particle_data['lon'] <= lon_lims[1])
    if hasattr(set,'lat_lims'):
        lat_mask = (particle_data['lat'] >= set.lat_lims[0]) & (particle_data['lat'] <= set.lat_lims[1])
    if hasattr(set,'depth_lims'):
        depth_mask = (particle_data['depth'] >= set.depth_lims[0]) & (particle_data['depth'] <= set.depth_lims[1])
    mask = lon_mask & lat_mask & depth_mask

    # include time if available
    if hasattr(particle_data,'time'):
        return {'lon':particle_data['lon'][mask],
                'lat':particle_data['lat'][mask],
                'depth':particle_data['depth'][mask],
                'time':particle_data['time'][mask].astype('datetime64[ns]')}
    else:
        return {'lon':particle_data['lon'][mask],
                'lat':particle_data['lat'][mask],
                'depth':particle_data['depth'][mask]}