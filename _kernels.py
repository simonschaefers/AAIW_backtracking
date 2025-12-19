########################################################################################################
# Author: Simon Schäfers
# Contact: sschaefers@geomar.de
# Date: 2025-01-21
########################################################################################################

# This script contains the kernels used in the advection routine. The kernels are used to apply boundary 
# conditions, cope with errors, and sample properties. 

from parcels import StatusCode


def periodic_bc(particle, fieldset,time):                                           
    """Kernel that applies periodic boundary conditions to the particle advection.
    If a particle exceeds the zonal boundaries, it will be moved to the opposite boundary.
    This Kernel is used in a parcels simulation (`pset.execute`)

    ## Parameters:
    - particle: class, the particle the boundary is applied to.
    - fieldset: class, the fieldset containing the field data
    - time: int, the time at which the particle is being advected.
    """
   
    if particle.lon < fieldset.halo_west:                                          
        particle_dlon += fieldset.halo_east - fieldset.halo_west                    
    elif particle.lon > fieldset.halo_east:
        particle_dlon -= fieldset.halo_east - fieldset.halo_west

def cope_errors(particle, fieldset, time):
    """Kernel that copes with errors in the particle advection.
    If a particle exceeds the field boundaries, it will be deleted.
    To avoid numerical errors, the depth of the particle will be set to 0.5 if it is below 0.5.
    If a particle crosses the surface nonetheless, it will be brought back to the surface.
    This Kernel is used in a parcels simulation (`pset.execute`)
    
    ## Parameters:
    - particle: class, the particle the errors are coped with.
    - fieldset: class, the fieldset containing the field data
    - time: int, the time at which the particle is being advected.
    """

    if particle.isStuck == 1:
        particle.delete()

    elif particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()
            print('particle deleted')

    elif particle.lat > -5:
            particle.delete()
            print('too far north')

    elif particle.state == StatusCode.ErrorThroughSurface or particle.depth < 0.5:
            particle.delete()
            print('surfaced')


def buoy(particle,fieldset,time):
    '''
    TEST KERNEL
    Kernel that tracks the velocities at the position of the particle. 
    This kernel is used to test accurate representations of velocities in the routine.
    This kernel is best used without advection, so the particles stay at the same position.
    '''
    particle.uvel = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    particle.vvel = fieldset.V[time, particle.depth, particle.lat, particle.lon]
    particle.wvel = fieldset.W[time, particle.depth, particle.lat, particle.lon]

def sample_props(particle, fieldset, time):
    # sample the mixed layer depth for the particle
    particle.HMXL = fieldset.HMXL[time, particle.depth, particle.lat, particle.lon]
    if particle.HMXL > particle.depth:
        particle.up = 1
def sample_props_mean(particle, fieldset, time):
    # sample the mean mixed layer depth for the particle
    particle.HMXL_mean = fieldset.HMXL_mean[time, particle.depth, particle.lat, particle.lon]
    if particle.HMXL_mean > particle.depth:
        particle.up = 1
def sample_all_props(particle, fieldset, time):
    # sample the mixed layer depth for the particle
    particle.HMXL = fieldset.HMXL[time, particle.depth, particle.lat, particle.lon]
    if particle.HMXL > particle.depth:
        particle.up = 1
    particle.SALT = fieldset.SALT[time, particle.depth, particle.lat, particle.lon]
    particle.TEMP = fieldset.TEMP[time, particle.depth, particle.lat, particle.lon]

#--------------------------------------------------------------------------------------
