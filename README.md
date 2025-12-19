The directory is structured as follows:
- `code` folder with python routines that include:
	- `particle_advection.py` contains the functions that execute the particle advection on a given field with given settings. This file has three subfiles: `_settings.py`,  `_kernels.py`, and `_start_conditions.py`
		- `_settings.py` contains the settings for the particle advection. It ensures consistent settings for the considered experiments, and simultaneously allows for easy adaptions 
		- `_kernels.py` contains the kernels that ensure smooth advection and tracking of variables along the particle trajectories
		- `_start_conditions.py` sets the initial particle positions and release time. While only `from_csv()` is used and shown, particle start positions can be freely defined.
	- `zarr_routines.py` are used to prepare the raw `.zarr`output for analysis
	- `metadata_production.py` contains routines to extract "metadata" such as subduction time and location, as well as mapped trajectories from the `.zarr`files of the Lagrangian advection output (adapted with the help of `zarr_routines.py`). These data are stored in easy to handle `.csv`files to be used for the analysis. 
	- `plotter.py` contains routines to visualise and quantify the results.
		- `_plotter_helper.py`contains support functions for plotting.

> #### Files only available in zenodo 
>  - `metadata` folder contains `.csv` subduction files of each experiment, where time, loaction, and properties at subduction for each particle is stored. It also contains `.zarr` pathway files, that contain the track of each particle mapped onto a 2x2° grid
> - `trajectory` folder contains the original trajectories (`.zarr` files) with a time step of 5 days. Here, only lon, lat, depth, and mixed layer depth (HMXL) are presented due to storage limitations.
