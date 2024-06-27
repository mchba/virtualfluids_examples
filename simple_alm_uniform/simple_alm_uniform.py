r"""
=======================================================================================
 ____          ____    __    ______     __________   __      __       __        __
 \    \       |    |  |  |  |   _   \  |___    ___| |  |    |  |     /  \      |  |
  \    \      |    |  |  |  |  |_)   |     |  |     |  |    |  |    /    \     |  |
   \    \     |    |  |  |  |   _   /      |  |     |  |    |  |   /  /\  \    |  |
    \    \    |    |  |  |  |  | \  \      |  |     |   \__/   |  /  ____  \   |  |____
     \    \   |    |  |__|  |__|  \__\     |__|      \________/  /__/    \__\  |_______|
      \    \  |    |   ________________________________________________________________
       \    \ |    |  |  ______________________________________________________________|
        \    \|    |  |  |         __          __     __     __     ______      _______
         \         |  |  |_____   |  |        |  |   |  |   |  |   |   _  \    /  _____)
          \        |  |   _____|  |  |        |  |   |  |   |  |   |  | \  \   \_______
           \       |  |  |        |  |_____   |   \_/   |   |  |   |  |_/  /    _____  |
            \ _____|  |__|        |________|   \_______/    |__|   |______/    (_______/

  This file is part of VirtualFluids. VirtualFluids is free software: you can
  redistribute it and/or modify it under the terms of the GNU General Public
  License as published by the Free Software Foundation, either version 3 of
  the License, or (at your option) any later version.

  VirtualFluids is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
  for more details.

  SPDX-License-Identifier: GPL-3.0-or-later
  SPDX-FileCopyrightText: Copyright © VirtualFluids Project contributors, see AUTHORS.md in root folder

  Author: mchba

  Description:
    Two actuator line (AL) turbines in uniform inflow.

=======================================================================================
"""
#%% Import python stuff
import numpy as np
from pathlib import Path
from pyfluids import gpu, parallel
import sys
sys.path.append('../') # To be able to helper functions
from helper_functions import load_planes, plot_horplane

#%%
run_simulation = 1
if run_simulation==1:
    #%% Define simulation name and create output folder
    sim_name = "simple_alm_uniform"
    output_path = Path(__file__).parent/Path("output") # path to the output folder
    output_path.mkdir(exist_ok=True) # Creates the output folder, if it doesn't already exist
    
    #%% Initialize some different objects
    grid_builder = gpu.grid_generator.MultipleGridBuilder()
    communicator = parallel.MPICommunicator.get_instance()
    para = gpu.Parameter(1, communicator.get_process_id())
    bc_factory = gpu.BoundaryConditionFactory()
    
    #%%
    # Turbine parameters
    tip_speed_ratio = 7.5
    turbine_diameter = 126.0
    nodes_per_diameter = 32
    n_blade_nodes = 32
    
    # Grid parameters
    length = np.array([8,3,3])*turbine_diameter  # domain size
    level = 0 # from which grid level that the actuator line extracts velocities 
    
    # Flow parameters
    viscosity = 1.56e-5
    velocity  = 10.0
    mach = 0.1
    density = 1.225
    
    # Turbine layout
    Nturbines = 2
    turbine_positions_x = np.array([1, 2])*turbine_diameter
    turbine_positions_y = np.ones(Nturbines)*length[1]/2
    turbine_positions_z = np.ones(Nturbines)*length[2]/2
    
    # Time parameters
    t_start_out        = 15.0      # when to start outputting flow field (actual the first output is at t_start_out + t_out)
    t_out              =  5.0      # delta time at which the whole flow field is output
    t_end              = 20.0      # total time of simulation
    t_start_averaging  =  0.0      # I think this is used to decide when to start calculating temporal statistics to the plane output
    t_start_out_probe  =  0.0      # start time of plane output (needs to be >= t_start_averaging, even if only the instantaneous values are output to the planes!)
    t_out_probe        =  1.0      # delta time of plane output
    
    #%% Calculate time and space discretization (these are the physical dx and dt).
    dx = turbine_diameter/nodes_per_diameter
    dt = dx * mach / (np.sqrt(3) * velocity)
    
    #%% Set various parameters
    para.set_output_prefix(sim_name) # use sim_name as prefix for the flow field output filename.
    para.set_print_files(True) # Controls if the flow field will be output (note, the initial flow field will always be output)
    
    # Calculate LBM units (I think these will be used to scale the grid and variables in the simulations)
    velocity_ratio = dx/dt
    velocity_LB = velocity / velocity_ratio  # LB units
    viscosity_LB = viscosity / (velocity_ratio * dx)  # LB units
    para.set_velocity_LB(velocity_LB)
    para.set_viscosity_LB(viscosity_LB)    
    para.set_velocity_ratio(dx/dt)
    para.set_viscosity_ratio(dx*dx/dt)
    
    # Set kernel and limiters
    para.configure_main_kernel(gpu.Kernel.compressible.K17CompressibleNavierStokes)
    para.set_quadric_limiters(1.0, 1.0, 1.0) # the apps/gpu/ActuatorLine example had these
    
    # Set when to output flow field and total simulation time
    para.set_timestep_start_out(int(t_start_out/dt))
    para.set_timestep_out(int(t_out/dt))
    para.set_timestep_end(int(t_end/dt))
    
    # Need to be turned on in order to use the ALM.
    para.set_is_body_force(True)
    
    #%% Set turbulence model ("Smagorinsky", "AMD", "QR" or "None"; set SGS constant)
    tm_factory = gpu.TurbulenceModelFactory(para)
    tm_factory.set_turbulence_model(gpu.TurbulenceModel.QR)
    tm_factory.set_model_constant(0.3333333)
    
    #%% Create grid and BCs
    # Enable scaling of grid
    grid_scaling_factory = gpu.GridScalingFactory()
    grid_scaling_factory.set_scaling_factory(gpu.GridScaling.ScaleCompressible)
    
    # Define grid and build it
    grid_builder.add_coarse_grid(0.0, 0.0, 0.0, length[0], length[1], length[2], dx)
    grid_builder.build_grids(False) # ("False"=turn off thin wall stuff)
    
    # Important that BCs are first set after the grid has been build! I think M=minus and P=plus, while the second latter corresponds to direction.
    grid_builder.set_velocity_boundary_condition(gpu.SideType.MX, velocity_LB, 0.0, 0.0) # inlet
    grid_builder.set_velocity_boundary_condition(gpu.SideType.MY, velocity_LB, 0.0, 0.0) # right side
    grid_builder.set_velocity_boundary_condition(gpu.SideType.PY, velocity_LB, 0.0, 0.0) # left side
    grid_builder.set_velocity_boundary_condition(gpu.SideType.MZ, velocity_LB, 0.0, 0.0) # bottom
    grid_builder.set_velocity_boundary_condition(gpu.SideType.PZ, velocity_LB, 0.0, 0.0) # top
    grid_builder.set_pressure_boundary_condition(gpu.SideType.PX, 0.0)                   # outlet
    
    # Need to "activate" the types of BCs we want to use in the bc_factory as well.
    bc_factory.set_velocity_boundary_condition(gpu.VelocityBC.VelocityWithPressureInterpolatedCompressible)
    bc_factory.set_pressure_boundary_condition(gpu.PressureBC.OutflowNonReflective)
    
    #%% Initial flow field
    para.set_initial_condition_uniform(velocity_LB, 0.0, 0.0)
    
    #%% Add turbines
    smearing_width = 2.0*dx
    blade_radius = 0.5*turbine_diameter
    hub_height_velocity = velocity
    rotor_speeds = np.ones(Nturbines)*tip_speed_ratio*hub_height_velocity/blade_radius
    alm = gpu.ActuatorFarmStandalone(turbine_diameter, n_blade_nodes, turbine_positions_x, turbine_positions_y, turbine_positions_z, rotor_speeds, density, smearing_width, level, dt, dx)
    #alm.enable_output("ALM", int(t_start_out_probe/dt), int(t_out_probe/dt) ) # This will output forces and velocities on the ALs.
    para.add_actuator(alm)
    
    #%% Output 
    # Add a horizontal plane at hub height
    xyplane_probe = gpu.probes.PlaneProbe("xyplane", para.get_output_path(), int(t_start_averaging/dt), 10, int(t_start_out_probe/dt), int(t_out_probe/dt))
    xyplane_probe.set_probe_plane(0, 0, length[2]/2, length[0], length[1], dx)
    xyplane_probe.add_statistic(gpu.probes.Statistic.Instantaneous) # only output instanteneous flow fields to the plane
    #xyplane_probe.add_all_available_statistics()  # can be turned on, if we also want to include calculate temporal statistics (mean and variance)
    para.add_probe(xyplane_probe)
    
    #%% Run simulation
    sim = gpu.Simulation(para, grid_builder, bc_factory, tm_factory, grid_scaling_factory)
    sim.run()

#%% Postprocess (can take some time to plot, so be careful with this)
run_postprocess = 0
if run_postprocess == 1:
    # Load all the planes
    xyplanes = load_planes('xyplane_bin*.vtu')
    
    # Plot the planes and output to output plots to the hor_pngs folder.
    for i in range(len(xyplanes['nr_list'])):
        plot_horplane(xyplanes['plane_list'][i],var='vx',varlim=[4,11],plane='xy');

