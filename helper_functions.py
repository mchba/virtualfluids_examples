import numpy as np
import meshio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import os
import fnmatch

"""
    Some functions that can be useful to postprocess pyfluids results.
"""

def load_horplane(filename):
    mesh = meshio.read(filename)
    x = mesh.points[:,0]
    y = mesh.points[:,1]
    z = mesh.points[:,2]
    vx = mesh.point_data['vx']
    parts = filename.split('_t_')
    nr = int(parts[1].split('.')[0])
    plane = {'x': x, 'y': y, 'z': z, 'vx': vx, 'nr': nr}
    return plane

def load_planes(fileprefix,outputf='output'):
    # Load all planes
    file_list = []
    plane_list = []
    nr_list = []
    minvx=1e10; maxvx=-1e10
    # Loop through all files in the directory
    for filename in os.listdir(outputf):
        # Check if the filename starts with 'planeProbeHorizontal_bin' and ends with '.vtu'
        if fnmatch.fnmatch(filename, fileprefix):
            file_list.append(filename)
            plane = load_horplane(outputf + '/' + filename)
            plane_list.append(plane)
            nr_list.append(plane['nr'])
            if np.min(plane['vx']) < minvx:
                minvx = np.min(plane['vx'])
            if np.max(plane['vx']) > maxvx:
                maxvx = np.max(plane['vx'])
        
    # Sort the planes according to timestep number
    sorted_lists = sorted(zip(nr_list, plane_list), key=lambda x: x[0])
    nr_list, plane_list = zip(*sorted_lists)
    
    planes = {'nr_list': nr_list, 'plane_list': plane_list, 'minvx': minvx, 'maxvx': maxvx}
    
    return planes

def plot_horplane(data,var='vx',varlim='None',plane='xy'):
    if varlim == 'None':
        varlim = [np.min(data[var]),np.max(data[var])]
    lvls = np.linspace(varlim[0], varlim[1], 100) # Use a lot of color levels to make it appear continous
    fig = plt.figure(figsize=(12,8))
    if plane=='xy':
        p = plt.tricontourf(data['x'], data['y'], data[var], lvls, cmap=cm.turbo)
        plt.xlabel('$x$')
        plt.ylabel('$y$',rotation=0,va='center',ha='right')
    if plane=='xz':
        p = plt.tricontourf(data['x'], data['z'], data[var], lvls, cmap=cm.turbo)
        plt.xlabel('$x$')
        plt.ylabel('$z$',rotation=0,va='center',ha='right')
    if plane=='yz':
        p = plt.tricontourf(data['y'], data['z'], data[var], lvls, cmap=cm.turbo)
        plt.xlabel('$y$')
        plt.ylabel('$z$',rotation=0,va='center',ha='right')    
    plt.axis('scaled')
    fig.subplots_adjust(right=0.8)
    cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    comcolRANS = plt.colorbar(p, cax=cbar, aspect=100)
    cbar.set_ylabel('$u$ [-]',rotation=0,va='center',ha='left')
    comcolRANS.set_ticks(np.arange(varlim[0],varlim[1],1.0))
    if not os.path.isdir('hor_pngs'):
        os.makedirs('hor_pngs')
    plt.savefig('hor_pngs/%s_plane_%d.png'%(plane,data['nr']),dpi=300,bbox_inches='tight')