#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:38:10 2025

@author: mengshutang
"""
import numpy as np
import beam_info as bi
import visual as vis
from openpmd_viewer import OpenPMDTimeSeries
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from multiprocessing import Pool
import os

def get_traj_info(params):
    """
    get single trajectory plot info
    """
    dirs = params[0]
    beam_info = params[1]
    L_plot = params[2]
    ts = OpenPMDTimeSeries(dirs + '/diags/hdf5')
    
    beam = bi.get_beam_info(ts = ts,beam = 'driver',iteration = beam_info['iteration'],tail_charge = beam_info['tail_c'],tot_charge = beam_info['tot_c'],dt = beam_info['dt'],num_bins = 100)

    # 2D plot info
    plt_traj = vis.plot_trajectory(beam = beam)
    
    z_sim,beta_sim = plt_traj.bf_in_sims(section = 'tail',axes = None,color = None,linestyle = None,linewidth = None,label = None)
        
    z_vac,beta_vac = plt_traj.bf_in_vac(section = 'tail',L = L_plot,axes = None,color = None,linestyle = None,linewidth = None,label = None)
        
    z,sig = plt_traj.full_beam_size_evln(section = 'tail', L = L_plot, axes = None,color = None,linestyle = None,linewidth = None,label = None)

    return [np.concatenate((z_sim,z_vac)),np.concatenate((beta_sim,beta_vac)),z,sig]

def get_traj_info2(params):
    """
    get single trajectory plot info
    """
    q,dirs,beam_info,L_plot = params
    
    ts = OpenPMDTimeSeries(dirs + '/diags/hdf5')
    
    beam = bi.get_beam_info(ts = ts,beam = 'driver',iteration = beam_info['iteration'],tail_charge = beam_info['tail_c'],tot_charge = beam_info['tot_c'],dt = beam_info['dt'],num_bins = 100)

    # 2D plot info
    plt_traj = vis.plot_trajectory(beam = beam)
    
    z_sim,beta_sim = plt_traj.bf_in_sims(section = 'tail',axes = None,color = None,linestyle = None,linewidth = None,label = None)
        
    z_vac,beta_vac = plt_traj.bf_in_vac(section = 'tail',L = L_plot,axes = None,color = None,linestyle = None,linewidth = None,label = None)
        
    z,sig = plt_traj.full_beam_size_evln(section = 'tail', L = L_plot, axes = None,color = None,linestyle = None,linewidth = None,label = None)

    q.put([np.concatenate((z_sim,z_vac)),np.concatenate((beta_sim,beta_vac)),z,sig])


def multiple_beam_sizes(dirs,data_lst,c_lst,beam_params,L_plot):
    
    start = time.time()
        
    # number of processes
    cores = len(data_lst)
    
    with Pool(cores) as p:
        trajs = p.map(get_traj_info, [(dirs+dat, beam_params,L_plot) for dat in data_lst])
    
    fig_beta,axes_beta = plt.subplots(1,2,figsize = (12,5))
        
    for count in range(len(data_lst)):
        data = data_lst[count]
        axes_beta[0].plot(trajs[count][0],trajs[count][1],c = c_lst[count],label = data)
        axes_beta[1].plot(trajs[count][2],trajs[count][3],c = c_lst[count],label = data)
        axes_beta[0].legend()
        axes_beta[0].set_yscale('log')
        axes_beta[1].yaxis.tick_right()
        axes_beta[1].yaxis.set_label_position('right')
            
        axes_beta[0].set_title('Betafunction')
        axes_beta[1].set_title('Beam transverse size')
            
        axes_beta[0].set_ylabel(r'$\beta$ (cm)')
        axes_beta[0].set_xlabel(r'$z$ (cm)')
        axes_beta[1].set_ylabel(r'$\sigma$ ($\mu$m)')
        axes_beta[1].set_xlabel(r'$z$ (cm)')
        axes_beta[0].tick_params(right=True,direction = 'in')
        axes_beta[1].tick_params(left=True,direction = 'in')
        
        axes_beta[1].set_ylim((0,50))
    
    end = time.time()
    print(end-start)
    
def use_process(dirs,data_lst,c_lst,beam_params,L_plot):
    
    start = time.time()
    print('program starting...')
        
    processes = []
    trajs = []
    
    for data in data_lst:
        #ctx = mp.get_context('spawn')
        q = mp.Queue()
        params = (q,dirs+data, beam_params,L_plot)
        p = mp.Process(target = get_traj_info2, args = (params,))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    if not q.empty():
        print(q.get())
        trajs.append(q.get())
    
    fig_beta,axes_beta = plt.subplots(1,2,figsize = (12,5))
        
    for count in range(len(data_lst)):
        data = data_lst[count]
        axes_beta[0].plot(trajs[count][0],trajs[count][1],c = c_lst[count],label = data)
        axes_beta[1].plot(trajs[count][2],trajs[count][3],c = c_lst[count],label = data)
        axes_beta[0].legend()
        axes_beta[0].set_yscale('log')
        axes_beta[1].yaxis.tick_right()
        axes_beta[1].yaxis.set_label_position('right')
            
        axes_beta[0].set_title('Betafunction')
        axes_beta[1].set_title('Beam transverse size')
            
        axes_beta[0].set_ylabel(r'$\beta$ (cm)')
        axes_beta[0].set_xlabel(r'$z$ (cm)')
        axes_beta[1].set_ylabel(r'$\sigma$ ($\mu$m)')
        axes_beta[1].set_xlabel(r'$z$ (cm)')
        axes_beta[0].tick_params(right=True,direction = 'in')
        axes_beta[1].tick_params(left=True,direction = 'in')
        
        axes_beta[1].set_ylim((0,50))
    
    end = time.time()
    print(end-start)
    
def serial(dirs,data_lst,c_lst,beam_params,L_plot):
    
    start = time.time()
         
    fig_beta,axes_beta = plt.subplots(1,2,figsize = (12,5))
        
    for count in range(len(data_lst)):
        data = data_lst[count]
        trajs = get_traj_info(params = (dirs+data, beam_params,L_plot))
        
        axes_beta[0].plot(trajs[0],trajs[1],c = c_lst[count],label = data)
        axes_beta[1].plot(trajs[2],trajs[3],c = c_lst[count],label = data)
        axes_beta[0].legend()
        
        axes_beta[0].set_yscale('log')
        axes_beta[1].yaxis.tick_right()
        axes_beta[1].yaxis.set_label_position('right')
            
        axes_beta[0].set_title('Betafunction')
        axes_beta[1].set_title('Beam transverse size')
            
        axes_beta[0].set_ylabel(r'$\beta$ (cm)')
        axes_beta[0].set_xlabel(r'$z$ (cm)')
        axes_beta[1].set_ylabel(r'$\sigma$ ($\mu$m)')
        axes_beta[1].set_xlabel(r'$z$ (cm)')
        axes_beta[0].tick_params(right=True,direction = 'in')
        axes_beta[1].tick_params(left=True,direction = 'in')
        
        axes_beta[1].set_ylim((0,50))
    
    end = time.time()
    print(end-start)
    
    
    
    
    
    
    
    
    
    
    