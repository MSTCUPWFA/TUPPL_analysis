#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:00:10 2024

@author: mengshutang

Functions for data analysis in HiPACE

Features include batch emittance, twiss parameters, energy gain calculations; 
Particle tracking for multiple slices;
"""
import numpy as np
import pandas as pd
from openpmd_viewer import OpenPMDTimeSeries, ParticleTracker

########### Functions for calculations

def cu(coord):
    ### change coordinate unit to um
    return coord*10**6

def P_to_E(P):
    ### convert normalized momentum to energy
    me = 9.1093837 * 10**(-31)
    clight = 3*10**8
    J_to_eV = 6.24152272506e+18
    return P*me*clight**2*J_to_eV/10**9

### emittance calculation
def second_central_moment(x,y):
    ### calculate variance
    
    N = len(x)
    return np.sum(x*y)/N - np.sum(x)*np.sum(y)/N**2

def ps_emittance_n_rms(x,px):
    ### normalized phase space emittance
    ### should be the same as ts_emittance_n_rms (in principle?)
    
    bracket_x_sqrd = second_central_moment(x,x)
    bracket_px_sqrd = second_central_moment(px,px)
    bracket_x_px = second_central_moment(x,px)
    
    return np.sqrt(bracket_x_sqrd + bracket_px_sqrd - bracket_x_px**2)

def ts_emittance_rms(x,x_prime):
    ### geometric emittance

    bracket_x_sqrd = second_central_moment(x,x)
    bracket_x_prime_sqrd = second_central_moment(x_prime,x_prime)
    bracket_x_x_prime = second_central_moment(x,x_prime)

    return np.sqrt(bracket_x_sqrd*bracket_x_prime_sqrd-bracket_x_x_prime**2)

def ts_emittance_n_rms(x,pz,x_prime):
    ### normalized trace space emittance

    bracket_x_sqrd = second_central_moment(x,x)
    bracket_x_prime_sqrd = second_central_moment(x_prime,x_prime)
    bracket_x_x_prime = second_central_moment(x,x_prime)

    return np.mean(pz)*np.sqrt(bracket_x_sqrd*bracket_x_prime_sqrd-bracket_x_x_prime**2)

def twiss_params(x,x_prime):
    ### calculate twiss parameters alpha, beta, and gamma
    geometric_emit = ts_emittance_rms(x,x_prime)
        
    alpha = -second_central_moment(x,x_prime)/geometric_emit
    beta = second_central_moment(x,x)/geometric_emit
    gamma = second_central_moment(x_prime,x_prime)/geometric_emit

    return alpha,beta,gamma

def get_everything(beam,ts,select):
    ### return parameters in SI unit, except emittance is in um
    
    iteration_lst = ts.iterations
    
    # 1st row ps emittance in um; 2nd row ts geometric emittance; 3nd row normalized ts emittance in um
    emit = np.empty((3,len(iteration_lst)))
    
    # rows in order of alpha, beta, gamma
    twiss = np.empty((3,len(iteration_lst)))
    
    # 1st row energy; 2nd row its std
    energy = np.empty((2,len(iteration_lst)))
                      
    L_travel = np.empty(len(iteration_lst))

    for iter_index in range(len(iteration_lst)):
        x,y,z,px,py,pz,w = ts.get_particle(species = beam,iteration = iteration_lst[iter_index], var_list = ['x','y','z','ux','uy','uz','w'],select=select)

        # calculate emittance
        x_um = x*10**6
        x_p = px/pz
        
        emit[0][iter_index] = ps_emittance_n_rms(x_um,px)
        emit[1][iter_index] = ts_emittance_rms(x_um,x_p)
        emit[2][iter_index] = ts_emittance_n_rms(x_um,pz,x_p)

        # calculate twiss parameters
        x_p = px/pz
        
        alpha,beta,gamma = twiss_params(x,x_p)
        twiss[0][iter_index] = alpha
        twiss[1][iter_index] = beta
        twiss[2][iter_index] = gamma 

        # calculate bunch energy
        energy[0][iter_index] = P_to_E(np.mean(pz))
        energy[1][iter_index] = P_to_E(np.std(pz))

        L_travel[iter_index] = ts.t[iter_index]*3e8

    # create pd dataframe to store all the data
    df_emit = pd.DataFrame(np.concatenate((np.array([L_travel]),emit)).T, columns = ['L','ps_n','geometric','ts_n'])
    df_E = pd.DataFrame(np.concatenate((np.array([L_travel]),energy)).T, columns = ['L','E','E_std'])
    df_twiss = pd.DataFrame(np.concatenate((np.array([L_travel]),twiss)).T, columns = ['L','alpha','beta','gamma'])
    
    return df_emit,df_E,df_twiss

def percent_charge_loc(aim_charge,tot_charge,pdf_z,xrange):
    ### return the longitudinal position of the targeted percentage of the total charge

    tot_sum = 0
    sum_pdf = np.trapz(pdf_z,x = xrange)
    dx = xrange[1] - xrange[0]
    for idx in range(len(xrange)):
        tot_sum += (pdf_z[idx] + pdf_z[idx+1])*dx/2 / sum_pdf * tot_charge
        if tot_sum > aim_charge:
            return idx

def plot_particle_info_slices(beam,ts,iteration,num,pdf_z,zedges,aimed_charge,tot_charge):
    ### retrieve particle information using particle tracker at a given iteration
    ### for plotting purposes
    ### Can operate in two modes: 
    ### (1) for a Gaussian beam, specify the number of slices according to std if num is entered
    ### (2) for a tailored profile, specify the percent charge of the trailing slice
    
    # initial condition
    x,y,z,px,py,pz,w = ts.get_particle(species = beam,iteration = 0, var_list =['x','y','z','ux','uy','uz','w'])
    sig_z = np.sqrt(second_central_moment(z,z))
    mean_z = np.mean(z)
    
    if num:
        # storage
        beam_info = np.empty((num,7))
    
        # special case when taking three slices
        if num == 3:
            core_slice = [mean_z - sig_z, mean_z + sig_z]
            head_slice = [mean_z + sig_z, np.inf]
            tail_slice = [-np.inf,mean_z-sig_z]
            pt_core = ParticleTracker(ts, iteration=0, select={'z':core_slice}, species=beam)
            pt_head = ParticleTracker(ts, iteration=0, select={'z':head_slice}, species=beam)
            pt_tail = ParticleTracker(ts, iteration=0, select={'z':tail_slice}, species=beam)
    
            head_info = ts.get_particle(species = beam,iteration = iteration, var_list =['x','y','z','ux','uy','uz','w'],select = pt_head)
            core_info = ts.get_particle(species = beam,iteration = iteration, var_list =['x','y','z','ux','uy','uz','w'],select = pt_core)
            tail_info = ts.get_particle(species = beam,iteration = iteration, var_list =['x','y','z','ux','uy','uz','w'],select = pt_tail)
    
            beam_info = [head_info,core_info,tail_info]
        
        # further divide tail into two sections
        elif num == 4:
            core_slice = [mean_z - sig_z, mean_z + sig_z]
            head_slice = [mean_z + sig_z, np.inf]
            tail_slice = [-np.inf,mean_z-sig_z]
            tail_slice2 = [-np.inf,mean_z-2*sig_z]
            pt_core = ParticleTracker(ts, iteration=0, select={'z':core_slice}, species=beam)
            pt_head = ParticleTracker(ts, iteration=0, select={'z':head_slice}, species=beam)
            pt_tail = ParticleTracker(ts, iteration=0, select={'z':tail_slice}, species=beam)
            pt_tail2 = ParticleTracker(ts, iteration=0, select={'z':tail_slice2}, species=beam)
            
            head_info = ts.get_particle(species = beam,iteration = iteration, var_list =['x','y','z','ux','uy','uz','w'],select = pt_head)
            core_info = ts.get_particle(species = beam,iteration = iteration, var_list =['x','y','z','ux','uy','uz','w'],select = pt_core)
            tail = ts.get_particle(species = beam,iteration = iteration, var_list =['x','y','z','ux','uy','uz','w'],select = pt_tail)
            tail2 = ts.get_particle(species = beam,iteration = iteration, var_list =['x','y','z','ux','uy','uz','w'],select = pt_tail2)
    
        
            beam_info = [head_info,core_info,tail,tail2]
        
        return beam_info
    else:
        idx = percent_charge_loc(aimed_charge,tot_charge,pdf_z,zedges)
        pt_tail = ParticleTracker(ts, iteration=0, select={'z':[-np.inf,zedges[idx]]}, species=beam)
        tail_info = ts.get_particle(species = beam,iteration = iteration, var_list =['x','y','z','ux','uy','uz','w'],select = pt_tail)
        pt_head = ParticleTracker(ts, iteration=0, select={'z':[zedges[idx],np.inf]}, species=beam)
        head_info = ts.get_particle(species = beam,iteration = iteration, var_list =['x','y','z','ux','uy','uz','w'],select = pt_head)
        return [head_info,tail_info]
        

def pt_info(beam,ts,num,pdf_z,zedges,aimed_charge,tot_charge):
    ### initialization of particle tracker
    ### two modes of operation:
    ### (1) for Gaussian bunch
    ### (2) for tailored beam profile
    
    x,y,z,px,py,pz,w = ts.get_particle(species = beam,iteration = 0, var_list =['x','y','z','ux','uy','uz','w'])
    sig_z = np.sqrt(second_central_moment(z,z))
    mean_z = np.mean(z)
    
    if num:
        # storage
        #beam_info = np.empty((num,7))
    
        # special case when taking three slices
        if num == 3:
            core_slice = [mean_z - sig_z, mean_z + sig_z]
            head_slice = [mean_z + sig_z, np.inf]
            tail_slice = [-np.inf,mean_z-sig_z]
            
            pt_core = ParticleTracker(ts, iteration=0, select={'z':core_slice}, species=beam)
            pt_head = ParticleTracker(ts, iteration=0, select={'z':head_slice}, species=beam)
            pt_tail = ParticleTracker(ts, iteration=0, select={'z':tail_slice}, species=beam)
            return [pt_head,pt_core,pt_tail]    
        # chop tail further into two sections
        elif num == 4:
            core_slice = [mean_z - sig_z, mean_z + sig_z]
            head_slice = [mean_z + sig_z, np.inf]
            tail_slice = [-np.inf,mean_z-sig_z]
            tail_slice2 = [-np.inf,mean_z-2*sig_z]
    
            pt_core = ParticleTracker(ts, iteration=0, select={'z':core_slice}, species=beam)
            pt_head = ParticleTracker(ts, iteration=0, select={'z':head_slice}, species=beam)
            pt_tail = ParticleTracker(ts, iteration=0, select={'z':tail_slice}, species=beam)
            pt_tail2 = ParticleTracker(ts, iteration=0, select={'z':tail_slice2}, species=beam)
            
            return [pt_head,pt_core,pt_tail,pt_tail2]   
    else:
        idx = percent_charge_loc(aimed_charge,tot_charge,pdf_z,zedges)
        pt_tail = ParticleTracker(ts, iteration=0, select={'z':[-np.inf,zedges[idx]]}, species=beam)
        pt_head = ParticleTracker(ts, iteration=0, select={'z':[zedges[idx],np.inf]}, species=beam)
        return [pt_tail,pt_head]
    


## vacuum propagation of the twiss parameters
def beta_prop(x,a,b,g,L):
    # Betafunction propagation after the plasma lens
    # x is the total length; L is the simulation length
    # returned Betafunction is in cm
    return b*100 - 2*(x-L)*a + (x-L)**2 * g/100

############ Functions for plotting

# trace space ellipse function
def ts_ellipse(x, x_prime, emit, a, b, g):
    return g*x**2 + 2*a*x*x_prime + b*x_prime**2 - emit

def plot_params_ts_ellipse(x_range, y_range, emit, a, b, g):
    # Create a grid of (x, y) values
    X, Y = np.meshgrid(x_range, y_range)

    # Compute functions over the grid
    Z = ts_ellipse(X,Y, emit, a, b, g)

    # return the plotting parameters
    return X, Y, Z

def ts_plot_params(x,px,pz,x_range,x_p_range):
    
    # trace space parameters
    x_p = px/pz * 10**3
    x *= 10**6
    alpha, beta, gamma = twiss_params(x,x_p)
    emit_ts = ts_emittance_rms(x,x_p)
    
    # return parameters for the 2D histogram
    H, xedges, yedges = np.histogram2d(x, x_p, bins=[150,150],range=[x_range, x_p_range])

    # return parameters for the ts ellipse
    xrange = np.linspace(xedges[0], xedges[-1], 10**3)
    yrange = np.linspace(yedges[0], yedges[-1], 10**3)
    X, Y, Z = plot_params_ts_ellipse(xrange, yrange, emit_ts, alpha, beta, gamma)
    
    return [H,xedges,yedges],[X,Y,Z]

##### plot current profiles and tail charges

def find_index(ts,iteration):
    for i in range(len(ts.iterations)):
        if ts.iterations[i] == iteration:
            return i
        
# normalize pdf to kA
def normalize_pdf_to_kA(pdf,xrange,tot_charge):
    clight = 3e8
    pdf = pdf*tot_charge/np.trapz(pdf,x = xrange)
    z_edge = (xrange[:-1] + xrange[1:])/2
    hist_I = (pdf[:-1]+pdf[1:]) * clight/2000
    return z_edge,hist_I
        
def plot_beam_density(x,z,charge,n0,bins,y_size):
    Z, xedges, yedges = np.histogram2d(z*1e6, x*1e6, bins=bins)
    #Zd = ndimage.gaussian_filter(Zd, sigma=1, order=0)
    
    tot_num_macros = len(x)
    x_grid_len = np.abs(xedges[0] - xedges[1]) # z-direction
    y_grid_len = np.abs(yedges[0] - yedges[1]) # x-direction
    
    Z = Z/tot_num_macros * charge * 6.25e18 / (x_grid_len*y_grid_len*y_size*1e12) # num/cm^3
    Z /= n0
    return Z,xedges,yedges

####### functions for tailoring beam profiles
def Gaussian(x,A,sig,mu):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

def linear(x,a,b):
    return a*x + b

def parabola(x,a,b,c):
    return a*x**2 + b*x + c

# two horn profile middle region
def mid_fcns(x,A1,mu1,sig1,A2,mu2,sig2,a,c):
    x_mid = (mu1 + mu2)/2
    b = -x_mid*2*a
    return Gaussian(x,A1,sig1,mu1) + Gaussian(x,A2,sig2,mu2) + parabola(x,a,b,c)

# entire two horn profile
def entire_region(x,a_l,b_l,A1,mu1,sig1,A2,mu2,sig2,a,c,a_r,b_r):
    region1 = x[x < mu1]
    region2 = x[(x>=mu1)&(x<=mu2)]
    region3 = x[x > mu2]
    arr1 = linear(region1,a_l,b_l)
    arr2 = mid_fcns(region2,A1,mu1,sig1,A2,mu2,sig2,a,c)
    arr3 = linear(region3,a_r,b_r)
    
    return np.concatenate((arr1,arr2,arr3), axis=None)

    











