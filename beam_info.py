#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:36:45 2025

@author: Shutang Meng
"""
import numpy as np
from openpmd_viewer import ParticleTracker
import matplotlib.pyplot as plt
import analysis as an

class get_beam_info:
    """ beam info for different parts of the beam
    
    Initialization:
        ts: time series info
        beam: beam name
        iteration: iteration
        tail_charge: tail charge
        tot_charge: total charge
        dt: time step length
        num_bins: number of bins along the longitudinal direction
    
    Attributes:
        ts_info: time series data from PIC simulation
        beam: beam species
        iteration: specific iteration number 
        tail_charge: tail charge; unit: Coulomb
        tot_charge: total charge of the beam; unit: Coulumb
        pt_head: particle tracker of the head beam
        pt_tail: particle tracker of the tail beam
        
        a_x/y: Twiss parameter (alpha) of the entire beam in the x/y direction; all in SI unit
        b_x/y: Twiss parameter (beta) of the entire beam in the x/y direction
        g_x/y: Twiss parameter (gamma) of the entire beam in the x/y direction
        (same attributes for head and tail charges)
        
        E: energy of the entire beam along the z direction; type -> array; unit: GeV
        E_head: energy of the head beam along the z direction: unit: type -> array; unit: GeV
        E_tail: energy of the tail beam along the z direction: unit: type -> array; unit: GeV
        
        geo_emit_x/y: geometric emittance for x/y direction
        norm_emit_x/y: normalized emittance for x/y direction
        (again same attributes for head and tail charges)
        
        L: length traveled at the specific iteration: unit: m
        
    Functions:
        plot_I_profile: plot current profile along with energy chirp
        plot_trace_space: plot the trace space profile in the x,x' plane along with the geometric emittance ellipse
        print_beam_info: print beam information
    """
    
    def __init__(self,ts,beam,iteration,tail_charge,tot_charge,dt,num_bins):
        
        self.ts_info = ts
        self.iteration = iteration
        self.beam = beam
        self.tail_charge = tail_charge
        self.tot_charge = tot_charge
        self.num_bins = num_bins
        self.dt = dt
        
        
        # particle tracking for head and tail charges
        pt_head,pt_tail = self.particle_tracker()
        self.pt_head = pt_head
        self.pt_tail = pt_tail
        
        
        # twiss paremeters
        (self.a_x, self.b_x, self.g_x),(self.a_y, self.b_y, self.g_y) = self.get_Twiss(None)
        (self.a_x_head, self.b_x_head, self.g_x_head),(self.a_y_head, self.b_y_head, self.g_y_head) = self.get_Twiss(self.pt_head)
        (self.a_x_tail, self.b_x_tail, self.g_x_tail),(self.a_y_tail, self.b_y_tail, self.g_y_tail) = self.get_Twiss(self.pt_tail)
        
        # energy
        self.E = self.get_energy(None)
        self.E_head = self.get_energy(self.pt_head)
        self.E_tail = self.get_energy(self.pt_tail)
        
        # geometric and normalized emittance
        self.geo_emit_x,self.geo_emit_y,self.norm_emit_x,self.norm_emit_y = self.get_emit(None)
        self.geo_emit_x_head,self.geo_emit_y_head,self.norm_emit_x_head,self.norm_emit_y_head = self.get_emit(self.pt_head)
        self.geo_emit_x_tail,self.geo_emit_y_tail,self.norm_emit_x_tail,self.norm_emit_y_tail = self.get_emit(self.pt_tail)
        
        # length traveled
        self.L = 3e8*iteration*dt
        
    def print_beam_info(self,section):
        """
        Parameters
        ----------
        section : string
            choose one of the 'tail', 'entire', or 'head'; which decides what part of the beam info to be printed.

        """
        if section == 'tail':
            pt = self.pt_tail
            e_x, e_y = self.norm_emit_x_tail, self.norm_emit_y_tail
            a_x,a_y = self.a_x_tail, self.a_y_tail
            b_x,b_y = self.b_x_tail, self.b_y_tail
            g_x,g_y = self.g_x_tail, self.g_y_tail
        
        elif section == 'head':
            pt = self.pt_head
            e_x, e_y = self.norm_emit_x_head, self.norm_emit_y_head
            a_x,a_y = self.a_x_head, self.a_y_head
            b_x,b_y = self.b_x_head, self.b_y_head
            g_x,g_y = self.g_x_head, self.g_y_head
        
        else:
            pt = None
            e_x, e_y = self.norm_emit_x, self.norm_emit_y
            a_x,a_y = self.a_x, self.a_y
            b_x,b_y = self.b_x, self.b_y
            g_x,g_y = self.g_x, self.g_y
            
        x,y = self.ts_info.get_particle(species = self.beam,iteration = self.iteration, var_list =['x','y'],select = pt)
        sig_x = np.std(x)
        sig_y = np.std(y)
            
        print("Beam species: " + self.beam + ' ' + section)
        print(f"Distance traveled: {self.L*1e2} cm")
        print("Beam spot size: " + r"sig_x = " + f"{np.round(sig_x*1e6,3)} um; " + r"sig_y = " + f"{np.round(sig_y*1e6,3)} um.")
        print("Normalized emittance: " + r"eps_x = " + f"{np.round(e_x*1e6,3)} um; " + r"eps_y = " + f"{np.round(e_y*1e6,3)} um.")
        print(r"Twiss parameters: a_xy = " + f"{np.round(a_x,3)},{np.round(a_y,3)}; " + r"b_xy = " + f"{np.round(b_x,3)},{np.round(b_y,3)}; " + r"g_xy = " + f"{np.round(g_x,3)},{np.round(g_y,3)}.")

    
    def plot_I_profile(self):
        """
        Description: plot current profile and E chirp

        """
        [z] = self.ts_info.get_particle(species = self.beam,iteration = self.iteration, var_list =['z'])

        hist, bin_edges = np.histogram(z,bins = self.num_bins)
        z_edge = (bin_edges[:-1] + bin_edges[1:])/2

        hist_charge = hist/np.trapz(hist,z_edge)*self.tot_charge 
        z_I_edge = (z_edge[:-1] + z_edge[1:])/2
        hist_I = (hist_charge[:-1]+hist_charge[1:]) * 3e8/self.num_bins
        
        # plot settings
        fig,axes = plt.subplots(1,1)
        
        axes.set_xlabel(r"$\xi$ ($\mu$m)")
        axes.set_ylabel("I (kA)")
        axes.yaxis.label.set_color('b')
        axes.tick_params(axis='y', colors='b')
        
        # plot current profile
        axes.plot(z_I_edge*1e6,hist_I,c = 'b')
        idx,x_idx = self.tail_charge_loc()
        axes.fill_between(z_I_edge[:idx]*1e6, hist_I[:idx], color='g', alpha=1,label = f'{self.tail_charge*1e12} pC')
        
        # plot E chirp profile
        E_chirp_plot = axes.twinx()
        E_chirp_plot.scatter(z*1e6,self.E, s = 0.01,c='r',alpha = 0.05)
        E_chirp_plot.yaxis.label.set_color('red')
        E_chirp_plot.set(ylabel=r"$E$ $(GeV)$")
        E_chirp_plot.tick_params(axis='y', colors='r')
        
        axes.legend(loc = 'upper left')
        
    def plot_trace_space(self):
        """
        Description: plot trace space profile and the geometric emittance ellipse in the (x,xp) plane.

        """
        fig,ax1 = plt.subplots(1,1)
        
        # beam density histogram
        x,px,pz = self.ts_info.get_particle(species = self.beam,iteration = self.iteration, var_list =['x','ux','uz'])
        x_p = an.x_prime(px, pz)
        
        H, xedges, yedges = np.histogram2d(x, x_p, bins=[150,150])
        
        beam = ax1.pcolormesh(xedges, yedges, H.T, cmap='magma')
        cb_beam = plt.colorbar(beam, ax=ax1, location='bottom', aspect=50, pad=0.15,label = 'Beam density')
        
        # ellipse in trace space
        xrange = np.linspace(xedges[0], xedges[-1], 10**3)
        yrange = np.linspace(yedges[0], yedges[-1], 10**3)
        
        X, Y, Z = an.plot_ts_ellipse(xrange, yrange, self.geo_emit_x, self.a_x, self.b_x, self.g_x)
        contour = ax1.contour(X, Y, Z, levels=[0], colors='white')
        
        for level, line in zip(contour.levels, contour.collections):
            line.set_linestyle('dashdot') 
            line.set_linewidth(1) 
    
        ax1.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
        ax1.set_xlabel(r'x (m)')
        ax1.set_ylabel(r'$x^{\prime}$ (rad)')
        
                
    def get_energy(self,pt):
        """
        Parameters
        ----------
        pt : particle tracker
            Particle tracker for selection pz.

        Returns
        -------
        energy of the sliced beam: in GV

        """
        [pz] = self.ts_info.get_particle(species = self.beam,iteration = self.iteration, var_list =['uz'],select = pt)
        
        return an.P_to_E(pz)
        
    def get_Twiss(self,pt):
        """
        Parameters
        ----------
        pt: 
            Particle tracker for the desired slice of the beam.

        Returns
        -------
        ((ax,bx,gx),(ay,by,gy)):
            Twiss paramters in both transverse directions.

        """
        x,y,px,py,pz = self.ts_info.get_particle(species = self.beam,iteration = self.iteration, var_list =['x','y','ux','uy','uz'],select = pt)
        x_p, y_p = an.x_prime(px, pz),an.x_prime(py, pz)
        
        return an.twiss_params(x, x_p),an.twiss_params(y, y_p)
    
    def get_emit(self,pt):
        """
        Parameters
        ----------
        pt : particle tracker of the sliced beam

        Returns
        -------
        ((geo_emit_x,geo_emit_y),(norm_emit_x,norm_emit_y)): 
            geo_emit is geometric emittance
            norm_emit is normalized emittance

        """
        x,y,px,py,pz = self.ts_info.get_particle(species = self.beam,iteration = self.iteration, var_list =['x','y','ux','uy','uz'],select = pt)
        x_p, y_p = an.x_prime(px, pz),an.x_prime(py, pz)
        
        geo_emit_x,geo_emit_y = an.ts_emittance_rms(x, x_p),an.ts_emittance_rms(y, y_p)
        
        norm_emit_x,norm_emit_y = an.ts_emittance_n_rms(x, pz, x_p),an.ts_emittance_n_rms(y, pz, y_p)
        
        return geo_emit_x,geo_emit_y,norm_emit_x,norm_emit_y
            
    def tail_charge_loc(self):
        """
        Returns
        -------
        idx : the index location in the binned 'z' array that seperates the tail charge and head charge
        loc : z location in SI unit
        """
        [z] = self.ts_info.get_particle(species = self.beam,iteration = self.iteration, var_list =['z'])

        hist, bin_edges = np.histogram(z,bins = self.num_bins)
        
        z_edge = (bin_edges[:-1] + bin_edges[1:])/2        
        tot_sum = 0
        sum_pdf = np.trapz(hist,x = z_edge)
        dx = z_edge[1] - z_edge[0]
        for idx in range(len(z_edge)):
            tot_sum += (hist[idx] + hist[idx+1])*dx/2 / sum_pdf * self.tot_charge
            if tot_sum > self.tail_charge:
                return idx,z_edge[idx]
        
        if idx == len(z_edge):
            return idx,z_edge[idx]
        else:
            print("Error in finding index location")
            
    def particle_tracker(self):
        """
        Returns
        -------
        particle tracker for head and tail charges

        """
        idx,x_idx = self.tail_charge_loc()
        
        if type(idx) != str:
            pt_head = ParticleTracker(self.ts_info, iteration=0, select={'z':[x_idx,np.inf]}, species=self.beam)
            pt_tail = ParticleTracker(self.ts_info, iteration=0, select={'z':[-np.inf,x_idx]}, species=self.beam)
        else:
            print("Error: cannnot initiate particle tracker")
            
        return pt_head,pt_tail

        
        



















