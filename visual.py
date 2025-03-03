#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:17:32 2025

@author: mengshutang
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import scipy.ndimage as ndimage
import analysis as an
import beam_info as bi


class visual:
    """ 
    Visualization of PWFA window.
    
    Initialization:
        ts : time series info.
        iteration : iteration number in PIC simulation.
        n0 : benchmark plasma density; unit: cm^-3.
    
    Attributes:
        ts : time series info.
        n0 : benchmark plasma density; unit: cm^-3.
        rho : normalized plasma density w.r.t n0.
        Ez : longitudinal electric field; unit: eV.
        grid_info : grid metadata
        
    Functions:
        plot_field: plot rho field and Ez lineout.
        plot_beam: plot beam density contour.
        
    """
    
    def __init__(self,ts,iteration,n0):
        self.ts = ts
        self.n0 = n0
        
        rho, rho_grid = ts.get_field(field = 'rho_electron',iteration = iteration)
        Ez, E_grid = ts.get_field(field = 'Ez',iteration = iteration)
        
        self.rho = rho * 6.25e12/n0
        self.Ez = Ez
        self.grid_info = rho_grid
        
    def plot_field(self,axes,vmin,plot_number_density):
        """
        plot rho field and Ez lineout at x = 0

        Parameters
        ----------
        axes : plt.axes
            axes for the field plot.
        vmin : int/float
            maximum absolute plasma density used in colorbar 
            (note it should be a negative number, i.e. -10 corresponds to -10*n0 plasma density).
        plot_number_density : Boolean
            whether or not the beam number density will be plotted.

        """
        #------------------------------------rho field------------------------------------
        
        field_cmap = colors.LinearSegmentedColormap.from_list("", ["Black","midnightblue","darkblue","royalblue","cornflowerblue","white"])

        grid_range_z = self.grid_info.z * 1e6 # in um
        grid_range_x = self.grid_info.x * 1e6 # in um
        
        rho_plot = axes.pcolormesh(grid_range_z, grid_range_x, np.transpose(self.rho),cmap = field_cmap, vmax = 0, vmin = vmin)
        
        if plot_number_density:
            cb_field = plt.colorbar(rho_plot, location='top', aspect=50, pad=0.01)
        else:
            cb_field = plt.colorbar(rho_plot, location='top', aspect=50, pad=0.07)
        cb_field.set_label(f'Plasma density ({self.n0}'+r" cm$^{-3}$)",labelpad=-42, y=0.45,fontsize = 12)
        
        #----------------------------------Ez center lineout------------------------------
        ax_E_plot = axes.twinx()
        Ez_center = self.Ez.T[int(np.rint(len(self.Ez)/2))]
        Ez_center /= 1e9
        
        ax_E_plot.plot(grid_range_z,Ez_center, c='r')
        
        ax_E_plot.yaxis.label.set_color('red')
        ax_E_plot.set(ylabel=r"$E_z$ $(GeV/m)$")
        ax_E_plot.tick_params(axis='y', colors='r')
        
        # put zero in the middle of the axis
        Ez_abs_max = abs(max(ax_E_plot.get_ylim(), key=abs))
        ax_E_plot.set_ylim(ymin=-Ez_abs_max, ymax=Ez_abs_max)
        
        
    def plot_beam(self,beam_info,plot_number_density,bin_y,axes,vmax):
        """
        Plot particles within a beam_info instance; Three features are included:
            1. bin_y: can choose to plot particles near y = 0 or the projection of all particles onto the 2D plane.
            2. 
        

        Parameters
        ----------
        beam_info : TYPE
            DESCRIPTION.
        plot_number_density : TYPE
            DESCRIPTION.
        bin_y : TYPE
            DESCRIPTION.
        axes : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
       
        x,y,z = self.ts.get_particle(species = beam_info.beam,iteration = beam_info.iteration, var_list = ['x','y','z'])
        
        cmap = colors.LinearSegmentedColormap.from_list("", ["white","darkorange","brown","darkred","maroon","black"])
        # cmap = anothercmap
        my_cmap = cmap(np.arange(cmap.N))
        # Set alpha
        my_cmap[:,-1] = np.linspace(0,1,cmap.N)
        # create a new colormap
        beam_cmap = ListedColormap(my_cmap)
        
        if not bin_y:
            # select particles within the grid near y = 0
            select_particles = (y<.5*self.grid_info.dx) & (y>-.5*self.grid_info.dx)
            z_in_cell = z[select_particles]
            x_in_cell = x[select_particles]
            axes.scatter(z_in_cell * 1e6,x_in_cell * 1e6,s = 0.02,c = 'k',alpha = 0.5)
        elif plot_number_density:
            Z, zedges, xedges= np.histogram2d(z, x, bins=[100,100])
            
            for _1 in range(len(zedges)-1):
                for _2 in range(len(xedges)-1):
                    if Z[_1][_2] <= 30:
                        Z[_1][_2] = 0
                        continue
                    z_lo = zedges[_1]
                    z_hi = zedges[_1+1]
                    x_lo = xedges[_2]
                    x_hi = xedges[_2+1]
                    dz = np.abs(z_lo - z_hi)
                    dx = np.abs(x_lo - x_hi)
                    dy = 3*np.std(y[(x < x_hi)&(x >= x_lo)&(z >= z_lo)&(z < z_hi)])
                    grid_volume = dx*dy*dz # in m^3
                    if grid_volume == 0:
                        Z[_1][_2] = 0
                    else:
                        Z[_1][_2] = Z[_1][_2]/len(z) * beam_info.tot_charge * 6.25e18/(grid_volume*1e6) /self.n0
                    
            Z = ndimage.gaussian_filter(Z, sigma=1, order=0)
            driver = axes.pcolormesh(zedges*1e6, xedges*1e6, Z.T, cmap= beam_cmap, vmin = 0, vmax = vmax)
            cb_driver = plt.colorbar(driver, location='top', aspect=50, pad=0.07)
            cb_driver.set_label(f'Beam density ({self.n0}'+r" cm$^{-3}$)", labelpad=-42, y=0.45,fontsize = 12)
        else:
            Z, zedges, xedges= np.histogram2d(z, x, bins=[100,100])
            Z = ndimage.gaussian_filter(Z, sigma=1, order=0)
            driver = axes.pcolormesh(zedges*1e6, xedges*1e6, Z.T, cmap= beam_cmap, vmin = 0)    
    
class plot_trajectory:
    """ 
    Plot beam parameters' evolution.
    
    Initialization:
        beam : get_beam_info object with an arbitrary iteration number.
    
    Attributes:
        ts : time series info.
        beam : get_beam_info object being initialized.
        beam_series : list of get_beam_info objects corresponding to each iteration in time series data.
        
    Functions:
        bf_in_sims : plot betafunction evolution within the simulation.
        bf_in_vac : plot betafunction propagation in vacuum after the simulation.
        vac_profile : plot vacuum propagation of the incoming beam with its initial parameters before entering the plasma.
        full_beam_size_evln : beam size (sigma) evolution inside the plasma and propagation in vacuum after exiting the plasma.
        E_gain : Energy evolution inside plasma.
        emittance : emittance evolution inside plasma.
        E_x_plot_info : E-x contour at certain distance from the starting of the simulation.
        trace_space : trace space contour at certain distance from the starting of the simulation.
    
    """
    
    def __init__(self,beam):
        self.ts = beam.ts_info
        self.beam = beam
        self.beam_series = [bi.get_beam_info(self.ts,self.beam.beam,iteration,self.beam.tail_charge,self.beam.tot_charge,self.beam.dt,self.beam.num_bins) for iteration in self.ts.iterations]
        
    def bf_in_sims(self,section,axes,color,linestyle,linewidth,label):
        """
        Plot betafunction evolution (unit: both in cm) in the simulations (i.e. within the time series data).

        Parameters
        ----------
        section : string
            'tail', 'head', or 'entire', corresponding to different part of the beam.
        axes : matplotlib.axes
            axes for plotting.
        color : string
            color for plotting.
        linestyle : string
            linestyle for plotting.
        linewidth : float/int
            linewidth for plotting.
        label : string
            plot label.

        Returns
        -------
        x_range : array
            x range; unit: cm.
        beta_arr : array
            betafunction; unit: cm.

        """
        x_range = self.ts.iterations * self.beam.dt * 3e8
        
        beta_arr = []
        for i in range(len(self.ts.iterations)):
            
            new_beam = self.beam_series[i]
            
            if section == 'tail':
                b_x = new_beam.b_x_tail
            elif section == 'head':
                b_x = new_beam.b_x_head
            else:
                b_x = new_beam.b_x

            beta_arr.append(b_x*1e2)
        
        if axes:
            axes.plot(x_range*1e2,beta_arr,c = color,linestyle = linestyle,linewidth = linewidth,label = label)
        else:
            plt.plot(x_range*1e2,beta_arr,c = color,linestyle = linestyle,linewidth = linewidth,label = label)
            
        return x_range*1e2,beta_arr
            
    def bf_in_vac(self,L,section,axes,color,linestyle,linewidth,label):
        """
        Plot betafunction (unit: both in cm) propagation in vacuum after the simulation.

        Parameters
        ----------
        x_range : int; unit of cm.
            propagation length in vacuum.
        section : string
            'tail', 'head', or 'entire', corresponding to different part of the beam.
        axes : matplotlib.axes
            axes for plotting.
        color : string
            color for plotting.
        linestyle : string
            linestyle for plotting.
        linewidth : float/int
            linewidth for plotting.
        label : string
            plot label.

        Returns
        -------
        x_range : array
            x range; unit: cm.
        beta_arr: array
            betafunction; unit: cm.

        """
        last_beam = self.beam_series[-1]
        x_range = np.linspace(0,L,num = 10**3)
        
        if section == 'tail':
            a,b,g = last_beam.a_x_tail,last_beam.b_x_tail,last_beam.g_x_tail
        
        elif section == 'head':
            a,b,g = last_beam.a_x_head,last_beam.b_x_head,last_beam.g_x_head
            
        else:
            a,b,g = last_beam.a_x,last_beam.b_x,last_beam.g_x
        
        beta_arr = an.beta_prop(x_range, a, b, g)
        
        if axes:
            axes.plot(x_range+last_beam.L*1e2, beta_arr, c = color, linestyle = linestyle, linewidth = linewidth, label = label)
        else:
            plt.plot(x_range+last_beam.L*1e2, beta_arr, c = color, linestyle = linestyle, linewidth = linewidth, label = label)
            
        return x_range+last_beam.L*1e2,beta_arr
    
    def vac_profile(self,L,section,axes,color,linestyle,linewidth,label):
        """
        Vacuum propagation of betafunction without plasma interaction; use the initialized beam profile at iteration 0.

        Parameters
        ----------
        L : int; unit of cm.
            propagation length in vacuum.
        section : string
            'tail', 'head', or 'entire', corresponding to different part of the beam.
        axes : matplotlib.axes
            axes for plotting.
        color : string
            color for plotting.
        linestyle : string
            linestyle for plotting.
        linewidth : float/int
            linewidth for plotting.
        label : string
            plot label.
        """
        
        first_beam = self.beam_series[0]
        x_range = np.linspace(0,L,num = 10**3)
        
        if section == 'tail':
            a,b,g = first_beam.a_x_tail,first_beam.b_x_tail,first_beam.g_x_tail
        
        elif section == 'head':
            a,b,g = first_beam.a_x_head,first_beam.b_x_head,first_beam.g_x_head
            
        else:
            a,b,g = first_beam.a_x,first_beam.b_x,first_beam.g_x
        
        beta_arr = an.beta_prop(x_range, a, b, g)
        
        if axes:
            axes.plot(x_range, beta_arr, c = color, linestyle = linestyle, linewidth = linewidth, label = label)
        else:
            plt.plot(x_range, beta_arr, c = color, linestyle = linestyle, linewidth = linewidth, label = label) 
            
    def full_beam_size_evln(self,L,section,axes,color,linestyle,linewidth,label):
        """
        Plot full range beam spot size evolution; sigma unit in um.

        Parameters
        ----------
        L : int; unit of cm.
            propagation length in vacuum.
        section : string
            choose from "tail", "head", or "entire".
        axes : plt.axes
            specific axes for plotting; enter None if no axes is used.
        color, linestyle, ... :
            common plotting settings in matplotlib.pyplot
            
        Returns
        -------
        x : array
            z range; unit: cm.
        sig_arr : array
            beam size (sigma) array; unit: um.

        """
        
        sig_arr_sims = []
        x_range = np.linspace(0,L,num = 10**3)
        
        for i in range(len(self.ts.iterations)):
            
            new_beam = self.beam_series[i]
            
            if section == 'tail':
                b_x = new_beam.b_x_tail
                e_x = new_beam.geo_emit_x_tail
                
            elif section == 'head':
                b_x = new_beam.b_x_head
                e_x = new_beam.geo_emit_x_head
                
            else:
                b_x = new_beam.b_x
                e_x = new_beam.geo_emit_x

            sig_arr_sims.append(np.sqrt(b_x*e_x)*1e6)
        
        if section == 'tail':
            a,b,g = new_beam.a_x_tail,new_beam.b_x_tail,new_beam.g_x_tail
            sig_arr_vac = np.sqrt(an.beta_prop(x_range, a, b, g)*1e-2*new_beam.geo_emit_x_tail)*1e6
        
        elif section == 'head':
            a,b,g = new_beam.a_x_head,new_beam.b_x_head,new_beam.g_x_head
            sig_arr_vac = np.sqrt(an.beta_prop(x_range, a, b, g)*1e-2*new_beam.geo_emit_x_head)*1e6
            
        else:
            a,b,g = new_beam.a_x,new_beam.b_x,new_beam.g_x
            sig_arr_vac = np.sqrt(an.beta_prop(x_range, a, b, g)*1e-2*new_beam.geo_emit_x)*1e6
        
        sig_arr = np.concatenate((sig_arr_sims,sig_arr_vac))
        x = np.concatenate((self.ts.iterations * self.beam.dt * 3e10, x_range + new_beam.L*1e2))
        
        if axes:
            axes.plot(x, sig_arr, c = color, linestyle = linestyle, linewidth = linewidth, label = label)
        else:
            plt.plot(x, sig_arr, c = color, linestyle = linestyle, linewidth = linewidth, label = label) 
            
        return x, sig_arr
    
    def E_gain(self,section,axes,color,linestyle,linewidth,label):
        """
        Plot energy evolution within plasma

        Parameters
        ----------
        section : string
            'tail', 'head', or 'entire', corresponding to different part of the beam.
        axes : matplotlib.axes
            axes for plotting.
        color : string
            color for plotting.
        linestyle : string
            linestyle for plotting.
        linewidth : float/int
            linewidth for plotting.
        label : string
            plot label.

        Returns
        -------
        None.

        """
        
        x_range = self.ts.iterations * self.beam.dt * 3e8
        
        if section == 'tail':
            E_arr = np.array([np.mean(beam.E_tail) for beam in self.beam_series])
        elif section == 'head':
            E_arr = np.array([np.mean(beam.E_head) for beam in self.beam_series])
        else:
            E_arr = np.array([np.mean(beam.E) for beam in self.beam_series])
            
        if axes:
            axes.plot(x_range*1e2, E_arr, c = color, linestyle = linestyle, linewidth = linewidth, label = label)
        else:
            plt.plot(x_range*1e2, E_arr, c = color, linestyle = linestyle, linewidth = linewidth, label = label)
        
    def emittance(self,section,axes,color,linestyle,linewidth,label):
        """
        Plot emittance evolution within the plasma
        
        Parameters
        ----------
        section : string
            'tail', 'head', or 'entire', corresponding to different part of the beam.
        axes : matplotlib.axes
            axes for plotting.
        color : string
            color for plotting.
        linestyle : string
            linestyle for plotting.
        linewidth : float/int
            linewidth for plotting.
        label : string
            plot label.

        Returns
        -------
        None.

        """
        
        x_range = self.ts.iterations * self.beam.dt * 3e8
        
        if section == 'tail':
            emit_arr = np.array([beam.norm_emit_x_tail for beam in self.beam_series])/self.beam_series[0].norm_emit_x_tail
        elif section == 'head':
            emit_arr = np.array([beam.norm_emit_x_head for beam in self.beam_series])/self.beam_series[0].norm_emit_x_head
        else:
            emit_arr = np.array([beam.norm_emit_x for beam in self.beam_series])/self.beam_series[0].norm_emit_x
        
        if axes:
            axes.plot(x_range*1e2, emit_arr, c = color, linestyle = linestyle, linewidth = linewidth, label = label)
        else:
            plt.plot(x_range*1e2, emit_arr, c = color, linestyle = linestyle, linewidth = linewidth, label = label)
            
    
    def E_x_plot_info(self,L,detect_threshold,axes):
        """
    
        Parameters
        ----------
        L : float
            distance from the start of the simulation; SI unit (m).
        detect_threshold : float/int
            detection threshold of the spectrometer.
        axes : matplotlib.axis
            axes for plotting.

        Returns
        -------
        H : array
            2D histogram with first dimension (row) being x; and second dimension (column) being E.
        xedges : array
            x array; unit: um.
        yedges : array
            E array; unit: GeV.

        """
        
        if L <= self.ts.iterations[-1] * self.beam.dt * 3e8: 
            abs_diff = np.abs(self.ts.iterations * self.beam.dt * 3e8 - L)
            min_index = np.argmin(abs_diff)
            x,pz = self.ts.get_particle(species = self.beam.beam,iteration = self.ts.iterations[min_index], var_list =['x','uz'])
            E = an.P_to_E(pz)
            H, xedges, yedges = np.histogram2d(x, E, bins=[200,200])

        else:
            x,px,pz = self.ts.get_particle(species = self.beam.beam,iteration = self.ts.iterations[-1], var_list =['x','ux','uz'])
            x_p = px/pz
            x_f = x + (L - self.ts.iterations[-1] * self.beam.dt * 3e8)*x_p
            E_f = an.P_to_E(pz)
            H, xedges, yedges = np.histogram2d(x_f, E_f, bins=[200,200])
        
        if detect_threshold:
            H[H < detect_threshold] = 0
            
        if axes:
            beam_f = axes.pcolormesh(xedges*1e6, yedges, H.T, cmap='magma')
            #cb_beam1 = plt.colorbar(beam_f,ax = axes, location='right', aspect=50, pad=0.05,label = 'Beam density')
        else:
            beam_f = plt.pcolormesh(xedges*1e6, yedges, H.T, cmap='magma')
            #cb_beam1 = plt.colorbar(beam_f,ax = axes, location='right', aspect=50, pad=0.05,label = 'Beam density')
        
        return H, xedges, yedges
    
    def trace_space(self,L):
        """

        Parameters
        ----------
        L : float
            distance from the start of the simulation; SI unit (m).

        Returns
        -------
        x : array
            x array; unit: meter.
        x_p : array
            x prime array; unit: rad.
        pz : array
            normalized momentum array.

        """
        if L <= self.ts.iterations[-1] * self.beam.dt * 3e8: 
            abs_diff = np.abs(self.ts.iterations * self.beam.dt * 3e8 - L)
            min_index = np.argmin(abs_diff)
            x,px,pz = self.ts.get_particle(species = self.beam.beam,iteration = self.ts.iterations[min_index], var_list =['x','ux','uz'])
            x_p = px/pz

        else:
            x,px,pz = self.ts.get_particle(species = self.beam.beam,iteration = self.ts.iterations[-1], var_list =['x','ux','uz'])
            x_p = px/pz
            x += (L - self.ts.iterations[-1] * self.beam.dt * 3e8)*x_p
            
        return x,x_p,pz
            
    
        
    
