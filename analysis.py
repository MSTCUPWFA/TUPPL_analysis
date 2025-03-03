#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:57:35 2025

@author: mengshutang
"""
import numpy as np


# Basic functions -------------------------------------------------------

def P_to_E(P):
    """
    Convert normalized momentum to energy

    Parameters
    ----------
    P : np.array
        Normalized momentum.

    Returns
    -------
    E : np.array
        Energy in GeV.

    """
    me = 9.1093837 * 10**(-31)
    clight = 3*10**8
    J_to_eV = 6.24152272506e+18
    return P*me*clight**2*J_to_eV/10**9

def x_prime(px,pz):
    return px/pz

def second_central_moment(x,y):
    """
    Calculate second central moment; in the case that x = y, it is simply the variance of x.
    """
    N = len(x)
    return np.sum(x*y)/N - np.sum(x)*np.sum(y)/N**2

def ts_emittance_rms(x,x_prime):
    """
    Calculate geometric emittance

    """
    bracket_x_sqrd = second_central_moment(x,x)
    bracket_x_prime_sqrd = second_central_moment(x_prime,x_prime)
    bracket_x_x_prime = second_central_moment(x,x_prime)

    return np.sqrt(bracket_x_sqrd*bracket_x_prime_sqrd-bracket_x_x_prime**2)

def ts_emittance_n_rms(x,pz,x_prime):
    """
    Calculate normalized emittance

    """
    bracket_x_sqrd = second_central_moment(x,x)
    bracket_x_prime_sqrd = second_central_moment(x_prime,x_prime)
    bracket_x_x_prime = second_central_moment(x,x_prime)

    return np.mean(pz)*np.sqrt(bracket_x_sqrd*bracket_x_prime_sqrd-bracket_x_x_prime**2)

def twiss_params(x,x_prime):
    """
    Calculate twiss parameters alpha, beta, and gamma

    """
    geometric_emit = ts_emittance_rms(x,x_prime)
    alpha = -second_central_moment(x,x_prime)/geometric_emit
    beta = second_central_moment(x,x)/geometric_emit
    gamma = second_central_moment(x_prime,x_prime)/geometric_emit

    return alpha,beta,gamma

def beta_prop(x,a,b,g):
    """ Evolution of Betafunction of beam in vacuum

    Parameters
    ----------
    x : int/float
        Length of propagation from the start (i.e. x = L is the start of the vacuum propagation).
    a : Twiss parameter alpha; SI unit.
    b : Twiss parameter beta; SI unit.
    g : Twiss parameter gamma; SI unit

    Returns
    -------
    Betafunction (in unit of cm) at a distance x.

    """
    return b*100 - 2*x*a + x**2 * g/100

def ts_ellipse(x, x_prime, emit, a, b, g):
    """
    Parameters
    ----------
    x : np.array
    x_prime : np.array
    emit : float
        geometric emittance.
    a : float
        Twiss paremter alpha.
    b : float
        Twiss paremter beta.
    g : float
        Twiss paremter gamma.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return g*x**2 + 2*a*x*x_prime + b*x_prime**2 - emit

def plot_ts_ellipse(x_range, y_range, emit, a, b, g):
    """
    Trace space ellipse corresponding to the geometric emittance

    Parameters
    ----------
    x_range : int
        x (horizontal) range for the grid.
    y_range : int
        y (vertical) range for the grid.
    emit : float
        geometric emittance.
    a : float
        Twiss paremter alpha.
    b : float
        Twiss paremter beta.
    g : float
        Twiss paremter gamma.

    Returns
    -------
    X : np.array
        x grid range.
    Y : np.array
        y grid range.
    Z : np.array
        ellipse on the (x,y) grid.

    """
    # Create a grid of (x, y) values
    X, Y = np.meshgrid(x_range, y_range)

    # Compute functions over the grid
    Z = ts_ellipse(X,Y, emit, a, b, g)

    # return the plotting parameters
    return X, Y, Z