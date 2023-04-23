#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 02 10:25:17 2021

@author: Juan Martinez Sykora

"""

# import builtin modules
import os

# import external public "common" modules
import numpy as np
# import matplotlib.pyplot as plt 


def deriv_dnw(xx, hh, **kwargs):
    """
    Returns the downwind 2nd order derivative of hh array respect to xx array. 

    Parameters 
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    -------
    `array`
        The downwind 2nd order derivative of hh respect to xx. Last 
        grid point is ill (or missing) calculated. 
    """
    return (np.roll(hh, -1) - hh)/(np.roll(xx, -1) - xx)

def deriv_upw(xx, hh, **kwargs):
    r"""
    returns the upwind 2nd order derivative of hh respect to xx. 

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    ------- 
    `array`
        The upwind 2nd order derivative of hh respect to xx. First 
        grid point is ill calculated. 
    """
    return (hh - np.roll(hh, 1)) / (xx - np.roll(xx, 1))

def deriv_cent(xx, hh, **kwargs):
    r"""
    returns the centered 2nd derivative of hh respect to xx. 

    Parameters
    ---------- 
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    -------
    `array`
        The centered 2nd order derivative of hh respect to xx. First 
        and last grid points are ill calculated. 
    """
    return (np.roll(hh, -1) - np.roll(hh, 1)) / (np.roll(xx, -1) - np.roll(xx, 1))


###################
### EXERCISE 1b ###
###################

def order_conv(hh, hh2, hh4, **kwargs):
    """
    Computes the order of convergence of a derivative function 

    Parameters 
    ----------
    hh : `array`
        Function that depends on xx. 
    hh2 : `array`
        Function that depends on xx but with twice number of grid points than hh. 
    hh4 : `array`
        Function that depends on xx but with twice number of grid points than hh2.
    Returns
    -------
    `array` 
        The order of convergence.  
    """
    # Slice arrays 
    hh2 = hh2[::2]
    hh4 = hh4[::4]
    return np.ma.log((hh4 - hh2)/(hh2 - hh))/np.log(2)
   
def deriv_4tho(xx, hh, **kwargs): 
    """
    Returns the 4th order derivative of hh respect to xx.

    Parameters 
    ---------- 
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx. 

    Returns
    -------
    `array`
        The centered 4th order derivative of hh respect to xx. 
        Last and first two grid points are ill calculated. 
    """
    return (-np.roll(hh, -2) + 8*np.roll(hh, -1) - 8*np.roll(hh, 1) + np.roll(hh, 2)) \
            /(12*(np.roll(xx, -1) - xx))


###################
### EXERCISE 2  ###
###################

def step_adv_burgers(xx, hh, a, cfl_cut = 0.98, 
                    ddx = lambda x,y: deriv_dnw(x, y), **kwargs): 
    r"""
    Right hand side of Burger's eq. where a can be a constant or a function that 
    depends on xx. 

    Requires 
    ---------- 
    cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default clf_cut=0.98. 
    ddx : `lambda function`
        Allows to select the type of spatial derivative. 
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array` 
        Time interval.
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """
    dt = cfl_adv_burger(a, xx)*cfl_cut
    rhs = -a*ddx(xx, hh)
    return dt, rhs

def cfl_adv_burger(a, x): 
    """
    Computes the dt_fact, i.e., Courant, Fredrich, and 
    Lewy condition for the advective term in the Burger's eq. 

    Parameters
    ----------
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis. 

    Returns
    ------- 
    `float`
        min(dx/|a|)
    """
    dx = np.diff(x)    # x[1] - x[0]
    if np.size(a) > 1:
        a = a[:-1] 
    return np.min(dx/np.abs(a))

def step_uadv_burgers(xx, hh, cfl_cut = 0.98, 
                    ddx = lambda x,y: deriv_dnw(x, y), **kwargs): 
    r"""
    Right hand side of Burger's eq. where a is u, i.e hh.  

    Requires
    --------
        cfl_adv_burger function which computes np.min(dx/a)

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to select the type of spatial derivative.
        By default lambda x,y: deriv_dnw(x, y)


    Returns
    -------
    dt : `array`
        time interval
    unnt : `array`
        right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """
    if 'a' in kwargs: 
        a = kwargs['a']
    else: 
        a = hh
    dt = cfl_diff_burger(a[:-1], xx)*cfl_cut
    rhs = -a*ddx(xx, hh)
    return dt, rhs 

def cfl_diff_burger(a,x): 
    r"""
    Computes the dt_fact, i.e., Courant, Fredrich, and 
    Lewy condition for the diffusive term in the Burger's eq. 

    Parameters
    ----------
    a : `float` or `array` 
        Either constant, or array which multiply the right hand side of the Burger's eq.
    x : `array`
        Spatial axis. 

    Returns
    -------
    `float`
        min(dx/|a|)
    """
    dx = np.diff(x)    # x[1] - x[0]
    return np.min(dx/np.abs(a))

def evolv_adv_burgers(xx, hh, nt, a, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.
    Requires
    ----------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
    ddx : `lambda function`
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y).  
    bnd_type : `string`
        Allows to select the type of boundaries. 
        By default 'wrap'.
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1].

    Returns
    ------- 
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """

    t = np.zeros(nt)
    unnt = np.zeros((len(xx), nt))
    unnt[:,0] = hh 

    for i in range(0, nt-1): 
        dt, rhs = step_adv_burgers(xx, unnt[:, i], a, cfl_cut=cfl_cut, ddx=ddx)

        ## Computes u(t+1)
        unnt_temp = unnt[:, i] + rhs*dt 

        ## Set the boundaries 
        if bnd_limits[1] != 0: 
            # downwind and central
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]] 
        else: 
            # upwind
            unnt1_temp = unnt_temp[bnd_limits[0]:] 

        ## Updates in time 
        unnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt 

    return t, unnt

def evolv_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being u.

    Requires
    --------
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `float`
        constant value to limit dt from cfl_adv_burger. 
        By default 0.98.
    ddx : `lambda function` 
        Allows to change the space derivative function. 
    bnd_type : `string` 
        It allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array` 
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    t         = np.zeros(nt)
    unnt      = np.zeros((len(xx), nt))
    unnt[:,0] = hh 

    for i in range(0, nt-1): 
        dt, rhs = step_uadv_burgers(xx, unnt[:, i], cfl_cut=cfl_cut, ddx=ddx)

        ## Computes u(t+1)
        unnt_temp = unnt[:, i] + rhs*dt 

        ## Set the boundaries 
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]] # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]               # upwind

        ## Updates in time 
        unnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt 

    return t, unnt

def evolv_Lax_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Lax method.

    Requires
    -------- 
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `array`
        Lambda function allows to change the space derivative function.
        By derault  lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries 
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    t = np.zeros(nt)
    unnt = np.zeros((len(xx), nt))
    unnt[:,0] = hh 

    for i in range(nt-1): 
        if 'a' in kwargs: 
            dt, rhs = step_uadv_burgers(xx, unnt[:, i], cfl_cut=cfl_cut, ddx=ddx, \
                a = kwargs['a'])
        else: 
            dt, rhs = step_uadv_burgers(xx, unnt[:, i], cfl_cut=cfl_cut, ddx=ddx)
        dx = xx[1] - xx[0]

        ## Computes u(t+1)
        unn1 = 0.5*(np.roll(unnt[:,i], -1) + np.roll(unnt[:,i], 1)) 
        unn2 = dt*unnt[:, i]/(2*dx) *(np.roll(unnt[:,i], -1) - np.roll(unnt[:,i], 1)) 
        unnt_temp = unn1 - unn2

        ## Set the boundaries 
        if bnd_limits[1] > 0: 
            # downwind and central
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]] 
        else: 
            # upwind
            unnt1_temp = unnt_temp[bnd_limits[0]:] 

        ## Updates in time 
        unnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt 

    return t, unnt

def evolv_Lax_adv_burgers(xx, hh, nt, a, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being a a fix constant or array.

    Requires
    --------
    step_adv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:,0] = hh

    for i in range(0, nt-1):
        dt, rhs = step_adv_burgers(xx, unnt[:,i], a=a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)

        ## Compute u(t+1)
        unnt_temp = 0.5*(np.roll(unnt[:,i], -1) + np.roll(unnt[:,i], 1)) + rhs*dt

        ## Set the boundaries
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]                # upwind
        
        ## Update in time 
        unnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt 

    return t, unnt

def evolv_Rie_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Rie method.

    Requires
    -------- 
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `array`
        Lambda function allows to change the space derivative function.
        By derault  lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries 
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    t = np.zeros(nt)
    unnt = np.zeros((len(xx), nt))
    unnt[:,0] = hh 
    dx = xx[1] - xx[0]

    for i in range(nt-1): 
        ## Computes the left and right of u 
        uL = unnt[:,i] # x < 0 u[i]
        # uR = unnt[:,i+1] # x > 0 u[i+1] # why does this not work???  
        uR = np.roll(unnt[:,i], -1)

        ## Compute the flux FL and FR 
        FL = 0.5*uL**2
        FR = 0.5*uR**2

        ## Compute the propagation speed v_a 
        v_a = np.max(np.array([np.abs(uL), np.abs(uR)]), axis=0)

        halfstep = 0.5*(FL + FR) - 0.5*v_a*(uR - uL)
        rhs = ( halfstep - np.roll(halfstep, 1) ) / dx 
        dt = cfl_diff_burger(v_a[:-1], xx)

        ## Computes u(t+1)
        unnt_temp = unnt[:, i] - rhs*dt

        ## Set the boundaries 
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]                # upwind

        ## Updates in time 
        unnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt 

    return t, unnt

def evolv_RieLax_uadv_burgers(xx, hh, nt, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs):
    r"""
    Advance nt time-steps in time the burger eq for a being u using the Rie method.

    Requires
    -------- 
    step_uadv_burgers

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    cfl_cut : `array`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `array`
        Lambda function allows to change the space derivative function.
        By derault  lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries 
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:, 0] = hh
    dx = xx[1] - xx[0]

    for i in range(0, nt-1): 
        ## FLUX RIE
        ## Computes the left and right of u 
        # uL = unnt[:,i] # x < 0 u[i]
        uL = np.roll(unnt[:,i], 0)
        uR = np.roll(unnt[:,i], -1)

        ## Compute the flux FL and FR 
        FL = 0.5*uL**2
        FR = 0.5*uR**2

        ## Compute the propagation speed v_a 
        v_a = np.max(np.array([np.abs(uL), np.abs(uR)]), axis=0)
        dt = cfl_diff_burger(v_a[:-1], xx)

        f_rie = 0.5*(FL + FR) - 0.5*v_a*(uR - uL)

        ## FLUX LAX
        unn1 = 0.5*(np.roll(unnt[:,i], -1) + np.roll(unnt[:,i], 1)) 
        unn2 = dt*unnt[:, i]/(2*dx) *(np.roll(unnt[:,i], -1) - np.roll(unnt[:,i], 1)) 
        f_lax = unn1 - unn2

        ## Flux limiter
        r = (unnt[:,i] - unnt[:,i-1]) / (unnt[:,i+1] + unnt[:,i])
        thetas = np.array([1., 2.])
        mins = np.zeros((len(thetas), len(r)))
        for j, theta in enumerate(thetas):
            mins[j] = np.min(( np.min(theta*r), np.min((1. + r)/2.), theta ))
        phi = np.max((0, np.max(mins)))

        ## TOTAL FLUX 
        f = f_rie + phi * (f_lax - f_rie)
        rhs = (f - np.roll(f, 1))
        
        ## Computes u(t+1)
        unnt_temp = unnt[:, i] - rhs*dt / dx

        ## Set the boundaries 
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]                # upwind

        ## Updates in time 
        unnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt 

    return t, unnt

###################
### EXERCISE 6.- ##
###################

def ops_Lax_LL_Add(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Additive Operator Splitting scheme.  Both steps are 
    with a Lax method. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        It allows to select the type of boundaries 
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    t = np.zeros(nt)
    unnt = np.zeros((len(xx), nt))
    unnt[:,0] = hh

    for i in range(nt-1):
        dt_u, rhs_u = step_adv_burgers(xx, unnt[:,i], a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        dt_v, rhs_v = step_adv_burgers(xx, unnt[:,i], b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)

        dt = dt_v - dt_u

        ## Compute u(t+1)
        unn = 0.5*(np.roll(unnt[:,i], -1) + np.roll(unnt[:,i], 1)) + rhs_u*dt
        vnn = 0.5*(np.roll(unnt[:,i], -1) + np.roll(unnt[:,i], 1)) + rhs_v*dt
        unnt_temp = unn + vnn - unnt[:,i]

        ## Set the boundaries
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]                # upwind
        
        ## Update in time 
        unnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt 

    return t, unnt

def ops_Lax_LL_Lie(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Lie-Trotter Operator Splitting scheme.  Both steps are 
    with a Lax method. 

    Requires: 
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float` 
        Limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    t = np.zeros(nt)
    unnt = np.zeros((len(xx), nt))
    unnt[:,0] = hh

    for i in range(nt-1):
        dt_u = cfl_adv_burger(a, xx) * cfl_cut
        dt_v = cfl_adv_burger(b, xx) * cfl_cut

        dt = np.min([dt_u, dt_v])

        _, rhs_u = step_adv_burgers(xx, unnt[:,i], a=a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        unn = 0.5*(np.roll(unnt[:,i], -1) + np.roll(unnt[:,i], 1)) + rhs_u*dt

        # v(t^n) = u(t^n+1)
        _, rhs_v = step_adv_burgers(xx, unn, a=b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        vnn = 0.5*(np.roll(unn, -1) + np.roll(unn, 1)) + rhs_v*dt


        ## Set the boundaries
        unnt_temp = unn + vnn - unnt[:,i]
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]                # upwind
        
        ## Update in time 
        unnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt 

    return t, unnt

def ops_Lax_LL_Strang(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Lie-Trotter Operator Splitting scheme. Both steps are 
    with a Lax method. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger
    numpy.pad for boundaries. 

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger.
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function.
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default `wrap`
    bnd_limits : `list(int)` 
        The number of pixels that will need to be updated with the boundary information.
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    t = np.zeros(nt)
    unnt = np.zeros((len(xx), nt))
    unnt[:,0] = hh

    for i in range(nt-1):
        dt_u = cfl_adv_burger(a, xx) * cfl_cut
        dt_v = cfl_adv_burger(b, xx) * cfl_cut

        dt = np.min([dt_u, dt_v])

        _, rhs_u = step_adv_burgers(xx, unnt[:,i], a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        unn = 0.5*(np.roll(unnt[:,i], -1) + np.roll(unnt[:,i], 1)) + rhs_u*dt*0.5 #half a timestep

        # v(t^n) = u(t^n+1)
        _, rhs_v = step_adv_burgers(xx, unn[i], b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        vnn = 0.5*(np.roll(unn, -1) + np.roll(unn, 1)) + rhs_v*dt

        dt_w, rhs_w = step_adv_burgers(xx, vnn, a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        wnn = 0.5*(np.roll(vnn, -1) + np.roll(vnn, 1)) + rhs_w*dt_w*0.5 #half a timestep

        ## Set the boundaries
        unnt_temp = wnn
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]                # upwind
        
        ## Update in time 
        unnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt 

    return t, unnt

def osp_Lax_LH_Strang(xx, hh, nt, a, b, cfl_cut = 0.98, 
        ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 
    r"""
    Advance nt time-steps in time the burger eq for a being a and b 
    a fix constant or array. Solving two advective terms separately 
    with the Strang Operator Splitting scheme. One step is with a Lax method 
    and the second step is the Hyman predictor-corrector scheme. 

    Requires
    --------
    step_adv_burgers
    cfl_adv_burger

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    b : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float` 
        Limit dt from cfl_adv_burger. 
        By default 0.98
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)
    bnd_type : `string`
        It allows to select the type of boundaries. 
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By default [0,1]

    Returns
    -------
    t : `array`
        Time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    t = np.zeros(nt)
    unnt = np.zeros((len(xx), nt))
    unnt[:,0] = hh

    for i in range(nt-1):
        dt = cfl_adv_burger(a, xx) # need common timestep??? maybe

        dt_u, rhs_u = step_adv_burgers(xx, unnt[:,i], a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        unn = 0.5*(np.roll(unnt[:,i], -1) + np.roll(unnt[:,i], 1)) + rhs_u*dt_u*0.5 #half a timestep

        # v(t^n) = u(t^n+1)
        dt_v, rhs_v = step_adv_burgers(xx, unn, b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)

        # Hyman predictor-corrector scheme
        if i == 0: # First step 
            unn, uold, dt_v = hyman(xx, unn, dt, b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        else: # The rest of the steps 
            unn, uold, dt_v = hyman(xx, unn, dt, b, cfl_cut = cfl_cut, ddx = ddx, fold = uold, dtold = dt_v, **kwargs)


        vnn = 0.5*(np.roll(unn, -1) + np.roll(unn, 1)) + rhs_v*dt_v # full timestep 

        dt_w, rhs_w = step_adv_burgers(xx, vnn, a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        wnn = 0.5*(np.roll(vnn, -1) + np.roll(vnn, 1)) + rhs_w*dt_w*0.5 #half a timestep

        ## Set the boundaries
        unnt_temp = wnn
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]                # upwind
        
        ## Update in time 
        unnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt 

    return t, unnt

###################
### EXERCISE 5.2 ##
###################

def step_diff_burgers(xx, hh, a, ddx = lambda x,y: deriv_cent(x, y), **kwargs): 
    r"""
    Right hand side of the diffusive term of Burger's eq. where nu can be a constant or a function that 
    depends on xx. 
    
    Parameters
    ----------    
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    ddx : `lambda function`
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_dnw(x, y)

    Returns
    -------
    `array`
        Right hand side of (u^{n+1}-u^{n})/dt = from burgers eq, i.e., x \frac{\partial u}{\partial x} 
    """
    # dt = cfl_diff_burger(a[:-1], xx)
    # rhs = a*(np.roll(hh, -1) - 2*hh - np.roll(hh, 1))

    rhs = a*ddx(xx, hh)
    return rhs 

def NR_f(xx, un, uo, a, dt, **kwargs): 
    r"""
    NR F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx. (u_new)
    uo : `array`
        Function that depends on xx. (u_original)
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float` 
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """
    dx = xx[1] - xx[0]
    return un - uo - step_diff_burgers(xx, un, a) * dt #/ (dx**2)

def jacobian(xx, un, a, dt, **kwargs): 
    r"""
    Jacobian of the F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float` 
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """
    dx = xx[1] - xx[0]
    J = np.zeros((len(xx), len(xx)))
    for i in range(len(xx)):
        J[i, i] = 1 + dt * 2*a / dx**2
        if i < np.size(xx) - 1: 
            J[i, i+1] = -dt*a / dx**2 
        if i > 1: 
            J[i, i-1] = -dt*a / dx**2
    return J

def Newton_Raphson(xx, hh, a, dt, nt, toll=1e-5, ncount=2, 
            bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    r"""
    NR scheme for the burgers equation. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `float`
        Time interval
    nt : `int`
        Number of iterations
    toll : `float` 
        Error limit.
        By default 1e-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]

    Returns
    -------
    t : `array`
        Array of time. 
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    errt : `array`
        Error for each timestep
    countt : `list(int)`
        number iterations for each timestep
    """    
    err=1.
    unnt = np.zeros((np.size(xx),nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:,0] = hh
    t=np.zeros((nt))
    
    ## Looping over time 
    for it in range(1,nt): 
        uo=unnt[:,it-1]
        ug=unnt[:,it-1] 
        count = 0 
        # iteration to reduce the error. 
        while ((err >= toll) and (count < ncount)): 

            jac = jacobian(xx, ug, a, dt) # Jacobian 
            ff1=NR_f(xx, ug, uo, a, dt) # F 
            # Inversion: 
            un = ug - np.matmul(np.linalg.inv(
                    jac),ff1)

            # error: 
            err = np.max(np.abs(un-ug)/(np.abs(un)+toll)) # error
            #err = np.max(np.abs(un-ug))
            errt[it]=err

            # Number of iterations
            count+=1
            countt[it]=count
            
            # Boundaries 
            if bnd_limits[1]>0: 
                u1_c = un[bnd_limits[0]:-bnd_limits[1]]
            else: 
                u1_c = un[bnd_limits[0]:]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un 
        err=1.
        t[it] = t[it-1] + dt
        unnt[:,it] = un
        
    return t, unnt, errt, countt

def NR_f_u(xx, un, uo, dt, **kwargs): 
    r"""
    NR F function.

    Parameters
    ----------  
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    uo : `array`
        Function that depends on xx.
    a : `float` and `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        function  u^{n+1}_{j}-u^{n}_{j} - a (u^{n+1}_{j+1} - 2 u^{n+1}_{j} -u^{n+1}_{j-1}) dt
    """

def jacobian_u(xx, un, dt, **kwargs): 
    """
    Jacobian of the F function. 

    Parameters
    ----------   
    xx : `array`
        Spatial axis. 
    un : `array`
        Function that depends on xx.
    a : `float` and `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.
    dt : `int`
        Time interval

    Returns
    -------
    `array`
        Jacobian F_j'(u^{n+1}{k})
    """    

def Newton_Raphson_u(xx, hh, dt, nt, toll= 1e-5, ncount=2, 
            bnd_type='wrap', bnd_limits=[1,1], **kwargs):
    """
    NR scheme for the burgers equation. 

    Parameters
    ----------
    xx : `array`
        Spatial axis.
    hh : `array`
        Function that depends on xx.
    dt : `float` 
        Time interval
    nt : `int`
        Number of iterations
    toll : `float` 
        Error limit.
        By default 1-5
    ncount : `int`
        Maximum number of iterations.
        By default 2
    bnd_type : `string` 
        Allows to select the type of boundaries.
        By default 'wrap'
    bnd_limits : `list(int)`
        Array of two integer elements. The number of pixels that
        will need to be updated with the boundary information.
        By default [1,1]        

    Returns
    -------
    t : `array`
        Time. 
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    errt : `array`
        Error for each timestep
    countt : `array(int)` 
        Number iterations for each timestep
    """    
    err=1.
    unnt = np.zeros((np.size(xx),nt))
    errt = np.zeros((nt))
    countt = np.zeros((nt))
    unnt[:,0] = hh
    t=np.zeros((nt))
    
    ## Looping over time 
    for it in range(1,nt): 
        uo=unnt[:,it-1]
        ug=unnt[:,it-1] 
        count = 0 
        # iteration to reduce the error. 
        while ((err >= toll) and (count < ncount)): 

            jac = jacobian_u(xx, ug, dt) # Jacobian 
            ff1=NR_f_u(xx, ug, uo, dt) # F 
            # Inversion: 
            un = ug - np.matmul(np.linalg.inv(
                    jac),ff1)

            # error
            err = np.max(np.abs(un-ug)/(np.abs(un)+toll)) 
            errt[it]=err

            # Number of iterations
            count+=1
            countt[it]=count
            
            # Boundaries 
            if bnd_limits[1]>0: 
                u1_c = un[bnd_limits[0]:-bnd_limits[1]]
            else: 
                u1_c = un[bnd_limits[0]:]
            un = np.pad(u1_c, bnd_limits, bnd_type)
            ug = un 
        err=1.
        t[it] = t[it-1] + dt
        unnt[:,it] = un
        
    return t, unnt, errt, countt


###################
### EXERCISE 5.3 ##
###################

def taui_sts(nu, niter, iiter): 
    """
    STS parabolic scheme. [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}

    Parameters
    ----------   
    nu : `float`
        Coefficient, between (0,1).
    niter : `int` 
        Number of iterations
    iiter : `int`
        Iterations number

    Returns
    -------
    `float` 
        [(nu -1)cos(pi (2 iiter - 1) / 2 niter) + nu + 1]^{-1}
    """
    return ( (nu - 1)*np.cos(np.pi*(2*iiter - 1) / (2*niter)) + nu + 1 )**(-1)

def evol_sts(xx, hh, nt, a, cfl_cut = 0.45, 
        ddx = lambda x,y: deriv_cent(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], nu=0.9, n_sts=10): 
    """
    Evolution of the STS method. 

    Parameters
    ----------
    xx : `array`
        Spatial axis. 
    hh : `array`
        Function that depends on xx.
    nt : `int`
        Number of iterations
    a : `float` or `array` 
        Either constant, or array which multiply the right hand side of the Burger's eq.
    cfl_cut : `float`
        Constant value to limit dt from cfl_adv_burger. 
        By default 0.45
    ddx : `lambda function` 
        Allows to change the space derivative function. 
        By default lambda x,y: deriv_cent(x, y)
    bnd_type : `string` 
        Allows to select the type of boundaries
        by default 'wrap'
    bnd_limits : `list(int)`
        List of two integer elements. The number of pixels that
        will need to be updated with the boundary information. 
        By defalt [0,1]
    nu : `float`
        STS nu coefficient between (0,1).
        By default 0.9
    n_sts : `int`
        Number of STS sub iterations. 
        By default 10

    Returns
    -------
    t : `array`
        time 1D array
    unnt : `array`
        Spatial and time evolution of u^n_j for n = (0,nt), and where j represents
        all the elements of the domain. 
    """
    t = np.zeros((nt))
    unnt = np.zeros((len(xx), nt))
    unnt[:,0] = hh

    dx = xx[1] - xx[0]
    dt_cfl = cfl_cut * np.min(dx**2/(4*np.abs(a))) # From the cfl_cut found in 5a 

    tmp_u = np.zeros((len(xx), n_sts))
    for i in range(nt-1): 
        
        un = unnt[:,i].copy()
        rhs = step_diff_burgers(xx, unnt[:,i], a, ddx)
        taui = 0
        dt_sts = 0
        tmp_u[:,0] = unnt[:,i]

        for ists in range(n_sts): 
            # rhs = step_diff_burgers(xx, unnt[:,i], a, ddx)
            rhs = step_diff_burgers(xx, hh, a, ddx)

            # taui += taui_sts(nu, n_sts, ists+1)*dt_cfl
            # dt_sts += taui
            # un += rhs*taui
            # rhs = step_diff_burgers(xx, unnt[:,i], a, ddx)

            # Compute u(t+1)
            tmp_u[:,ists] = hh + rhs*taui_sts(nu, n_sts, ists+1)*dt_cfl
            hh = tmp_u[:,ists]

            unnt_temp = tmp_u[:,ists]
            # unnt_temp = unnt[:,j] + rhs*taui_sts(nu, n_sts, j+1) * delta_cfl
        
        unnt[:,i+1] = unnt[:,i] + rhs*dt_sts
        unnt_temp = unnt[:,i+1]

        ## Set the boundaries
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]                # upwind

        ## Update in time 
        unnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        dt          = taui
        t[i+1]      = t[i] + dt

    return t, unnt

def hyman(xx, f, dth, a, fold=None, dtold=None,
        cfl_cut=0.8, ddx = lambda x,y: deriv_dnw(x, y), 
        bnd_type='wrap', bnd_limits=[0,1], **kwargs): 

    dt, u1_temp = step_adv_burgers(xx, f, a, ddx=ddx)

    if (np.any(fold) == None):
        firstit=False
        fold = np.copy(f)
        f = (np.roll(f,1)+np.roll(f,-1))/2.0 + u1_temp * dth 
        dtold=dth

    else:
        ratio = dth/dtold
        a1 = ratio**2
        b1 =  dth*(1.0+ratio   )
        a2 =  2.*(1.0+ratio    )/(2.0+3.0*ratio)
        b2 =  dth*(1.0+ratio**2)/(2.0+3.0*ratio)
        c2 =  dth*(1.0+ratio   )/(2.0+3.0*ratio)

        f, fold, fsav = hyman_pred(f, fold, u1_temp, a1, b1, a2, b2)
        
        if bnd_limits[1]>0: 
            u1_c =  f[bnd_limits[0]:-bnd_limits[1]]
        else: 
            u1_c = f[bnd_limits[0]:]
        f = np.pad(u1_c, bnd_limits, bnd_type)

        dt, u1_temp = step_adv_burgers(xx, f, a, cfl_cut, ddx=ddx)

        f = hyman_corr(f, fsav, u1_temp, c2)

    if bnd_limits[1]>0: 
        u1_c = f[bnd_limits[0]:-bnd_limits[1]]
    else: 
        u1_c = f[bnd_limits[0]:]
    f = np.pad(u1_c, bnd_limits, bnd_type)
    
    dtold=dth

    return f, fold, dtold

def hyman_corr(f, fsav, dfdt, c2):

    return  fsav  + c2* dfdt

def hyman_pred(f, fold, dfdt, a1, b1, a2, b2): 

    fsav = np.copy(f)
    tempvar = f + a1*(fold-f) + b1*dfdt
    fold = np.copy(fsav)
    fsav = tempvar + a2*(fsav-tempvar) + b2*dfdt    
    f = tempvar
    
    return f, fold, fsav



































        # # Calculate the fluxes
        # F_h = u*u
        # F_m = u*u*u
        # F_e = u*u*u + Pg

        # # Calculate the right hand side of the equations
        # ddx_h = ddx(xx, u)
        # ddx_m = ddx(xx, u*u)
        # ddx_e = ddx(xx, u*u + Pg)

        # # Calculate the new values of the variables
        # u = u - dt*ddx_h
        # u = u - dt*ddx_m
        # u = u - dt*ddx_e