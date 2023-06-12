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
    return (np.roll(hh, -1) - hh) / (np.roll(xx, -1) - xx)

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
    dx = np.gradient(x) # x[1] - x[0]
    return np.min(dx/np.abs(a))

def cfl_diff_burger(a, x): 
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
    dx = np.gradient(x) # x[1] - x[0]
    return np.min(dx**2 / (2*np.abs(a)))

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
    dt = cfl_adv_burger(hh, xx)*cfl_cut
    # dt = cfl_diff_burger(a[:-1], xx)*cfl_cut
    rhs = -a*ddx(xx, hh)
    return dt, rhs 

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

### XXX I believe the function beneath is wrong: 3a 
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



###################
### EXERCISE 4  ###
###################

# XXX Dont need the rhs - use dt directly? 
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
        dt = cfl_adv_burger(v_a, xx)

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
        # dt = cfl_diff_burger(v_a[:-1], xx)
        dt = cfl_adv_burger(v_a, xx)

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

        dt = np.min([dt_v, dt_u]) * 0.5
        dx = xx[1] - xx[0]

        ## Compute u(t+1)
        unn = 0.5*(np.roll(unnt[:,i], -1) + np.roll(unnt[:,i], 1)) - ((a*dt)/(2*dx) * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1))) #+ rhs_u*dt
        vnn = 0.5*(np.roll(unnt[:,i], -1) + np.roll(unnt[:,i], 1)) - ((b*dt)/(2*dx) * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1))) #+ rhs_v*dt
        unnt_temp = unn + vnn - (0.5 * np.roll(unnt[:, i], -1) + 0.5 * np.roll(unnt[:, i], 1)) #- unnt[:,i]  # made stable by taking the surrounding half steps 

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
    vnnt = np.zeros((len(xx), nt))
    vnnt[:,0] = hh

    for i in range(nt-1):
        dt_u = cfl_adv_burger(a, xx) * cfl_cut
        dt_v = cfl_adv_burger(b, xx) * cfl_cut

        dt = np.min([dt_u, dt_v]) * 0.5
        dx = xx[1] - xx[0]

        # _, rhs_u = step_adv_burgers(xx, unnt[:,i], a=a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        unnt[:, i] = 0.5*(np.roll(vnnt[:,i], -1) + np.roll(vnnt[:,i], 1)) - ((a*dt) / (2*dx) * (np.roll(vnnt[:, i], -1) - np.roll(vnnt[:, i], 1))) #+ rhs_u*dt

        # v(t^n) = u(t^n+1)
        # _, rhs_v = step_adv_burgers(xx, unn, a=b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        vnnt[:, i] = 0.5*(np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - ((b*dt) / (2*dx) * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1))) #+ rhs_v*dt

        unnt_temp = vnnt[:, i]

        ## Set the boundaries
        # unnt_temp = unn + vnn - unnt[:,i]
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]                # upwind
        
        ## Update in time 
        vnnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt 

    return t, vnnt

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
    vnnt = np.zeros((len(xx), nt))
    vnnt[:,0] = hh
    wnnt = np.zeros((len(xx), nt))
    wnnt[:,0] = hh

    for i in range(nt-1):
        dt_u = cfl_adv_burger(a, xx) * cfl_cut
        dt_v = cfl_adv_burger(b, xx) * cfl_cut

        dt = np.min([dt_u, dt_v]) * 0.5 # half a timestep 
        dx = xx[1] - xx[0]

        unnt[:, i] = 0.5*(np.roll(wnnt[:, i], -1) + np.roll(wnnt[:, i], 1)) - ((a*dt) / (4*dx) * (np.roll(wnnt[:, i], -1) - np.roll(wnnt[:, i], 1))) 
        vnnt[:, i] = 0.5*(np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - ((b*dt) / (2*dx) * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1)))
        wnnt[:, i] = 0.5*(np.roll(vnnt[:, i], -1) + np.roll(vnnt[:, i], 1)) - ((a*dt) / (4*dx) * (np.roll(vnnt[:, i], -1) - np.roll(vnnt[:, i], 1)))

        # _, rhs_u = step_adv_burgers(xx, unnt[:,i], a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        # unn = 0.5*(np.roll(unnt[:,i], -1) + np.roll(unnt[:,i], 1)) + rhs_u*dt*0.5 #half a timestep
        # v(t^n) = u(t^n+1)
        # _, rhs_v = step_adv_burgers(xx, unn[i], b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        # vnn = 0.5*(np.roll(unn, -1) + np.roll(unn, 1)) + rhs_v*dt

        # dt_w, rhs_w = step_adv_burgers(xx, vnn, a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        # wnn = 0.5*(np.roll(vnn, -1) + np.roll(vnn, 1)) + rhs_w*dt_w*0.5 #half a timestep

        ## Set the boundaries
        unnt_temp = wnnt[:, i]
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]                # upwind
        
        ## Update in time 
        wnnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt

    return t, wnnt

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
    vnnt = np.zeros((len(xx), nt))
    vnnt[:,0] = hh
    wnnt = np.zeros((len(xx), nt))
    wnnt[:,0] = hh

    for i in range(nt-1):
        # dt_u, rhs_u = step_adv_burgers(xx, unnt[:,i], a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        # dt_v, rhs_v = step_adv_burgers(xx, unnt[:,i], b, cfl_cut = cfl_cut, ddx = ddx, **kwargs) # v(t^n) = u(t^n+1)
        # dt_w, rhs_w = step_adv_burgers(xx, unnt[:,i], a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        # dt = np.min([dt_u, dt_v, dt_w]) * 0.5 # half a timestep

        dt_u = cfl_adv_burger(a, xx) * cfl_cut
        dt_v = cfl_adv_burger(b, xx) * cfl_cut
        dt = np.min([dt_u, dt_v]) * 0.5 # half a timestep 
        dx = xx[1] - xx[0]

        unnt[:, i] = 0.5*(np.roll(wnnt[:, i], -1) + np.roll(wnnt[:, i], 1)) - ((a*dt) / (4*dx) * (np.roll(wnnt[:, i], -1) - np.roll(wnnt[:, i], 1))) 
        vnnt[:, i] = 0.5*(np.roll(unnt[:, i], -1) + np.roll(unnt[:, i], 1)) - ((b*dt) / (2*dx) * (np.roll(unnt[:, i], -1) - np.roll(unnt[:, i], 1)))

        # Hyman predictor-corrector scheme
        if i == 0: # First step 
            vnnt[:, i], uold, dt_v = hyman(xx, unnt[:, i], dt, b, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        else: # The rest of the steps 
            vnnt[:, i], uold, dt_v = hyman(xx, unnt[:, i], dt, b, cfl_cut = cfl_cut, ddx = ddx, fold = uold, dtold = dt_v, **kwargs)

        # dt_w, rhs_w = step_adv_burgers(xx, unnt[:,i], a, cfl_cut = cfl_cut, ddx = ddx, **kwargs)
        wnnt[:, i] = 0.5*(np.roll(vnnt[:, i], -1) + np.roll(vnnt[:, i], 1)) - ((a*dt) / (4*dx) * (np.roll(vnnt[:, i], -1) - np.roll(vnnt[:, i], 1)))

        ## Set the boundaries
        unnt_temp = wnnt[:,i]
        if bnd_limits[1] > 0: 
            unnt1_temp = unnt_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
        else: 
            unnt1_temp = unnt_temp[bnd_limits[0]:]                # upwind
        
        ## Update in time 
        wnnt[:,i+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
        t[i+1]      = t[i] + dt

    return t, wnnt


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
    dt = cfl_diff_burger(a, xx)
    rhs = a*ddx(xx, ddx(xx, hh)) # no minus? 
    return dt, rhs 

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

def tau_sts(nu, n, dt_cfl): 
    a = n / (2 * np.sqrt(nu))
    b1 = (1 + np.sqrt(nu))**2 * n - (1 - np.sqrt(nu))**2 * n
    b2 = (1 + np.sqrt(nu))**2 * n + (1 - np.sqrt(nu))**2 * n
    dt_sts = a * (b1/b2) * dt_cfl
    return dt_sts

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
    tcfl = (dx**2)/a
    tsts = []

    # tmp_u = np.zeros((len(xx), n_sts))
    for i in range(nt-1): 
        ts = []
        unts = np.zeros((np.size(xx), n_sts))
        unts[:, 0] = unnt[:,i]

        for ists in range(0, n_sts-1): 
            # ists -> taui_sts will be zero for the first iteration.
            dti = tcfl * taui_sts(nu, n_sts, ists+1) # taui_sts needs a non-zero integer 
            dt, u1_temp = step_diff_burgers(xx, unts[:,ists], a, cfl_cut=cfl_cut , ddx=ddx)

            u1_temp = unts[:, ists] + u1_temp * dti

             ## Set the boundaries
            if bnd_limits[1] > 0: 
                unnt1_temp = u1_temp[bnd_limits[0]:-bnd_limits[1]]  # downwind and central
            else: 
                unnt1_temp = u1_temp[bnd_limits[0]:]                # upwind

            unts[:, ists+1] = np.pad(unnt1_temp, bnd_limits, bnd_type)
            unntmp = unts[:, ists+1]
            ts.append(dti)
        tsts.append(ts)
        unnt[:, i+1] = unntmp
        t[i+1] = t[i] + np.sum(ts)

    return t, unnt, tsts

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



###################
##### Project #####
###################
from scipy.optimize import fsolve

def sod_shock_tube_analytical(x, t_end, gamma, init):

    Pg_L = init[0]
    Pg_R = init[1]
    rho_L = init[2]
    rho_R = init[3]
    
    # Constants 
    c1 = np.sqrt(gamma*Pg_L/rho_L) # Sound speed left 
    c4 = np.sqrt(gamma*Pg_R/rho_R) # Sound speed right 
    u1 = 0
    u4 = 0

    beta = (gamma - 1)/(2*gamma)
    Gamma = (gamma - 1)/(gamma + 1)

    ratio = lambda P34: ( (1 \
            + ((u1 - u4)*(gamma - 1)/(2*c1)) \
            - ((gamma - 1)*(c4/c1)*(P34 - 1)/np.sqrt(2*gamma*((gamma - 1) + (gamma + 1)*P34))))**(beta**(-1)) \
            )/P34 - (Pg_R/Pg_L)

    P34 = fsolve(ratio, 5)[0]
    if np.isclose(ratio(P34), 0) == False:
        print('Warning: P34 not calculated correctly!')

    Pg_3 = P34*Pg_R
    rho34 = ((Gamma**(-1)) + P34) / (1 + (Gamma**(-1)))
    rho_3 = rho_R*((Pg_3 + Gamma*Pg_R) / (Pg_R + Gamma*Pg_3))

    Pg_2 = Pg_3

    u3 = u4 + c4/gamma*P34*((2*gamma/(gamma + 1))/(P34 + (Gamma)))**0.5 # u2 = u3 ? 
    u2 = u1 + 2*c1/(gamma - 1)*(1 - (Pg_2/Pg_L)**(beta))

    rho_2 = rho_L*(Pg_2/Pg_L)**(1/gamma)

    c3 = np.sqrt(gamma*Pg_3/rho_3)
    c2 = np.sqrt(gamma*Pg_2/rho_2)

    # Set boundaries for x-grid 
    x1 = 0.5 + (u1 - c1)*t_end 
    x2 = 0.5 + (u2 + u4 - c2)*t_end 
    x3 = 0.5 + (u2 + u4)*t_end
    x4 = 0.5 + (c4*np.sqrt(beta + (gamma + 1)/(2*gamma)*P34) + u4)*t_end

    l = len(x)
    x_grid = np.linspace(x1, x2, len(x[int(x1*len(x)):int(x2*len(x))]))

    # Rarefraction wave
    u_rare = 2/(gamma + 1)*(c1 + (x_grid - 0.5)/t_end)
    rho_rare = rho_L*(1 - ((gamma - 1)/2 * u_rare/c1))**(2/(gamma - 1))
    Pg_rare = Pg_L*(1 - (gamma - 1)/2 * u_rare/c1)**(2/(gamma - 1))

    # Initialize the grids
    rho_grid = np.zeros(len(x))
    u_grid = np.zeros(len(x))
    e_grid = np.zeros(len(x))

    # Insert boundary values: 
    rho_grid[:int(x1*l)] = rho_L
    rho_grid[int(x1*l):int(x2*l)] = rho_rare
    rho_grid[int(x2*l):int(x3*l)] = rho_2
    rho_grid[int(x3*l):int(x4*l)] = rho_3
    rho_grid[int(x4*l):] = rho_R

    u_grid[:int(x1*l)] = u1*rho_L
    u_grid[int(x1*l):int(x2*l)] = u_rare*rho_rare
    u_grid[int(x2*l):int(x3*l)] = u2*rho_2
    u_grid[int(x3*l):int(x4*l)] = u2*rho_3
    u_grid[int(x4*l):] = u4*rho_R

    e_grid[:int(x1*l)] = 0.5*rho_L*u1**2 + Pg_L/(gamma-1)
    e_grid[int(x1*l):int(x2*l)] = 0.5*rho_rare*u_rare**2 + Pg_rare/(gamma-1)
    e_grid[int(x2*l):int(x3*l)] = 0.5*rho_2*u2**2 + Pg_2/(gamma-1)
    e_grid[int(x3*l):int(x4*l)] = 0.5*rho_3*u2**2 + Pg_3/(gamma-1)
    e_grid[int(x4*l):] = 0.5*rho_R*u4**2 + Pg_R/(gamma-1)
    
    return rho_grid, u_grid, e_grid




def evol_hd_sts(xx, rho, u, Tg, gamma=5/3, kappa=0, kB=1, mH=1, nt=1000, cfl_cut=0.48, \
                    ddx=lambda x,y: deriv_cent(x, y), bnd_type='wrap', bnd_limits=[1,1], \
                        nu=0.05, S=lambda x, t: 0, g=0, **kwargs): 
    """
    Evolve the hydrodynamical system nt steps in time, 
    including thermal conduction and a possible source term 
    with the super timestepping method.

    Parameters
    ----------
        xx : `array`
            Spatial axis. 
        rho : `array`
            Initial density.
        u : `array`
            Initial velocity.
        Tg : `array`
            Initial temperature.
        gamma : `float`
            Adiabatic index.
        kappa : `float`
            Thermal conduction coefficient.
        kb : `float`
            Boltzmann constant.
        mh : `float`
            Mass of hydrogen atom.
        nt : `int`
            Number of timesteps.
        cfl_cut : `float`
            CFL condition to limit the timestep. 
        ddx : `function`
            Derivative method.
        bnd_type : `str`
            Boundary type.
        bnd_limits : `list`
            Boundary limits.
        nu : `float`
            Dampening factor between 0 and 1
        n_sts : `int`
            Number of super timesteps, positive 
        S : `function`
            Source term.

    Returns
    -------
        t : `array`
            Time axis.
        rho_arr : `array`
            Density evolution.
        mom_arr : `array`
            Momentum evolution.
        e_arr : `array`
            Energy evolution.
    """

    eps = 1e-6 # to avoid division by zero

    ## Declare the arrays ##
    t = np.zeros((nt))               # time
    rx_arr = np.zeros((len(xx), nt)) # density
    px_arr = np.zeros((len(xx), nt)) # momentum
    e_arr1 = np.zeros((len(xx), nt)) # energy without thermal conduction
    e_arr2 = np.zeros((len(xx), nt)) # energy with thermal conduction
    e_arr  = np.zeros((len(xx), nt)) # energy
    
    ## Initialize variables ##
    rx_arr[:,0] = rho
    px_arr[:,0] = rho*u          # Momentum        from u = p/rho 
    Pg          = rho*kB*Tg/mH   # Gas pressure    from Pg = n*kB*Tg where n = rho/mH
    e_arr[:,0]  = Pg/(gamma - 1) # Internal energy from Pg = (gamma - 1)*e

    ## Evolve the system in time ##
    for i in range(nt - 1): 
        
        ## Update variables ##
        u  = px_arr[:,i]/(rx_arr[:,i] + eps)                 # Velocity
        Tg = e_arr[:,i]*mH*(gamma - 1) / (kB*rx_arr[:,i])    # Gas temperature
        Pg = rx_arr[:,i]*kB*Tg / mH                          # Gas pressure
        cs = np.sqrt(np.abs(gamma*Pg / (rx_arr[:,i] + eps))) # Sound speed 
        S_arr = S(xx, i)

        ## CFL condition ##
        dt1 = np.min(np.gradient(xx)    / np.abs(u + eps))         # Eigenvalue u 
        dt2 = np.min(np.gradient(xx)    / np.abs(u - cs + eps))    # Eigenvalue u - c_s
        dt3 = np.min(np.gradient(xx)    / np.abs(u + cs + eps))    # Eigenvalue u + c_s
        dt4 = np.min(np.gradient(xx)**2 / np.abs((2*kappa) + eps)) # Thermal conduction term

        dt_cfl = np.min(np.array([dt1, dt2, dt3]))
        n_sts = np.int(np.sqrt(dt_cfl/ dt4) + 1)   # dt_cfl/ dt4 > 1
        dt_sts = tau_sts(nu, n_sts, dt4)

        dt = cfl_cut*np.min(np.array([dt_cfl, dt_sts]))    # Final timestep

        # n_sts = 10
        # dt_cfl = np.min(np.array([dt1, dt2, dt3, dt4]))
        # # can find which n_sts to use in order to get dt_sts to match dt_cfl
        # dt_sts = tau_sts(nu, n_sts, dt_cfl)

        ###--- Time evolution for density and momentum ---###

        ## Spatially averaged variables ##
        rx_Lax = (np.roll(rx_arr[:,i], -1) + rx_arr[:,i] + np.roll(rx_arr[:,i], 1)) / 3
        px_Lax = (np.roll(px_arr[:,i], -1) + px_arr[:,i] + np.roll(px_arr[:,i], 1)) / 3

        ## RHS of the HD equations ##
        rhs_rx = -ddx(xx, rx_arr[:,i]*u)                       # Continuity equation
        rhs_px  = -ddx(xx, px_arr[:,i]*u + Pg) - rx_arr[:,i]*g  # Momentum balance

        rx_t = rx_Lax + rhs_rx*dt # Density
        px_t = px_Lax + rhs_px*dt  # Momentum
            
        # Boundary conditions: 
        if bnd_limits[1] > 0:  # downwind or centered scheme
            rx_Bc = rx_t[bnd_limits[0]: -bnd_limits[1]]
            px_Bc = px_t[bnd_limits[0]: -bnd_limits[1]]
        else:
            rx_Bc = rx_t[bnd_limits[0]:]  # upwind
            px_Bc = px_t[bnd_limits[0]:]
        
        ## Update arrays with boundary conditions ##
        rx_arr[:,i+1] = np.pad(rx_Bc, bnd_limits, bnd_type)
        px_arr[:,i+1] = np.pad(px_Bc, bnd_limits, bnd_type)

    ###--- Time evolution for energy (Operator Splitting) ---###

        if np.abs(kappa) > 0: # Thermal conduction 
            
            ## Spatially averaged variables ## 
            e_Lax1 = (np.roll(e_arr[:,i], -1) + e_arr[:,i] + np.roll(e_arr[:,i], 1)) / 3

            ## RHS of the HD energy equation (no thermal conduction) ##
            rhs_energy1 =  -ddx(xx, e_arr[:,i]*u) - Pg*ddx(xx, u) + S_arr # energy balance 1

            e_t1 = e_Lax1 + rhs_energy1*dt # Energy

            # Boundary conditions: 
            if bnd_limits[1] > 0:  # downwind or centered scheme
                e_Bc1 = e_t1[bnd_limits[0]: -bnd_limits[1]]
            else:
                e_Bc1 = e_t1[bnd_limits[0]:]
            
            ## Update arrays with boundary conditions ##
            e_arr1[:,i+1] = np.pad(e_Bc1, bnd_limits, bnd_type)

            ### Super timesteppig for thermal conduction ###

            ## Declare the arrays ##
            e_sts = np.zeros((np.size(xx), n_sts))
            t_sts = np.zeros((n_sts))

            ## Initialize variables ##
            e_sts[:,0] = e_arr[:,i]

            ## Evolve the system in time ##
            for j in range(n_sts-1):
                ## Initialize variables based on the previous super-timesteps ##
                dtj = dt_cfl*taui_sts(nu, n_sts, j)
                # Tg_sts = e_sts[:,j]*mH*(gamma - 1) / kB # XXX 
                Tg_sts = e_sts[:,j]*mH*(gamma - 1) / (kB*rx_arr[:,i])
                
                ## Spatially averaged variables ##
                e_Lax2 = (np.roll(e_sts[:,j], -1) + e_sts[:,j]  + np.roll(e_sts[:,j], 1)) / 3

                ## RHS of the HD equations ##
                rhs_energy2 = kappa*ddx(xx, ddx(xx, Tg_sts)) 

                e_t2 = e_Lax2 + rhs_energy2*dtj # Energy

                # Boundary conditions: 
                if bnd_limits[1] > 0:  # downwind or centered scheme
                    e_Bc2 = e_t2[bnd_limits[0]: -bnd_limits[1]]
                else:
                    e_Bc2 = e_t2[bnd_limits[0]:]
                
                ## Update arrays with boundary conditions ##
                e_sts[:,j+1] = np.pad(e_Bc2, bnd_limits, bnd_type)
                t_sts[j+1]   = t_sts[j] + dtj
            
            ## Variable update for normal stepping ##
            e_arr2[:,i+1] = e_sts[:, n_sts-1]

            ## Adding the advances in the operator split to get the full time advance ##
            e_arr[:,i+1] = e_arr1[:,i+1] + e_arr2[:,i+1] - ((np.roll(e_arr[:,i],1) + e_arr[:,i] + np.roll(e_arr[:,i],-1)) / 3)
            t[i+1]       = t[i] + np.sum(dt)

        else: # No Thermal Conduction

            ## Spatially averaged variables ## 
            e_Lax = (np.roll(e_arr[:,i], -1) + e_arr[:,i] + np.roll(e_arr[:,i], 1)) / 3

            ## RHS of the HD equations ##
            rhs_energy = -ddx(xx, e_arr[:,i]*u) - Pg*ddx(xx, u) + S_arr

            e_t = e_Lax + rhs_energy*dt # Energy
            
            # Boundary conditions:
            if bnd_limits[1] > 0:  # downwind or centered scheme
                e_Bc = e_t[bnd_limits[0]: -bnd_limits[1]]
            else:
                e_Bc = e_t[bnd_limits[0]:]
            
            ## Update arrays witn boundary conditions ##
            e_arr[:,i+1] = np.pad(e_Bc, bnd_limits, bnd_type)
            t[i+1]       = t[i] + dt

    return t, rx_arr, px_arr, e_arr


import matplotlib.pyplot as plt 

def plot_hd_init(xx, density, velocity, temperature, pressure, energy, \
                    source_function=None): 
    """ 
    Function for plotting the initial values for the density, velocity, 
    temperature, pressure and energy. 

    Parameters 
    ----------
    xx : `array`
        Spatial axis. 
    density: `array`
        1D array of the density.
    velocity: `array`
        1D array of the velocity.
    temperature: `array`
        1D array of the temperature.
    pressure: `array`
        1D array of the pressure.
    energy: `array`
        1D array of the energy.

    Returns
    -------
        A 6 panel plot of the initial values for the density, velocity, 
        temperature, pressure and energy. 
    """
    dpi = 200 

    fig, ax = plt.subplots(3, 2, figsize=(12,12), dpi=dpi)

    ax[0,0].plot(xx, density, color='mediumpurple', label="Density")
    ax[0,0].set_xlabel("x", fontsize=14)
    ax[0,0].grid()
    ax[0,0].set_title("Density", fontsize=14)

    ax[0,1].plot(xx, velocity, color='green', label="Velocity")
    ax[0,1].set_xlabel("x", fontsize=14)
    ax[0,1].grid()
    ax[0,1].set_title("Velocity", fontsize=14)

    ax[1,0].plot(xx, temperature, color='orange', label="Temperature")
    ax[1,0].set_xlabel("x", fontsize=14)
    ax[1,0].grid()
    ax[1,0].set_title("Temperature",fontsize=14)

    ax[1,1].plot(xx, pressure, color='blue', label="Pressure")
    ax[1,1].set_xlabel("x", fontsize=14)
    ax[1,1].grid()
    ax[1,1].set_title("Pressure",fontsize=14)

    ax[2,0].plot(xx, energy, color='cyan', label="Energy")
    ax[2,0].set_xlabel("x", fontsize=14)
    ax[2,0].grid()
    ax[2,0].set_title("Energy",fontsize=14)

    ax[2,0].plot(xx, energy, color='cyan', label="Energy")
    ax[2,0].set_xlabel("x", fontsize=14)
    ax[2,0].grid()
    ax[2,0].set_title("Energy",fontsize=14)

    if source_function is not None: 
        ax[2,1].plot(xx, source_function, color='red', label="S(x,t)")
        ax[2,1].set_xlabel("x", fontsize=14)
        ax[2,1].grid()
        ax[2,1].set_title("Source Function",fontsize=14)
    else: 
        ax[2,1].plot(xx, density, color='mediumpurple', label="density")
        ax[2,1].plot(xx, velocity, color='green', label="velocity")
        ax[2,1].plot(xx, temperature, color='orange', label="temperature")
        ax[2,1].plot(xx, pressure, color='blue', linestyle='dotted', label="pressure")
        ax[2,1].plot(xx, energy, color='cyan', label="energy")
        ax[2,1].set_xlabel("x", fontsize=14)
        ax[2,1].grid()
        ax[2,1].set_title("All", fontsize=14)
        ax[2,1].legend(loc='lower right', fontsize=14)

    plt.tight_layout()


def plot_hd_evolv(xx, density, velocity, temperature, pressure, energy, \
                    source_function=None, plot=None): 
    """ 
    Function for plotting the values for the density, velocity, 
    temperature, pressure and energy, over time. 

    Parameters 
    ----------
    xx : `array`
        Spatial axis. 
    density: `array`
        1D array of the density.
    velocity: `array`
        1D array of the velocity.
    temperature: `array`
        1D array of the temperature.
    pressure: `array`
        1D array of the pressure.
    energy: `array`
        1D array of the energy.

    Returns
    -------
        A 6 panel plot or video of the initial values for the density, velocity, 
        temperature, pressure and energy. 
    """

    return 0