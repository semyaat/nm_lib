import numpy as np 
from nm_lib import nm_lib as nm

x0 = -2.6 
xf =  2.6 

nint = 256
nump = nint - 1
xx = np.arange(nump)/(nump-1.0) * (xf - x0) + x0


def u(x, t=False): 
    """ 
    Solves the initial condition for u(x, t=t0) from equation 2). 
    Valid when a = const 

    Parameters
    ----------
    x : `array`
       Spatial axis. 
    t : `bool`
       
    Returns
    ------- 
    `array`
        Initial condition of u(x, t=t0)
    """
    return np.cos(6*np.pi*x / 5)**2 / (np.cosh(5*x**3))

def u_exact(x, t, a): 
    r""" 
    Solves the exact solution of u(x,t) when a=const, u(x,t) = u0(x-at)

    Requires
    ----------
    Some function to define u(x,t=t0)

    Parameters
    ----------
    x : `array`
       Spatial axis. 
    t : `array`
        Time axis. 
    a : `float` or `array`
        Either constant, or array which multiply the right hand side of the Burger's eq.

    Returns
    ------- 
    uu : `array`
        Spatial and time evolution of u(x,t) = u0(x-at)
    """
    
    xx = np.zeros((len(x), len(t)))
    x_tmp = np.zeros((len(x), len(t)))

    for i in range(0, len(t)): 
        # Sets boundaries using mod (bring solution back into domain)
        # Taking mod of the length of the grid (2.6) returns the remainder of the divide 
        # -> The right column and row         
        x_tmp[:,i] = ((x - a*t[i]) - x0) % (xf - x0) + x0 

        # Insert into u0
        xx[:,i]    = u(x_tmp[:,i]) # u0(x-at)
    return xx

def test_ex_2b():
    a = -1
    nt = 200

    t, unnt = nm.evolv_adv_burgers(xx, u(xx), nt, a)
    unnt_exact = u_exact(xx, t, a)

    # Assert that diff between analytical and numerical are good 
    tol = 1e-9
    assert np.abs(np.sum(unnt) - np.sum(unnt_exact)) < tol 
