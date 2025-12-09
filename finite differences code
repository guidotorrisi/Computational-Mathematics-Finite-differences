import numpy as np

def approxfinite_dif(f, x, order=1, h=1e-5, method="central"):
    
    if order == 1 and method == "forward":
        fprime = (f(x+h) - f(x))/h
    elif order == 1 and method == "backward":
        fprime = (f(x) - f(x-h))/h
    elif order == 1 and method == "central":
        fprime = (f(x+h) - f(x-h))/(2*h)
    elif order == 2 and method == "forward":
        fprime = (-3*f(x)+4*f(x+h)-f(x+2*h))/(2*h)
    elif order == 2 and method == "backward":
        fprime = (3*f(x)-4*f(x-h)+ f(x-2*h))/(2*h)
    elif order == 2 and method == "central":
        fprime = (f(x+h)-2*f(x)+f(x+h))/(h**2)
    else:
        raise ValueError("Check the instructions")
    return fprime

def approx_pderiv(f, x, i, h=1e-5)
    x= np.array(x, dtype=float)
    x_forward = x.copy()
    x_backward = x.copy()
    x_forward[i] += h
    x_backward[i] -= h
    fprime = (f(x_forward) - f(x_backward)) / (2*h)
    
    return fprime

