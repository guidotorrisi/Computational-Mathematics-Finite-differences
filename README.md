### **Aim and purpose**
Quick code to emulate [`approx.fprime`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html) in a simpler way using finite differences to approximates derivatives. 
Finite differences are the differences between consecutive values in a sequence, which can be used to approximate derivatives for a function at discrete points. The method is fundamental in numerical analysis for approximating derivatives in fields like computational fluid dynamics and solving differential equations numerically. This code has been written as part of the study plan of my Computational Mathematics course. I might add a study of the approximation in the future. 

### **Structure of the code**
The code defines a functions that can compute up to the second order approximation of the derivative of a function using either backward, forward or central differences. 

### **Parameters**

1.  _x_ = {float}; the discrete point at which to determine the derivative of _f_
3. _f_ = callable Function of which to estimate the derivatives of _x_ 
4. _order_ = {float}, optional; order of the derivative. It takes value in the interval [1,2]
5. h = {float}, optional; Increment to _x_ to use for determining the function.
6. method = {str}, optional: method of approximation to use for determining the function

### **Returns**

1.  fprime = {float, function}, result of the approximation
