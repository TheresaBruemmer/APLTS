import numpy as np
from numpy import sqrt,sin,cos,exp,log
"""
---------------------------------------------------
Fitting Functions
---------------------------------------------------
""" 
def Lorentzian(x,mu,gammaL):
    #This gammaL is the HWHM value, not electron energy or CSP
    """ Return Lorentzian line shape at mu,x with HWHM gammaL """
    return gammaL / np.pi / ((x-mu)**2 + gammaL**2)
def Lorentzian_amp(x,mu,gammaL,amp):
    #This gammaL is the HWHM value, not electron energy or CSP
    """ Return Lorentzian line shape at mu,x with HWHM gammaL and amplitue amp"""
    return gammaL / np.pi / ((x-mu)**2 + gammaL**2)/(1/np.pi/gammaL)*amp
def Lorentzian_amp2(x,mu,gammaL,amp):
    #This gammaL is the HWHM value, not electron energy or CSP
    """ Return Lorentzian line shape at mu,x with HWHM gammaL """
    return gammaL / np.pi / ((x-mu)**4 + gammaL**4)/(1/np.pi/gammaL)*amp


def gaussian(x,mu,sigma):
    return amp/sqrt(2*np.pi*sigma**2)*exp(-(x-mu)**2/(2*sigma**2))

def gaussian_amp(x,mu,sigma,amp):
    return amp*exp(-(x-mu)**2/(2*sigma**2))

def double_gaussian_amp(x,mu1,sigma1,amp1,mu2,sigma2,amp2):
    return amp1*exp(-(x-mu1)**2/(2*sigma1**2))+amp2*exp(-(x-mu2)**2/(2*sigma2**2))

def triple_gaussian_amp(x,mu1,sigma1,amp1,mu2,sigma2,amp2,mu3,sigma3,amp3):
    return amp1*exp(-(x-mu1)**2/(2*sigma1**2))+amp2*exp(-(x-mu2)**2/(2*sigma2**2))+amp3*exp(-(x-mu3)**2/(2*sigma3**2))


#Fit the Lorentzian and take FWHM = 2gamma
def fit_line(x,a,b):
    return a*x+b
def fit_lineplusLorentzian(x,a,b,mu,gammaL,amp):
    return amp*(fit_line(x,a,b) + Lorentzian(x,mu,gammaL))
def fit_function_absLinear(x,a,b,c):
    return a*(abs(x-b))+c
def fit_function_quad(x,a,b,c):
    return a+b*x+c*x**2
def fit_function_cube(x,a,b,c,d):
    return a+b*x+c*x**2+d*x**3
def fit_function_quart(x,a,b,c,d,e):
    return a+b*x+c*x**2+d*x**3+e*x**4
def fit_function_poly6(x,a,b,c,d,e,f,g):
    return a+b*x+c*x**2+d*x**3+e*x**4+f*x**5+g*x**6
def fit_function_sqrt(x,a,b,c,d):
    return a*np.sqrt(b+c*(x+d)**2)
def fit_function_sqrt_asymm(x,a,b,c,d,e):
    return a*np.sqrt(b+c*(x-d)**2)+e*(x-d)#+f
