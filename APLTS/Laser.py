"""
This module contains Gaussian laser tools including a laser class
"""
#import sys
#sys.path.append("/p/project/plasmabbq/tbruemmer/software/APLTS/utilities/")
import APLTS.utilities.physical_constants as constants
import numpy as np
from numpy import sqrt,cos,sin,tan,exp, pi, log

def Ep_calc(wavelength, a0, w0, tau_fwhm):
    """
    Calculates laser pulse energy for a Gaussian laser
    input:
    wavelength: laser wavelength (m), float
    a0: peak amplitude, i.e. laser strength parameter (dimensionless), float
    w0: laser focal waist (m), float
    tau_fwhm: laser pulse duration (s, FWHM), float
    output:
    Ep: laser pulse energy (J), float
    """
    tau=tau_fwhm/(sqrt(2.*np.log(2)))
    return (constants.e_charge/(2*pi*constants.m_e*constants.c_light**2))**2*1/(sqrt(pi/2)*pi/4*constants.eps_0*constants.c_light)*wavelength**2*tau/w0**2/a0**2
def a0_calc(wavelength,Ep,w0,tau_fwhm):
    """
    Calculates peak laser strength parameter for a Gaussian laser
    input:
    wavelength: laser wavelength (m), float
    Ep: laser pulse energy (J), float
    w0: laser focal waist (m), float
    tau_fwhm: laser pulse duration (s, FWHM), float
    output:
    a0: peak amplitude, i.e. laser strength parameter (dimensionless), float
    """
    tau=tau_fwhm/(sqrt(2.*np.log(2)))
    return wavelength*constants.e_charge/(2*constants.m_e*constants.c_light**2*pi)*sqrt(Ep/(sqrt(pi/2)*pi/4*constants.eps_0*constants.c_light*w0**2*tau))
def tau_calc(wavelength,Ep,a0, w0):
    """
    Calculates fwhm pulse duration for a Gaussian laser
    input:
    wavelength: laser wavelength (m), float
    Ep: laser pulse energy (J), float
    w0: laser focal waist (m), float
    a0: peak amplitude, i.e. laser strength parameter (dimensionless), float
    output:
    tau_fwhm: laser pulse duration (s, FWHM), float
    """
    tau=(constants.e_charge/(2*pi*constants.m_e*constants.c_light**2))**2*1/(sqrt(pi/2)*pi/4*constants.eps_0*constants.c_light)*wavelength**2*Ep/w0**2/a0**2
    tau_fwhm=tau*(sqrt(2.*np.log(2)))
    return tau_fwhm
def w0_calc(wavelength,Ep,a0,tau_fwhm):
    """
    Calculates waist of a Gaussian laser
    input:
    wavelength: laser wavelength (m), float
    Ep: laser pulse energy (J), float
    a0: peak amplitude, i.e. laser strength parameter (dimensionless), float
    tau_fwhm: laser pulse duration (s, FWHM), float
    output:
    w0: laser focal waist (m), float
    """
    tau=tau_fwhm/(sqrt(2.*np.log(2)))
    return wavelength*constants.e_charge/(2*constants.m_e*constants.c_light**2*pi)*sqrt(Ep/(sqrt(pi/2)*pi/4*constants.eps_0*constants.c_light*a0**2*tau))


class Laser():
    """
    Defines a Gaussian laser
    Calculates defining parameters
    """
    def __init__(self,_wavelength=800.0e-9,_Ep=500.0e-3,_a0=None,_w0=None,_tau_FWHM=None,_Delta_lambda=None):
        """
        Initialise laser parameters:
        ----------------------------
        _wavelength: laser wavelength (m), float
        _Ep: laser pulse energy (J), float
        _a0: peak amplitude, i.e. laser strength parameter (dimensionless), float
        _w0: laser focal waist (m), float
        _tau_fwhm: laser pulse duration (s, FWHM), float
        _Delta_lambda: laser bandwidth (m), float
        """
        self._wavelength = _wavelength
        self._Ep=_Ep
        self._a0=_a0
        self._w0=_w0
        self._tau_FWHM=_tau_FWHM
        self._Delta_lambda=_Delta_lambda
        if self._a0 and self._w0 and self._tau_FWHM:
            print("Too many laser input variables, overwrite _tau_FWHM")
            self.set_a0w0(_a0,_w0)
            print("new tau_FWHM = " +str(self._tau_FWHM))
        else:
            if _a0 and _w0:
                self.set_a0w0(_a0,_w0)
            elif _a0 and _tau_FWHM:
                self.set_a0tau(_a0,_tau_FWHM)
            elif _w0 and _tau_FWHM:
                self.set_w0tau(_w0,_tau_FWHM)
        if self._Delta_lambda==None:
            print("No laser bandwidth defined, Fourier-limited bandwidth calculated")
            self.set_FourierLimitedBandwidth()
    def set_a0w0(self,a0,w0):
        self._a0=a0
        self._w0=w0
        self._tau_FWHM=tau_calc(self._wavelength,self._Ep,self._a0, self._w0)
    def set_a0tau(self,a0,tau_FWHM):
        self._a0=a0
        self._tau_FWHM=tau_FWHM
        self._w0=w0_calc(self._wavelength,self._Ep,self._a0,self._tau_FWHM)
    def set_w0tau(self,w0,tau_FWHM):
        self._w0=w0
        self._tau_FWHM=tau_FWHM
        self._a0=a0_calc(self._wavelength,self._Ep,self._w0,self._tau_FWHM)
    def set_FourierLimitedBandwidth(self):
        #time-bandwidth product for Gaussian laser
        tbp_Gauss=0.44#tau_fwmh*df
        f=constants.c_light/self._wavelength
        df=tbp_Gauss/self._tau_FWHM
        #df/f=dl/l
        self._Delta_lambda=df/f*self._wavelength
