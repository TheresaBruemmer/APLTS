"""
This module contains Thomson Scattering class, loading laser and ebunch
"""
#import sys
#sys.path.append("/p/project/plasmabbq/tbruemmer/software/APLTS/")
from APLTS import ThomsonScatteringTools as TST
import APLTS.utilities.physical_constants as constants
import numpy as np
from numpy import sqrt,cos,sin,tan,exp, pi, log
from mpmath import mp
import math
mp.dps = 50

class ThomsonScattering():
    """
    Defines the Thomson interaction
    """
    def __init__(self,laser,bunch):
        """
        Initiate the laser and bunch from instances of classes ElectronBunch and Laser
        """
        self.laser=laser
        self.bunch=bunch
        #calling params of laser or bunch class via laser._wavelength etc.
        #check for all params:
        for item in vars(self.laser):
            if vars(self.laser)[item]==None:
                print(str(item)+" is None. It should be set in laser instance.")
        for item in vars(self.bunch):
            if vars(self.bunch)[item]==None:
                print(str(item)+" is None. It should be set in bunch instance.")
                if str(item)=="_Q":
                    self.bunch._Q=10.e-12
                    print("Is set to "+str(self.bunch._Q)+" C")
    def Photons_tot(self):
        """
        Required parameters:
        #laser
        self.laser._wavelength,
        self.laser._Ep,
        self.laser._tau_FWHM,
        self.laser._w0,
        #bunch
        self.bunch._gammae,
        self.bunch._Q,
        self.bunch._sigma_r,
        self.bunch._sigma_l,
        self.bunch._emittance
        """
        N=TST.Photons_tot(self.laser._wavelength,self.laser._Ep,self.laser._tau_FWHM,self.laser._w0,self.bunch._gammae,self.bunch._Q,self.bunch._sigma_r,self.bunch._sigma_l,self.bunch._emittance)
        return N

    def Photons_cone(self,collimation_angle,BWlim=0,coll_angle_steps=50):
        """
        Required parameters:
        #laser
        self.laser._wavelength,
        self.laser._Ep,
        self.laser._tau_FWHM,
        self.laser._w0,
        #bunch
        self.bunch._gammae,
        self.bunch._Q,
        self.bunch._sigma_r,
        self.bunch._sigma_div,
        self.bunch._sigma_l,
        self.bunch._emittance
        #detector
        collimation_angle,
        coll_angle_steps
        """
        theta=collimation_angle
        thetasteps=coll_angle_steps
        N_cone=TST.Photons_cone(self.laser._wavelength,self.laser._Ep,self.laser._tau_FWHM,self.laser._w0,self.bunch._gammae,self.bunch._Q,self.bunch._sigma_r,self.bunch._sigma_div,self.bunch._sigma_l,self.bunch._emittance,theta,BWlim=BWlim,angle_steps=thetasteps)[0]
        return N_cone
    def Bandwidth_collimation(self,collimation_angle):
        return TST.BW_collimation(self.bunch._gammae,collimation_angle)
    def Bandwidth_bunchDivergence(self):
        return TST.BW_divergence(self.bunch._gammae,self.bunch._sigma_div)
    def Bandwidth_a0(self):
        return BW_a0(self.laser._a0)
    def Bandwidth_bunchEnergySpread(self):
        if self.bunch._Delta_gammae==None:
            print("Input Error: Electron bunch energy spread is not defined. Bandwidth cannot be calculated.")
        else:
            return TST.BW_ebunchEnergySpread(self.bunch._Delta_gammae)
    def Bandwidth_laserBandwidth(self):
        if self.laser.Delta_lambda==None:
            print("Input Error: Laser bandwidth is not defined. Bandwidth cannot be calculated.")
        else:
            return BW_laserBandwidth(self.laser.Delta_lambda)
    def Total_Bandwidth(self,collimation_angle):
        """
        Gaussian addition of bandwidth contributions (Rykovanov)
        -------------------
        Required parameters:
        
        """
        return sqrt(self.Bandwidth_collimation()**2+self.Bandwidth_bunchDivergence()**2+self.Bandwidth_a0()**2+self.Bandwidth_bunchEnergySpread()**2+self.Bandwidth_laserBandwidth()**2)
