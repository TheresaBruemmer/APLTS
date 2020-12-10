"""
This module contains the electron bunch
"""
import sys
sys.path.append("/beegfs/desy/group/fla/ICS/tools/ModulesAndClasses/")
import physical_constants as constants

class ElectronBunch():
    """
    Defines an electron bunch (in focus)
    """
    def __init__(self,_gammae=100.0,_emittance=1.0e-6,_sigma_r=1.0e-6,_Q=None,_sigma_l=1.5e-6,_sigma_div=None,_Delta_gammae=None):
        """
        Initialise electron-bunch parameters:
        ----------------------------
        _gammae: electron energy (dimensionless), float
        _emittance: laser pulse energy (J), float
        _sigma_r: RMS bunch waist (m), float
        _Q: bunch charge (C), float
        _sigma_l: longitudinal RMS bunch length (m), float
        """
        self._gammae=_gammae
        self._emittance=_emittance
        self._sigma_r=_sigma_r
        self._Q=_Q
        self._sigma_l=_sigma_l
        self._sigma_div=_sigma_div
        if self._sigma_div==None:
            print("No divergence defined, focal divergence calculated")
            self.set_focalDivergence()
    def Ekin(self):
        return self.gammae*constants.E0
    def set_focalDivergence(self):
        self._sigma_div=self._emittance/self._gammae/self._sigma_r
