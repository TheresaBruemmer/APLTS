"""
This module defines the Active plasma lens focusing setup which allows to calculate the bunch parameters as it propagates through the setup
Required matrix formalism functions are defined. 
"""
#import sys
#sys.path.append("/p/project/plasmabbq/tbruemmer/software/APLTS/")

from APLTS import ElectronBunch
import numpy as np
from numpy import sqrt,pi
import APLTS.utilities.physical_constants as const
mu_0=const.mu0
e=const.e_charge
m_0=const.m_e
c=const.c_light

#Transformation matrices
def M_freedrift(s):
    """
    Free drift matrix
    """
    return np.array([[1,s],[0,1]])
def M_thinlens(r_0,I_0,gamma,L):
    """
    Lens matrix for thin-lens approximation
    """
    k_temp=APL_setup.k_calc(r_0,I_0,gamma)
    f_temp=1/(k_temp*L)
    return np.array([[1,0],[-1/f_temp,1]])
#define initial Twiss parameters
def Twiss(epsilon_n,gammae,sigmar):
    """
    Returns Courant-Snyder (Twiss) parameters of an electron bunch in its focus
    input: 
    epsilon_n: normalised bunch emittance (m rad), float
    gammae: electron bunch central energy (dimensionless), float
    sigmar: electron bunch waist (m),float
    output:
    alpha, beta, gamma: Counrant-Snyder parameters, array of floats
    """
    epsilon=epsilon_n/gammae
    alpha=0  # is zero in focus
    beta=sigmar**2/epsilon
    gammaT=(1+alpha**2)/beta
    return np.array([beta,alpha,gammaT])
#propagate the Twiss parameters
def propagateTwiss(Twiss_arr,Matrix):
    """
    Propagates Courant-Snyder parameters with given propagation matrix
    input: 
    Twiss_arr: beta, alpha, gamma: Courant-Snyder parameters, array of floats
    Matrix: Propagation matrix, e.g. M_freedrift, M_thinlens, ...
    output:
    beta2,alpha2,gamma2: Courant-Snyder parameters, array of floats
    """
    beta1=Twiss_arr[0]
    alpha1=Twiss_arr[1]
    gamma1=Twiss_arr[2]
    M11=Matrix[0,0]
    M12=Matrix[0,1]
    M21=Matrix[1,0]
    M22=Matrix[1,1]
    beta2=M11**2*beta1-2*M11*M12*alpha1+M12**2*gamma1
    alpha2=-1*M21*M11*beta1+(1+2*M12*M21)*alpha1-M12*M22*gamma1
    gamma2=M21**2*beta1-2*M22*M21*alpha1+M22**2*gamma1
    return np.array([beta2,alpha2,gamma2])
def gammae_foc(f_0,I_0,L,r_0,z_0):
    '''
    This function is an approximation for the focused gammae at given lens parameters. 
    The approximated formula was calculated via Mathematica
    input:
    f_0: focal length (m), float
    I_0: APL current (A), float
    L: APL length (m), float
    r_0: APL radius (m), float
    z_0: distance between plasma acceleration stage and APL (m), float
    output: 
    focused_gammae: electron energy focused for given input parameters (dimensionless)
    '''
    k_0=e/(m_0*c)*mu_0/(2.*pi)
    focused_gammae = (k_0*I_0*L)/(12*r_0**2*(f_0+L+z_0))*(L**2+3*L*z_0+3*f_0*(L+2*z_0)+sqrt(L**2*(L+3*z_0)**2+6*f_0*L*(L**2+L*z_0+2*z_0**2)+3*f_0**2*(3*L**2+4*L*z_0+12*z_0**2)))
    return focused_gammae
def FindAPLConfig(gammae,sigmar_i,eps_n,sigmar_f,L_APL,r_APL,z_0,I0_min,I0_max,Nrun):
    '''
    returns the APL config for a given target electron energy focus and APL setup
    #
    We want to do Nrun runs to find the optimum.
    Each time we close in more on the opt values
    #
    Input:
    gammae: Target electron energy
    sigmar_i
    eps_n
    sigmar_f: target focal waist
    L_APL,r_APL,z_0: lenth, radius and z position of APL
    I0_min,I0_max: Range of initial APL current array
    Nrun: number of opt loops
    #
    Output:
    I_0: APL current for target focus
    z_F: resulting focal plane
    check_Waist: calculates the waist from optI0 and at zF, and should be compared to target sigmar_f from input
    '''
    for opt_run in range(0,Nrun+1):
        I0_arr = np.linspace(I0_min,I0_max,10)
        delta_I0=I0_arr[1]-I0_arr[0]
        instances=APL_setup(L=L_APL,I_0=I0_arr,z_0=z_0,r_0=r_APL,gammae=gammae,eps_n=eps_n,sigmar_i=sigmar_i)
        match_arg = np.nanargmin(abs(instances.focalWaist()-sigmar_f))
        I0=I0_arr[match_arg]
        zF=instances.focalPlane()[match_arg]
        check_Waist=instances.focalWaist()[match_arg]
        I0_min=I0-2*delta_I0
        I0_max=I0+2*delta_I0
    return I0,zF,check_Waist

############################################################################

#class APL:
#Lens Matrices

class APL_setup():
    '''
    Defines the Active Plasma Lens focusing setup.
    Calculates bunch parameters in the setup. 
    Electron bunch starts in focus at end of plasma acceleration stage, 
    free drift to APL, propagation to APL, free drift to focus
    '''
    def __init__(self,L=0.1,I_0=500,z_0=5e-2,r_0=1e-3,gammae=127,eps_n=1.0e-6,sigmar_i=1.0e-6):
        """
        Initialise parameters of the focusing setup:
        ------------------
        APL parameters:
        L:   float
             length of the APL (m)
        I_0: float
             APL current (A)
        z_0: float
             distance between plasma acceleration stage and APL (m)
        r_0: float
             radius of the APL capillary (m)
        Bunch parameters:
        gammae: float
             bunch energy 
        eps_n: float
             normalised bunch emittance (m rad)
        sigmar_i: float
             initial bunch waist (m)
        alpha, beta, gamma: floats
             initial Courant-Snyder parameters
        B: float
             APL maximum magnetic field (T)
        """
        self.L = L
        self.I_0 = I_0
        self.z_0 = z_0
        self.r_0 = r_0
        self.gammae=gammae
        self.eps_n = eps_n
        self.sigmar_i = sigmar_i
        self.alpha = 0
        self.beta,self.alpha,self.gamma = Twiss(self.eps_n,self.gammae,self.sigmar_i)
        self.B = 4*np.pi*1e-7*self.I_0*self.r_0/2/(np.pi*self.r_0**2)
    def k_calc(self):
        '''
        calculate lens focusing strength
        '''
        return e/(m_0*c)*mu_0*self.I_0/(2.*pi*self.r_0**2)/self.gammae
    def M_lens(self):
        '''
        Lens focusing matrix
        '''
        k_temp=APL_setup.k_calc(self)
        phi=self.L*np.sqrt(np.abs(k_temp))
        M11=np.cos(phi)
        M12=1./np.sqrt(np.abs(k_temp))*np.sin(phi)
        M21=-np.sqrt(np.abs(k_temp))*np.sin(phi)
        M22=np.cos(phi)
        return np.array([[M11,M12],[M21,M22]])
    def propagate(self):
        '''
        Propagate a beam to the APL and through the APL setup
        Returns
        init: initial Courant-Snyder parameters (CSPs) at end of plasma acceleration stage (focused bunch)
        FD_: CSPs after the free drift between plasma acc. stage and APL
        APL_: CSPs at end of APL
        sigma_APL_entry: RMS bunch size at beginning of APL (=end of free drift)
        sigma_f: RMS bunch size in bunch focus
        f: focal length, i.e. length from end of APL to bunch focus
        '''
        init=Twiss(self.eps_n,self.gammae,self.sigmar_i)
        FD_=propagateTwiss(init,M_freedrift(self.z_0))
        sigma_APL_entry = np.sqrt(FD_[0]*self.eps_n/self.gammae)
        APL_=propagateTwiss(FD_,APL_setup.M_lens(self))
        f=APL_[1]*APL_[0]/(1+APL_[1]**2)
        sigma_f = np.sqrt((APL_[0]-2*f*APL_[1]+f**2*APL_[2])*self.eps_n/self.gammae)
        return init,FD_,sigma_APL_entry,APL_,f,sigma_f
    def focalLength(self):
        '''
        returns APL focal length, i.e. length from end of APL to bunch focus
        NAN if focus is negative/virtual
        either float or array of floats
        '''
        init,FD_,sigma_APL_entry,APL_,f,sigma_f = APL_setup.propagate(self)
        if np.ndim(f)==0:
            if f>0 and sigma_APL_entry>sigma_f:
                focal_length = f
            else:
                focal_length=np.NAN
        else:
            focal_length = np.zeros(np.shape(sigma_f))
            focal_length[:]=np.NAN
            focal_length[(f>0) & (sigma_APL_entry>sigma_f)]=f[(f>0) & (sigma_APL_entry>sigma_f)]
        return focal_length
    def focalPlane(self):
        '''
        returns focal plane, i.e. length between plasma acc. stage and bunch focus
        '''
        return APL_setup.focalLength(self)+self.z_0+self.L
    def focusgammae(self,f_0):
        '''
        returns electron energy focused at fixed focal length f_0 for the given APLTS setup
        f_0: focal length, i.e. length from end of APL to bunch focus
        '''
        k_0=e/(m_0*c)*mu_0/(2.*pi)
        return (k_0*self.I_0*self.L)/(12*self.r_0**2*(f_0+self.L+self.z_0))*(self.L**2+3*self.L*self.z_0+3*f_0*(self.L+2*self.z_0)+sqrt(self.L**2*(self.L+3*self.z_0)**2+6*f_0*self.L*(self.L**2+self.L*self.z_0+2*self.z_0**2)+3*f_0**2*(3*self.L**2+4*self.L*self.z_0+12*self.z_0**2)))   
    def focalWaist(self):
        '''
        returns bunch focal waist (RMS)
        '''
        init,FD_,sigma_APL_entry,APL_,f,sigma_f = APL_setup.propagate(self)
        if np.ndim(sigma_f)==0:
            if f>0 and sigma_APL_entry>sigma_f:
                focal_waist = sigma_f
            else:
                focal_waist = np.NAN
        else:
            focal_waist = np.zeros(np.shape(sigma_f))
            focal_waist[:]=np.NAN
            focal_waist[(f>0) & (sigma_APL_entry>sigma_f)]=sigma_f[(f>0) & (sigma_APL_entry>sigma_f)]
        return focal_waist
    def focalDivergence(self):
        '''
        returns bunch focal divergence (RMS)
        '''
        return self.eps_n/self.gammae/APL_setup.focalWaist(self)
    def waist_freeDrift(self,s):
        '''
        returns bunch waist after propagation distance s behind the APL
        s: distance from APL exit to point of interest
        '''
        APL_=APL_setup.propagate(self)[3]
        FD_=propagateTwiss(APL_,M_freedrift(s))
        return np.sqrt(FD_[0]*self.eps_n/self.gammae)
    def return_focusedBunch(self):
        """
        Creates object of type ElectronBunch as obtained from APL focusing 
        """
        APL_ElectronBunch=ElectronBunch.ElectronBunch(self.gammae,self.eps_n,self.focalWaist())
        return APL_ElectronBunch
