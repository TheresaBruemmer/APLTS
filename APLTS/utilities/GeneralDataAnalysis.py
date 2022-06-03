import numpy as np
import math
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt

"""
---------------------------------------------------
General Data Analysis Tools: Functions and factors
---------------------------------------------------
"""
def test():
    print("It works")
RMStoFWHM = 2.*np.sqrt(2.*np.log(2.))
def find_FWHM(x_arr,data_arr,plot=True,out=True):
    if plot==True:
        plt.plot(x_arr,data_arr,'.')
    Imax=np.nanmax(data_arr)
    #at energy index                                                                       
    #print(data_arr)          
    E_Imax_index=np.nanargmax(data_arr)
    #print(E_Imax_index)
    #at energy                                                                                       
    E_Imax=x_arr[E_Imax_index]
    #print("E_{peak}="+str(E_Imax)+'keV')                                                            
    #Find FWHM                                                                                       
    #I=0.5Imax below E(Imax)                                                                         
    lower_cone=np.nanargmin(abs(data_arr[0:E_Imax_index]-Imax/2))
    #above                                                                                           
    upper_cone=np.nanargmin(abs(data_arr[E_Imax_index:]-Imax/2))+E_Imax_index
    FWHM=x_arr[upper_cone]-x_arr[lower_cone]
    FWHM_perc=FWHM/E_Imax
    if out==True:
        print("FWHM="+str(FWHM_perc*100)+" %")                                                   
    return E_Imax,FWHM,FWHM_perc
def find_FWHM_0center(x_arr,data_arr,plot=True,out=True):
    if plot==True:
        plt.plot(x_arr,data_arr,'.')
    Imax=np.nanmax(data_arr)
    #at energy index                                                                                 
    E_Imax_index=np.nanargmax(data_arr)
    #at energy                                                                                       
    E_Imax=x_arr[E_Imax_index]
    #print("E_{peak}="+str(E_Imax)+'keV')                                                            
    #Find FWHM                                                                                       
    #I=0.5Imax below E(Imax)                                                                         
    lower_cone=np.argmin(abs(data_arr[0:E_Imax_index]-Imax/2))
    #above                                                                                           
    upper_cone=np.argmin(abs(data_arr[E_Imax_index:]-Imax/2))+E_Imax_index
    FWHM_simple_cone=x_arr[upper_cone]-x_arr[lower_cone]
    if out==True:
        print("center="+str(E_Imax))
        print("low=" +str(x_arr[lower_cone]))

        print("high=" +str(x_arr[upper_cone]))

        print("FWHM="+str(FWHM_simple_cone))                                                   
    return E_Imax,FWHM_simple_cone
from scipy.signal import savgol_filter
#savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
def find_FWHM_savgol(x_arr,data_arr,plotData=True):
    if plotData==True:
        dataplot=plt.plot(x_arr,data_arr,'.')
        y_arr = savgol_filter(data_arr, 5, 2)#, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        color=dataplot[0].get_color()
        plt.plot(x_arr,y_arr,color=color)
    else:
        y_arr = savgol_filter(data_arr, 5, 2)#, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        plt.plot(x_arr,y_arr)
    Imax=np.nanmax(y_arr)
    #at energy index                                                                                 
    E_Imax_index=np.nanargmax(y_arr)
    #at energy                                                                                       
    E_Imax=x_arr[E_Imax_index]
    #print("E_{peak}="+str(E_Imax)+'keV')                                                            
    #Find FWHM                                                                                       
    #I=0.5Imax below E(Imax)                                                                         
    lower_cone=np.argmin(abs(data_arr[0:E_Imax_index]-Imax/2))
    #above                                                                                           
    upper_cone=np.argmin(abs(data_arr[E_Imax_index:]-Imax/2))+E_Imax_index
    FWHM_simple_cone=(x_arr[upper_cone]-x_arr[lower_cone])/E_Imax
    #print("FWHM="+str(FWHM_simple_cone*100)+" %")                                                   
    return E_Imax,FWHM_simple_cone
def find_FWHM_savgol_0center(x_arr,data_arr,plotData=False,poly_order=4):
    sorted_pairs = sorted((i,j) for i,j in zip(x_arr,data_arr))
    new_x, new_y = zip(*sorted_pairs)
    window_width=odd(int(len(new_x)/10))
    if plotData==True:
        dataplot=plt.plot(new_x,new_y,'.',alpha=0.2)
        y_arr = savgol_filter(new_y, window_width, poly_order)#, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        #color=dataplot[0].get_color()
        plt.plot(new_x,y_arr)#,color=color)
    else:
        y_arr = savgol_filter(new_y,window_width, poly_order)#, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        #plt.plot(x_arr,y_arr)
    ymax=np.nanmax(y_arr)
    level_mask = np.squeeze(np.where(y_arr > 0.5*ymax))
    left_i, right_i = level_mask[0], level_mask[-1]
    # Finesse the locations
    left_v_f = _lin_interp(new_x,y_arr, left_i, 0.5*ymax)
    right_v_f = _lin_interp(new_x,y_arr, right_i, 0.5*ymax)
    return new_x[np.nanargmax(y_arr)],abs(left_v_f - right_v_f)

def find_FWHM_savgol_0center_old(x_arr,data_arr,plotData=True):
    if plotData==True:
        dataplot=plt.plot(x_arr,data_arr,'.')
        y_arr = savgol_filter(data_arr, 5, 2)#, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        color=dataplot[0].get_color()
        plt.plot(x_arr,y_arr,color=color)
    else:
        y_arr = savgol_filter(data_arr, 5, 2)#, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        plt.plot(x_arr,y_arr)
    Imax=np.nanmax(y_arr)
    #at energy index                                                                                 
    E_Imax_index=np.nanargmax(y_arr)
    #at energy                                                                                       
    E_Imax=x_arr[E_Imax_index]
    #print("E_{peak}="+str(E_Imax)+'keV')                                                            
    #Find FWHM                                                                                       
    #I=0.5Imax below E(Imax)                                                                         
    lower_cone=np.argmin(abs(data_arr[0:E_Imax_index]-Imax/2))
    #above                                                                                           
    upper_cone=np.argmin(abs(data_arr[E_Imax_index:]-Imax/2))+E_Imax_index
    FWHM_simple_cone=(x_arr[upper_cone]-x_arr[lower_cone])
    #print("FWHM="+str(FWHM_simple_cone*100)+" %")                                                   
    return E_Imax,FWHM_simple_cone
def _nanargmin(arr):
    try:
       return np.nanargmin(arr)
    except ValueError:
       return np.nan
def normalize(x):
    return x/np.nanmax(x)
def weighted_avg_and_std(values, weights, return_average=False):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    if return_average==True:
        return average
    else:
        return math.sqrt(variance)#(average, math.sqrt(variance))
def std_rms(values,printing=False):
    n = len(values)
    mean=np.sum(values)/n
    if printing==True:
        print(mean)
    dev = (np.array([values[i] for i in range(0,n)])-mean)**2
    return np.sqrt(np.sum(dev)/n)
def std_rms_weighted(values, weight):
    mean = np.sum(values*weight)/np.sum(weight)
    n=len(values)
    dev = (np.array([values[i] for i in range(0,n)])-mean)**2
    return np.sqrt(np.sum(dev)/n)
def weighted_avg_and_std(values, weights, return_average=False):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    if return_average==True:
        return average
    else:
        return math.sqrt(variance)#(average, math.sqrt(variance))
def avg_and_std(values, return_average=False):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values)
    # Fast and numerically precise:
    variance = np.average((values-average)**2)
    if return_average==True:
        return average
    else:
        return math.sqrt(variance)#(average, math.sqrt(variance))    

def _lin_interp(x, y, i, half):
    """ Perform simple linear interpolation to fine tune widths """
    try:
        return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))
    except IndexError:
        return np.nan
from scipy.signal import savgol_filter

def binData(x,y,plotData=False):
    #np.histogram(x, bins=nbins, weights=y)
    nbins = int(np.sqrt(len(x)))
    #print(nbins)
    n, _ = np.histogram(x, bins=nbins)
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    if plotData==True:
        plt.plot(x, y, 'b.',alpha=0.3)
        plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='r-')
        #
        plt.show()
    x=(_[1:] + _[:-1])/2
    mean[np.where(np.isnan(mean))]=0
    return x, mean


def binData_old(gamma,Ngamma,num):
    gamma_bin=dict()
    Ngamma_bin=dict()
    Ne_bin=dict()

    #weight=10e-12/1.6022e-19/20000                                                                      
    #num=int(len(gamma)/200)
    gamma_bin=np.linspace(np.nanmin(gamma),np.nanmax(gamma),num)
    Ngamma_bin=np.zeros(num)#total number of emitted photons per gamma-bin (unknown by how many MPs)     
    Ne_bin=np.zeros(num)#number of macroparticles per gamma-bin                                          
    for i, gb in enumerate(gamma_bin):
        for j, g in enumerate(gamma):
            #if i==0:                                                                                    
            #numb+=1                                                                                     
            if i>=1:
                if g<=gb and g>gamma_bin[i-1]:
                    #print(j)                                                                            
                    Ne_bin[i]+=1
                    Ngamma_bin[i]+=Ngamma[j]
            else:
                #print(j)                                                                                
                if g<=gb:
                    Ne_bin[i]+=1
                    Ngamma_bin[i]+=Ngamma[j]
    return gamma_bin,Ngamma_bin,Ne_bin

def odd(x_):
    x=int(x_)
    return x - 1 if x % 2 == 0 else x
#print("FWHM="+str(FWHM_simple_cone*100)+" %")
    return E_Imax,FWHM_simple_cone



def rms(x):
    return np.sqrt(np.sum(x**2)/len(x))
