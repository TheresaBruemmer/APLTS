"""
ThomsonScattering class tester
"""
import sys
sys.path.append("/p/project/plasmabbq/tbruemmer/software/APLTS/")
from APLTS.Laser import Laser
from APLTS.ElectronBunch import ElectronBunch as ebunch
from APLTS.ThomsonScattering import ThomsonScattering as TS
#
myLaser=Laser(_a0=0.55,_tau_FWHM=0.5e-12)
myBunch=ebunch(_Q=10e-12)
myThomson=TS(myLaser,myBunch)
#print(vars(myLaser))
#print(vars(myBunch))
#print(vars(myThomson))
#print(vars(myThomson.laser))
#print(vars(myThomson.bunch))
print("{:.2e}".format(myThomson.Photons_tot()))
print("{:.2e}".format(myThomson.Photons_cone(0.5e-3)))
