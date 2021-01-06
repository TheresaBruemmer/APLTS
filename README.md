# APLTS
Active-Plasma Lens + Thomson Scattering

APLTS is a tool to calculate Thomson Scattering (TS) from a Gaussian laser with an electron bunch focused by an Active Plasma Lens (APL)

Author: Dr. Theresa Karoline Brümmer

This is a collection of classes and modules for the following applications:

Calculation:
- TS:
  - Calculate photon number and expected bandwidth of a Thomson interaction (based on Rykovanov et al., JPB 2014)
  - Optimise a TS interaction for electron bunch and laser parameters
- APL focusing: 
  - Plan APL parameters for optimised TS interaction (e.g. as input for simulation)
  - Calculate APL paramaters and electron-bunch focusing
- APL+TS: Determine chromatic focussing effect on TS output (based on Brümmer et al., PRAB 2020, and Brümmer et al., to be submitted 2021)


Requirements: 
mpmath

=> source APLTS.modules file
