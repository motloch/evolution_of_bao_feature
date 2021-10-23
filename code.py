#!/usr/bin/env python3

"""
Video showing time evolution of an initially point-like (adiabatic) overdensity

Inspired by Fig. 1 from 'On the Robustness of the Acoustic Scale in the Low-Redshift
Clustering of Matter' by Daniel J. Eisenstein, Hee-Jong Seo and Martin White
https://arxiv.org/pdf/astro-ph/0604361.pdf
"""
from matplotlib import pyplot as plt
import numpy as np
import camb

#Set up CAMB cosmology (Planck 2018, TT,TE,EE+lowE+lensing+BAO)
#https://arxiv.org/pdf/1807.06209v4.pdf
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.66, ombh2=0.02242, omch2=0.11933, mnu=0.06, omk=0, tau=0.0561)
pars.InitPower.set_params(As=2.105e-9, ns=0.9665, r=0)

# Calculate background cosmology
results = camb.get_background(pars)

# Range of redshifts at which we plot evolution. 
# Start at the earliest time (highest redshift).
zs = np.arange(10000, 0, -10)
# Conformal times of those redshifts.
etas = [results.conformal_time(z) for z in zs]

# Grid of k's we will be integrating over
num_ks = 4000
kmin = 0.001
kmax = 2.*np.pi
ks = np.linspace(kmin, kmax, num_ks)

# Lower and upper limit for each grid interval; size of each interval
klow  = ks[:-1]
khigh = ks[1:]
dk = khigh - klow

# Physical radius at which we will evaluate the matter profile [in Mpc]
rs = np.arange(0.2, 200, 0.4)
# Sample cdm and baryons on a less fine grid
rs_cdm = np.arange(0.2, 200, 1)

# Get the transfer functions
data = camb.get_transfer_functions(pars)

# Get the mode evolution data. Indices:
#       ev[which_k, which_time, which_species]
ev = data.get_time_evolution(
        ks,
        etas,  #times at which we take a snapshot
        ['delta_cdm', 'delta_baryon', 'delta_photon', 'delta_nu'], #which overdensities
        lAccuracyBoost=8 #increase the calculation accuracy
      )

# Keeps track of the highest overdensity value [for y axis range]
max_values = np.zeros(len(zs))

def green(r, which_time, which_species):
    """
    Evaluate the Green's function at given time and distance from the origin. Depends on
    the species and is related to the precalculated transfer function T(k, t) through

        G(r,t) = int_0^infty k^2dk/(2pi^2) j0(kr)T(k,t)

    where j0(x) = sin(x)/x is the spherical Bessel function.

    CAMB uses normalization with respect to unit _primordial curvature_, so our ev
    corresponds to k^2 T(k,t)

    We only evaluate the integral between kmin and kmax

    Because of numerics, we will split the integral into a product of a sin function and a
    smooth envelope, 
        
        int k^2dk/(2pi^2) T(k)/(k*r) * sin(kr)

    On each k interval we then use a linear interpolation on the smooth envelope and use
    the analytic result for

       int_kmin^kmax (h0 + (h1-h0)*(k-kmin)/dk)*sin(k r)

    """

    #envelope
    integrand_divided_by_sin = 1./2./np.pi**2 * ev[:, which_time, which_species]/(r*ks)

    #split the k range into small bits, do linear interpolation on them
    h0 = integrand_divided_by_sin[:-1]
    h1 = integrand_divided_by_sin[1:]
    clow  = np.cos(klow * r)
    chigh = np.cos(khigh * r)
    slow  = np.sin(klow * r)
    shigh = np.sin(khigh * r)

    #Use the analytic formula on k intervals
    interval_contributions = (h0*dk*r*clow - h1*dk*r*chigh-(h0-h1)*(shigh-slow))/(dk*r**2)

    #Riemann sum
    return np.sum(interval_contributions)

"""
Main loop over redshifts
"""
for z_idx, z in enumerate(zs):
    plt.figure()

    #Green function multiplied by r^2 to agree with paper convention.
    #Photon and neutrino densities divided by 4/3 for the same reason
    gf_c   = [       r**2 * green(r, z_idx, 0) for r in rs_cdm]
    gf_b   = [       r**2 * green(r, z_idx, 1) for r in rs_cdm]
    gf_g   = [3/4. * r**2 * green(r, z_idx, 2) for r in rs]
    gf_v   = [3/4. * r**2 * green(r, z_idx, 3) for r in rs]

    #Store the information about the current maximal overdensity 
    current_max = np.max([np.max(gf_c), np.max(gf_b), np.max(gf_v), np.max(gf_g)])
    max_values[z_idx] = current_max
    #Make sure the y axis limit is not descreasing as it does not look nice
    current_max = np.max(max_values)

    #Plot the results and save
    plt.plot(rs_cdm, gf_c, label = 'cdm', color = 'k')
    plt.plot(rs_cdm, gf_b, label = 'b', color = 'b')
    plt.plot(rs, gf_v, label = '$\\nu$', color = 'g')
    plt.plot(rs, gf_g, label = '$\gamma$', color = 'r', ls = '--')
    plt.legend()
    plt.ylim([-0.05*current_max, 1.05*current_max])
    plt.title(f'Redshift {z}')
    plt.xlabel('$\mathrm{Radius\ [Mpc]}$')
    plt.ylabel('$\mathrm{Mass\ profile\ of\ perturbation}$')
    plt.savefig(f'img/plot_z_{z_idx:03d}')
    plt.close()
