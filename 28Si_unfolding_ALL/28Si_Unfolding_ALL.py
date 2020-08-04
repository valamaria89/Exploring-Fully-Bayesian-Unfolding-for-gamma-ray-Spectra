import ompy as om 
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as mc
import scipy as sp
from theano import tensor as T
import theano; theano.gof.cc.get_module_cache().clear()
import math as m
import pandas as pd 
from scipy.stats import gamma, norm, lognorm
import fbu
import shutil  
import os


# The code used to unfolded the raw matrix of 28Si:

raw = om.Matrix(path="/home/vala/Downloads/h_Ex_Eg_improved_bgsubtr_noneg_28Si.m")



raw.cut_diagonal(E1=(1800, 500), E2=(10500, 10000))

raw.cut('Ex', 0, 10000)
raw.cut('Eg', 0, 10000)
raw.rebin(axis= "Eg", factor=6)
raw.rebin(axis= "Ex", factor=1)
raw.plot()


Ex = raw.Ex
Eg = raw.Eg

# Using line_mask from the OMpy library to get the index of the diagonal that is cut in the cut_diagonal function:
a = raw.line_mask(E1=(1800, 500), E2=(10500, 10000))
index_diag = []
for i in range(a.shape[0]-1):
    lst =np.where(a[i] == True)[0]
    index_diag.append(lst[0]) #list of index of the diagonal 



def unfolding_FBU(thres, cut_Ex, samples, tuning):
    
    for i in range(cut_Ex, len(Ex)):
        
        print(i)
        print(Ex[i])
        raw_py, E = raw.projection(axis="Eg", Emin=Ex[i], Emax=Ex[i])

        # In the upper part of the spectrum Ex = Eg = 10000 keV the counts where not set to zero under the diagonal
        # So this if statement was made, but was not needed since I only unfolded up to 8000 keV
        if (i < len(index_diag)):
            raw_bay = raw_py[thres:index_diag[i]]
            energy = Eg[thres: index_diag[i]]

        else:
            raw_bay = raw_py[thres:]
            energy = Eg[thres:]    

        # Below is the code for the unfolding with the response, priors, tuning steps and sampling initiated. 
        # It is done in the same way as for the case of unfolding the first excited state of 28Si. 
        # These steps are thoroughly explained in the jupyter files where the first excited state is unfolded.  
        print(len(raw_bay))
        print(len(energy))
        fwhm_abs = 30.0
        folderpath = "/home/vala/Documents/Master/MachineLearning/ompy/OCL_response_functions/oscar2017_scale1.15"
        response = om.Response(folderpath)

        R_bay, R_tab_bay = response.interpolate(energy, fwhm_abs=fwhm_abs, return_table=True)
        R_bay_unf, R_tab_bay_unf = response.interpolate(energy, fwhm_abs=fwhm_abs/10, return_table=True)

        eff =0.5*R_tab_bay["eff_tot"].values

        pFE = R_tab_bay['pFE'].values

        resp =R_bay.values*eff[:,np.newaxis]

        prior_l = np.zeros(len(raw_bay))
        prior_u = raw_bay/(eff*pFE)

        # After Ex[100] = 3127 there is no need to add a constant to the prior. The constant is only needed when the re-distribution of counts
        # are going mainly to the 2^+ transition (1779) keV (first excited state).
        if (i < 100):
            max_peak = np.max(raw_bay)

            prior_u += max_peak

        # The upper bound cannot be zero (i.e cannot be equal to the lower bound), this is to prevent this. 
        # if lower = upper PyMC3 will execute the error: Bad initial energy     
        prior_u = np.where(raw_bay==0, 1e-4, prior_u)
            
        myfbu = fbu.PyFBU()
        myfbu.response = resp

        myfbu.data =raw_bay
        myfbu.prior = 'Uniform'
        myfbu.upper = prior_u   
        myfbu.lower = prior_l

        myfbu.nTune = tuning
        myfbu.nMCMC = samples
        myfbu.nCores = 2
        myfbu.run()

        trace = myfbu.trace
        
        # Saving the trace containing the posterior distribution
        cols = []
        for tr in range(len(trace)):
            name = 'truth%d'%i
    
            cols.append(name)

        df = pd.DataFrame(dict(zip(cols, trace)))

        file_name_trace = 'Traces/unfolding_Si28_trace_%d.csv'%Ex[i]
        df.to_csv(file_name_trace, index=False) 

        # Calculating the HPD interval and its mean
        hpd = np.zeros((len(trace),2))
        mean_inter = np.zeros(len(hpd))
        for sampl in range(len(trace)):
            hpd[sampl:] = mc.stats.hpd(trace[sampl], credible_interval=0.68)
            mean_inter[sampl] = (hpd[sampl, 0]+hpd[sampl, 1])/2

        # Saving the HPD and mean to file 
        df1 = pd.DataFrame({'est. mean': mean_inter,
                           'HPD min': hpd[:,0],
                           'HPD max': hpd[:,1]})

        file_name_hpd = 'hpd_intervals/unfolding_Si28_credint_68_%d.csv'%Ex[i] 
        df1.to_csv(file_name_hpd, index=False)
 

    
# initiating the code to unfold the raw matrix with FBU 
unfolding_FBU(5, 20, 3000, 1000)
