# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:04:11 2023

@author: 6xk
"""

import brainpy
import pandas as pd
import numpy as np
import math
from scipy.stats import binom
from scipy.optimize import least_squares
from scipy.signal import deconvolve
import matplotlib.pyplot as plt
import re
from sklearn.metrics import mean_squared_error
from functools import cache
import brainpy


class hashabledict(dict): #if needed to prevent collisions when memoizing formulae
    def __hash__(self):
        return hash(tuple(sorted(list(self.items()), key = lambda x: x[0])))

@cache
def formparser(chemformula): #Only works for single letter isotopes; This version uses memoize to speed up if there are many calculations with repeated formulae
    chemform = chemformula.strip(" ")  
    parsed = re.findall(r'\D\d*',chemform)
    composition = {p[0]:int(p[1:]) if len(p) > 1 else 1 for p in parsed}
    #return(hashabledict(composition))
    return(composition)

#def formparser_popH(chemformula): #Only works for single letter isotopes; without memoize
#    chemformula = chemformula.strip()  
#    parsed = re.findall(r'\D\d*',chemformula)
#    composition = {p[0]:int(p[1:]) if len(p) > 1 else 1 for p in parsed}
#    composition.pop("H")
#    return(composition)
#    return [p.intensity for p in isotopic_variants(composition, npeaks = 3)]

def deu_binom(totalhydrogen, pD, nonexH = 3): #Generate binomial distribution; need to adjust nonexH based on how many hydrogen in the formula are expected to back exchange with the solvent and be unlabeled
    lit = []
    for n in range(totalhydrogen-nonexH+1):
        lit.append(binom.pmf(n, totalhydrogen-nonexH, pD))
    arr = np.array(lit)
    return arr

@cache
def natisodist(formuladict): # Generated natural isotope distribution for correction; uses memoize to potentially speed processing
    theoretical_isotopic_cluster = brainpy.isotopic_variants(formuladict, npeaks=5, charge=1)
    array = np.empty(0)
    for peak in theoretical_isotopic_cluster:
        array = np.append(array, peak.intensity)
    return(array)

def natcor_deudist(formula, pD, scaled = True): #Combine natural isotope distribution with deuterium distibution
    pD = float(pD)
    formdict = hashabledict(formparser(str(formula)))
    nH = formdict.pop('H')
    natdist = natisodist(formdict)
    deudist = deu_binom(nH, pD)
    totdist = np.convolve(natdist, deudist)
    if scaled:
        totdist = totdist/max(list(totdist))
    return(totdist)

def error(pD,formula,actualdata): 
    '''
    calculate expected dist from elements given using above function.
    calculate RMS error from actualdata as compared to calculated data.
    '''
    expecteddata = natcor_deudist(formula,pD)
    actualdataext = np.append(actualdata, [0,0,0,0,0,0])
    actualdatatrim = actualdataext[:len(expecteddata)]
    return(mean_squared_error(actualdatatrim, expecteddata, squared=False))

def optimize(pD, formula, actualdata):
    '''
    Use err function for least squared optimization of pD.
    Takes RMS error calculated by err and varies pD to minimize RMSE
    Returns least_squares optimization output
    Needs pD to be in the ballpark of the best value, otherwise behaviour can be weird
    Assumes Carbon, Hydrogen, Oxygen, Nitrogen, and actualdata are defined prior to using optimize
    '''
    results = least_squares(error, x0 = pD, bounds = (0,1), method = 'dogbox', args=(formula, actualdata), ftol = 1e-9) #trf/dogbox
    param = results.x
    return(results)

def auto_pD_fitting(dataframe): #Automates above individual functions; Must have column labeled as Formula which contains the chemical formulae, and column labeled pD for seed estimates, and data columns labeled _____
    maxiso = int(str(dataframe.iloc[:,-1].name)[1:])
    fitpDs = []  
    RMSE = []
    messages = []
    data_dict = {"M"+str(n):[] for n in range(maxiso+1)}
    data_dict['index'] = []
    dataframeworking = dataframe.copy()
    for index, row in dataframeworking.iterrows():
        pD = row["pD"]
        formula = row["Formula"]
        actualdata = np.array([row["M" + str(n)] for n in range(maxiso+1)])
        fitpD = optimize(pD, formula, actualdata)
        bestfitdist = natcor_deudist(formula, fitpD['x'])
        fitpDs.append(float(fitpD["x"]))
        RMSE.append(fitpD["cost"])
        messages.append(fitpD["message"])
    #    plt.figure(index)
    #    plt.plot(bestfitdist, c = 'r')
    #    plt.bar(range(len(actualdata)), actualdata)
        for n in range(maxiso+1):
            if n in pd.Series(bestfitdist):
                data_dict["M"+str(n)].append(pd.Series(bestfitdist)[n])
            else:
                data_dict["M"+str(n)].append(0)
    dataframeworking["BestFit pD"] = fitpDs
    dataframeworking["RMSE"] = RMSE
    dataframeworking["fun termination"] = messages
    idx = data_dict.pop('index')
    dataframeworking = pd.concat([dataframeworking,pd.DataFrame(data_dict)], axis = 1, ignore_index = True)
    return(dataframeworking)

def auto_natcor_deudist(dataframe):
    dataframeworking = dataframe.copy()
    data_dict = {"M"+str(n):[] for n in range(78)}
    for index, row in dataframeworking.iterrows():
        pD = row["pD"]
        formula = row["Formula"]
        deudist = natcor_deudist(formula, pD)
        for n in range(78):
            if n in pd.Series(deudist):
                data_dict["M"+str(n)].append(pd.Series(deudist)[n])
            else:
                data_dict["M"+str(n)].append(0)
    dataframeworking = pd.concat([dataframeworking,pd.DataFrame(data_dict)], axis = 1, ignore_index = True)
    return(dataframeworking)
        #plt.figure(index)
        #plt.plot(bestfitdist, c = 'r')
        #plt.bar(range(len(actualdata)), actualdata)


def errordecon(pD, fulldist, subdist, HGform): 
    '''
    calculate expected dist from elements given using above function.
    calculate RMS error from actualdata as compared to calculated data.
    '''
    subdist2 = natcor_deudist(HGform, pD)
    fitdist = np.convolve(subdist, subdist2)
    fitdist2 = np.append(fitdist, [0,0,0,0,0,0])
    fitdisttrim = fitdist2[:len(fulldist)]
    return(mean_squared_error(fulldist, fitdisttrim, squared=False))


def optimize_deconvolution(pD, fulldist, subdist, HGform):
    results = least_squares(errordecon, x0 = pD, bounds = (0,1), method = 'dogbox', args=(fulldist, subdist, HGform), ftol = 1e-9, gtol = 1e-15) #trf/dogbox
    return(results)    



df = pd.read_csv("C:\\Users\\6xk\\Desktop\\Records\\Deuterated Lipids\\TestDifferentPythonScripts\\DL_ExtraPosFiles_11-2-22.csv", sep=',', encoding="UTF-8")

dfproc = auto_pD_fitting(df)

dfproc.to_csv("C:\\Users\\6xk\\Desktop\\Records\\Deuterated Lipids\\TestDifferentPythonScripts\\DL_ExtraPosFiles_11-2-22_SPYDER_3exH-dog.csv")

