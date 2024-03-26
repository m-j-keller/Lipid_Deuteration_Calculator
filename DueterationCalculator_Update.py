# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:31:51 2024

@author: 6xk
"""

from functools import cache
import re
from brainpy import isotopic_variants
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from scipy.stats import binom

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

def deu_binom(totalhydrogen, pD, nonexH = 3): #Generate binomial distribution; need to adjust nonexH based on how many hydrogen in the formula are expected to back exchange with the solvent and be unlabeled
    lit = []
    for n in range(totalhydrogen-nonexH+1):
        lit.append(binom.pmf(n, totalhydrogen-nonexH, pD))
    arr = np.array(lit)
    return arr

@cache
def natisodist(formuladict): # Generated natural isotope distribution for correction; uses memoize to potentially speed processing
    theoretical_isotopic_cluster = isotopic_variants(formuladict, npeaks=5, charge=1)
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
    results = minimize(error, x0 = pD, bounds = [[0,1]], method = 'Nelder-Mead', args=(formula, actualdata)) #trf/dogbox
    param = results.x
    return(results)

def deuterium_fitting(df):
    int_cols = [*df.columns[7:]]
    int_iter = list(zip(*[df[c] for c in int_cols]))
    pDs = list(df["pD"])
    formula = list(df["Formula"])
    test = [optimize(pD, form, row) for row, pD, form in zip(int_iter, pDs, formula)]
    maxiso = int(str(df.iloc[:,-1].name)[1:])
    df["BestFit pD"] = [t['x'][0] for t in test]
    df["RMSE"] = [t['fun'] for t in test]
    df["fun termination"] = [t['message'] for t in test]
    def exp_dist(formula, x):
        dist = natcor_deudist(formula, x)
        dist = np.concatenate((dist,np.zeros(maxiso - len(dist))))
        return dist
    dists = np.array([exp_dist(f, t['x'][0]) for f,t in zip(formula, test)])
    dists = pd.DataFrame(dists, columns= [f'isotope_{i}' for i in range(dists.shape[1])])
    df = pd.concat([df,dists], axis = 1)    
    return df

df = pd.read_csv("\data.csv", sep=',', encoding="UTF-8")

dfproc = deuterium_fitting(df)

