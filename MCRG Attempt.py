#!/usr/bin/env python
# coding: utf-8

# # MCRG Code following Swendsen Description circa 1982 
#   With Metropolis Hastings MC updates
# # This version is NOT USED IN PRODUCTION

from __future__ import division #safeguard against evil floor division
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
import timeit
np.set_printoptions(precision = 3,suppress=False)


### Block spin transform, scale factor = b 

def assignBlockSpin(total):
    '''Rule for assigning block spin value. Random tiebreaker'''
    if total > 0:
        s = 1;
    elif total < 0:
        s = -1;
    else:
        s = np.random.choice([-1,1])
    return s


def RGTransform(S,b):
    '''Take a spin config S and produce renormalized block spin config that 
    groups b*b spins together into one block. Does not modify input S'''
    L = S.shape[0];
    assert L//b >= 2, "Renormalized lattice will have linear dimension <=1"
    newS = np.empty([L//b, L//b],dtype=int)
    for j in np.arange(L//b):
        for i in np.arange(L//b):
            block = S[(b*i):(b*i+b),(b*j):(b*j+b)]
            total = np.sum(block)
            newS[i,j] = assignBlockSpin(total);    
    return newS


### First 7 short range even couplings

def AllEvenCoupling(S):
    '''for spin field config S, 
    Integrate measurement of first 7 even correlation functions in one vector'''
    L = S.shape[0];
    assert L >=3, "Lattice too small to fit first 7 even couplings"
    val = np.zeros(7,dtype = float);
    for j in np.arange(L):
        for i in np.arange(L):
            val += [S[i,j]*(S[i,(j+1)%L] + S[(i+1)%L,j]), #nearest neighbor (1,0)
                    S[i,j]*(S[(i+1)%L,(j+1)%L] + S[(i-1)%L,(j+1)%L]), #next nearest neighbor (1,1)
                    S[i,j]*(S[i,(j+2)%L] + S[(i+2)%L,j]), #3rd nearest neighbor (2,0)
                    S[i,j]*(S[(i+1)%L,(j+2)%L] + S[(i+2)%L,(j+1)%L] 
                            +S[(i-1)%L,(j+2)%L] + S[(i-2)%L,(j+1)%L]),#4th nearest neighbor (2,1)
                    S[i,j]*(S[(i+2)%L,(j+2)%L] + S[(i-2)%L,(j+2)%L]),#5th nearest neighbor (2,2)
                    S[i,j]*S[i,(j+1)%L]*S[(i+1)%L,(j+1)%L]*S[(i+1)%L,j], # plaquette
                    S[(i+1)%L,j]*S[i,(j+1)%L]*S[(i-1)%L,j]*S[i,(j-1)%L]] # sublattice plaquette
    return val


### first 4 short range couplings

def AllOddCoupling(S):
    '''for spin field config S, 
    Integrate measurement of first 4 odd correlation functions in one vector'''
    L = S.shape[0];
    assert L >=3, "Lattice too small to fit first 4 odd couplings"
    val = np.zeros(4,dtype = float);
    for j in np.arange(L):
        for i in np.arange(L):
            val += [0, #magnetization
                    S[i,j]*S[(i+1)%L,j]*S[(i+1)%L,(j+1)%L]+
                    S[i,j]*S[i,(j+1)%L]*S[(i-1)%L,(j+1)%L]+
                    S[i,j]*S[(i-1)%L,j]*S[(i-1)%L,(j-1)%L]+
                    S[i,j]*S[i,(j-1)%L]*S[(i+1)%L,(j-1)%L], #3 spin plaquette
                    S[i,j]*S[(i+1)%L,j]*S[(i+2)%L,(j+1)%L]+
                    S[i,j]*S[i,(j+1)%L]*S[(i-1)%L,(j+2)%L]+
                    S[i,j]*S[(i-1)%L,j]*S[(i-2)%L,(j-1)%L]+
                    S[i,j]*S[i,(j-1)%L]*S[(i+1)%L,(j-2)%L], # 3 spin angle
                    S[i,j]*(S[(i+1)%L,j]*S[(i+2)%L,j] + 
                            S[i,(j+1)%L]*S[i,(j+2)%L])] #3 spin row
    val[0] = np.sum(S);
    return val

### Integrated MC + RG simulation function

def Energy(S):
    '''Brute force Find energy of spin configuration S for sanity check'''
    L = S.shape[0];
    E = 0;
    for i in np.arange(L):
        for j in np.arange(L):
            E += K*S[i,j]*(S[i,(j+1)%L] + S[(i+1)%L,j])
    E += h*np.sum(S)
    return E

def RunMCRG(K,h):
    '''Run MCRG simulation to find y_t, y_h exponent, 
    keeping Nc_even and Nc_odd coupling terms'''
    print('running MCRG for linear size',L,'lattice.')
    print('Setting K =', K, " and h = ",h)
    
    #measurement accumulators for y_t
    evenK = np.zeros(7,dtype = float)
    evenK_1 = np.zeros(7,dtype = float)
    mix_11 = np.zeros((7,7),dtype=float)
    mix_01 = np.zeros((7,7),dtype=float)
    #measurement accumulators for y_h
    oddK = np.zeros(4,dtype = float)
    oddK_1 = np.zeros(4,dtype = float)
    mix_11_odd = np.zeros((4,4),dtype=float)
    mix_01_odd = np.zeros((4,4),dtype=float)
    
    #Run simulation
    k = 0
    for n in np.arange(nmeas+nwarm):
        # Every MC n-loop, go through every [i,j] pair and propose 
        # flipping s[i,j]. 
        # Result: A S-field config drawn with probability propto Boltzmann weight
        for j in np.arange(L):
            for i in np.arange(L):
                # calculate alpha based on energy difference
                d_hterm = -2*h*S[i,j];
                d_tterm = -2*K*S[i,j]*(S[(i+1) % L,j]+S[(i-1) % L,j]
                          +S[i,(j+1)%L]+S[i,(j-1)%L])
                alpha = np.exp(d_tterm);
                #Probability Min(1,alpha) of accepting flip 
                #If accept, update sigma field
                #If reject, do nothing, look at next entry of S
                if beta_array[n,i,j] < alpha:
                    S[i,j] = -S[i,j]; 
                    
        #sanity check
        if n % interval == 0:
            energy[k] = Energy(S)
            k +=1
        
        # take measurements every (interval) steps if finished warmup
        if n % interval == 0 and n >= nwarm:
            S1 = RGTransform(S,b);
            evenK += AllEvenCoupling(S)
            evenK_1 += AllEvenCoupling(S1)
            oddK += AllOddCoupling(S)
            oddK_1 += AllOddCoupling(S1)
            #A*B = C, B is unknown, A is symmetric
            #Problem: C is not symmetric!?
            mix_11 += np.outer(AllEvenCoupling(S1),AllEvenCoupling(S1))
            mix_01 += np.outer(AllEvenCoupling(S1),AllEvenCoupling(S))
            mix_11_odd += np.outer(AllOddCoupling(S1),AllOddCoupling(S1))
            mix_01_odd += np.outer(AllOddCoupling(S1),AllOddCoupling(S))
    
    #Results
    evenK /= ndata; evenK_1 /= ndata; mix_11 /= ndata; mix_01 /= ndata;
    oddK /= ndata; oddK_1 /= ndata; mix_11_odd /= ndata; mix_01_odd /= ndata;
    print('evenK = ',evenK)
    print('evenK_1 = ', evenK_1)
    print('mix_11 = ', mix_11)
    print('subtract ',np.outer(evenK_1,evenK_1))
    print('mix_01 = ', mix_01)
    print('subtract ',np.outer(evenK_1,evenK))
    MatA = mix_11-np.outer(evenK_1,evenK_1)
    MatC = mix_01-np.outer(evenK_1,evenK)
    #print('MatA (lhs) = ',MatA)
    #print('MatC (rhs) = ',MatC)
    LinRGMat = la.solve(MatA[0:Nc_even,0:Nc_even],MatC[0:Nc_even,0:Nc_even])
    print('linearized RG transformation = ',LinRGMat)
    lmbd = la.eigvals(LinRGMat);
    print('eigenvalues are',lmbd)
    amplitude = np.absolute(lmbd)
    print('eigenvalue amplitudes are',amplitude)
    #Only the eigenvalue with maximum amplitude is important.
    #This eigenvalue should generically be real
    imax = np.argmax(amplitude)
    y_t = np.log(lmbd[imax])/np.log(b)
    print('exponent y_t = ',y_t,'\n')
    
    print('oddK = ',oddK)
    print('oddK_1 = ', oddK_1)
    print('mix_11 = ', mix_11_odd)
    print('subtract ', np.outer(oddK_1,oddK_1))
    print('mix_01 = ', mix_01_odd)
    print('subtract ', np.outer(oddK_1,oddK))
    #TODO: cancellation bad, how to avoid?
    MatA = mix_11_odd-np.outer(oddK_1,oddK_1)
    MatC = mix_01_odd-np.outer(oddK_1,oddK)
    #print('MatA (lhs) = ',MatA)
    #print('MatC (rhs) = ',MatC)
    LinRGMat = la.solve(MatA[0:Nc_odd,0:Nc_odd],MatC[0:Nc_odd,0:Nc_odd])
    print('linearized RG transformation = ',LinRGMat)
    lmbd = la.eigvals(LinRGMat);
    print('eigenvalues are',lmbd)
    amplitude = np.absolute(lmbd)
    print('eigenvalue amplitudes are',amplitude)
    #Only the eigenvalue with maximum amplitude is important.
    #This eigenvalue should generically be real
    imax = np.argmax(amplitude)
    y_h = np.log(lmbd[imax])/np.log(b)
    plt.figure();
    plt.imshow(S);plt.colorbar();
    print('exponent y_h = ',y_h)
    
    return y_t,y_h


## Test for specific set of input parameters

# Lattice and MC Parameters
L = int(input("Linear Dimension of Lattice:"));
Kc = np.arccosh(3)/4; # Critical temperature Kc assumed to be known
K = Kc; h = 0; #Start on Critical manifold
nwarm = int(input("number of warm up Monte Carlo sweeps:")); 
nmeas = int(input("number of measurement Monte Carlo sweeps:"));
interval = int(input("interval between data measurements: "));

# RG analysis setting
Nc_even = int(input("number of even coupling constants to consider:")); 
Nc_odd = int(input("number of odd coupling constants to consider:"));
b = 2; 

# Derived constants
ndata = nmeas//interval
Ns = L*L; #total number of grid points
energy = np.zeros((nmeas+nwarm)//interval,dtype=float)

#Initialize 2d spin field
S = np.random.choice([-1,1],(L,L))
#Initialize an array of random beta in [0,1] to use in M-Hastings
beta_array = np.random.random_sample((nwarm+nmeas,L,L))


## GO!
RunMCRG(K,h)

plt.plot(energy)





