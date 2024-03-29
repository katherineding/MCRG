#!/usr/bin/env python
# coding: utf-8

# # MCRG Code following Swendsen Description circa 1982
#   Using Wolff's Algorithm to combat critical slowing down

from __future__ import division #safeguard against evil floor division
import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt

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
                    S[i,j]*S[i,(j+1)%L]*S[(i+1)%L,(j+1)%L]*S[(i+1)%L,j], # plaquette
                    S[i,j]*(S[i,(j+2)%L] + S[(i+2)%L,j]), #3rd nearest neighbor (2,0)
                    S[i,j]*(S[(i+1)%L,(j+2)%L] + S[(i+2)%L,(j+1)%L] 
                            +S[(i-1)%L,(j+2)%L] + S[(i-2)%L,(j+1)%L]),#4th nearest neighbor (2,1)
                    S[(i+1)%L,j]*S[i,(j+1)%L]*S[(i-1)%L,j]*S[i,(j-1)%L], # sublattice plaquette
                    S[i,j]*(S[(i+2)%L,(j+2)%L] + S[(i-2)%L,(j+2)%L])]#5th nearest neighbor (2,2)
    return val


### First 4 short range odd couplings

def AllOddCoupling(S):
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


### Clustering for Wolff Algorithm

def NNBonds(p):
    '''returns set of bonds that connect site p to its 4 nearest neighbors'''
    i = p[0]; j = p[1];
    nbrs = [(i,(j+1)%L),(i,(j-1)%L),((i+1)%L,j),((i-1)%L,j)]
    bonds = set();
    for n in nbrs:
        bonds.add(frozenset({p,n}))
    return bonds


def buildCluster(S):
    ''''Build Wolff cluster starting from random site for spin configuration S'''
    #random seed location
    L = S.shape[0];
    init = (np.random.choice(L),np.random.choice(L))
    Si = S[init[0],init[1]]
    
    #cluster starts with 1 element
    cluster = {init}
    
    #nearest neighbors make up the frontier
    bonds = NNBonds(init)
    
    #set of points already considered for adding to cluster
    checked = set();
    
    #while the set of fresh bonds is nonempty, do...
    while (len(bonds) > 0):
        if len(cluster) == Ns:
            break;
        #take out one bond in fresh bond set
        #frozenset ijbond represent unordered edge (i,j)
        ijbond = bonds.pop() 
        #add to list of bonds that have been checked
        checked.add(ijbond)
        #pick out element j from edge (i,j)
        jwrap = ijbond - cluster 
        #both i and j may already be in cluster, in this case skip to next iteration
        if len(jwrap) == 0:
            continue;
        #otherwise, only i in cluster already, we are left with j
        j = set(jwrap).pop() #convert to usable form
        Sj = S[j[0],j[1]]
        #if parallel to seed spin, activate bond with probability Pij
        if Sj == Si:
            r = np.random.random()
            if r < Pij:
                #add j to cluster, add nearest bonds of j to the fresh bond list
                #also remove bonds already considered from the fresh bond list
                cluster.add(j) 
                bonds |= NNBonds(j) 
                bonds -= checked 
    return cluster


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
    evenK_new = np.zeros(7,dtype = float)
    mix_11 = np.zeros((7,7),dtype=float)
    mix_01 = np.zeros((7,7),dtype=float)
    #measurement accumulators for y_h
    oddK = np.zeros(4,dtype = float)
    oddK_new = np.zeros(4,dtype = float)
    mix_11_odd = np.zeros((4,4),dtype=float)
    mix_01_odd = np.zeros((4,4),dtype=float)
    
    #Run simulation
    k = 0;
    for n in np.arange(nmeas+nwarm):
        # Every MC n-loop, build a Wolff cluster and flip it
        # Result: A S-field config drawn with probability propto Boltzmann weight
        cluster = buildCluster(S)
        for p in cluster:
            S[p[0],p[1]] = -S[p[0],p[1]]
            
        #Sanity checks
        if n % interval == 0:
            #energy[k] = Energy(S);
            clustersize[k] = len(cluster);
            k = k+1   
            
        # take measurements every (interval) steps if finished warmup
        if n % interval == 0 and n >= nwarm:
            if n % 100 == 0:
                #monitor progress
                print("iteration",n)
            if layer == 1:
                S1 = RGTransform(S,b);
                evenK += AllEvenCoupling(S)
                evenK_new += AllEvenCoupling(S1)
                oddK += AllOddCoupling(S)
                oddK_new += AllOddCoupling(S1)
                #A*B = C, B is unknown, A is symmetric
                #Problem: C is not symmetric!?
                mix_11 += np.outer(AllEvenCoupling(S1),AllEvenCoupling(S1))
                mix_01 += np.outer(AllEvenCoupling(S1),AllEvenCoupling(S))
                mix_11_odd += np.outer(AllOddCoupling(S1),AllOddCoupling(S1))
                mix_01_odd += np.outer(AllOddCoupling(S1),AllOddCoupling(S))

            elif layer == 2:
                assert L > 8, "Lattice too small"
                S1 = RGTransform(S,b);
                S2 = RGTransform(S1,b);
                evenK += AllEvenCoupling(S1)
                evenK_new += AllEvenCoupling(S2)
                oddK += AllOddCoupling(S1)
                oddK_new += AllOddCoupling(S2)
                #A*B = C, B is unknown, A is symmetric
                #Problem: C is not symmetric!?
                mix_11 += np.outer(AllEvenCoupling(S2),AllEvenCoupling(S2))
                mix_01 += np.outer(AllEvenCoupling(S2),AllEvenCoupling(S1))
                mix_11_odd += np.outer(AllOddCoupling(S2),AllOddCoupling(S2))
                mix_01_odd += np.outer(AllOddCoupling(S2),AllOddCoupling(S1))

            elif layer == 3:
                assert L > 16, "Lattice too small"
                S1 = RGTransform(S,b);
                S2 = RGTransform(S1,b);
                S3 = RGTransform(S2,b);
                evenK += AllEvenCoupling(S2)
                evenK_new += AllEvenCoupling(S3)
                oddK += AllOddCoupling(S2)
                oddK_new += AllOddCoupling(S3)
                #A*B = C, B is unknown, A is symmetric
                #Problem: C is not symmetric!?
                mix_11 += np.outer(AllEvenCoupling(S3),AllEvenCoupling(S3))
                mix_01 += np.outer(AllEvenCoupling(S3),AllEvenCoupling(S2))
                mix_11_odd += np.outer(AllOddCoupling(S3),AllOddCoupling(S3))
                mix_01_odd += np.outer(AllOddCoupling(S3),AllOddCoupling(S2))

            elif layer == 4:
                assert L > 32, "Lattice too small"
                S1 = RGTransform(S,b);
                S2 = RGTransform(S1,b);
                S3 = RGTransform(S2,b);
                S4 = RGTransform(S3,b);
                evenK += AllEvenCoupling(S3)
                evenK_new += AllEvenCoupling(S4)
                oddK += AllOddCoupling(S3)
                oddK_new += AllOddCoupling(S4)
                #A*B = C, B is unknown, A is symmetric
                #Problem: C is not symmetric!?
                mix_11 += np.outer(AllEvenCoupling(S4),AllEvenCoupling(S4))
                mix_01 += np.outer(AllEvenCoupling(S4),AllEvenCoupling(S3))
                mix_11_odd += np.outer(AllOddCoupling(S4),AllOddCoupling(S4))
                mix_01_odd += np.outer(AllOddCoupling(S4),AllOddCoupling(S3))
            else:
                raise ValueError("too many RG iterations")
                       
    #Results
    evenK /= ndata; evenK_new /= ndata; mix_11 /= ndata; mix_01 /= ndata;
    oddK /= ndata; oddK_new /= ndata; mix_11_odd /= ndata; mix_01_odd /= ndata;

    print('evenK = ',evenK)
    print('evenK_new = ', evenK_new)
    print('mix_11 = ', mix_11)
    print('subtract ',np.outer(evenK_new,evenK_new))
    print('mix_01 = ', mix_01)
    print('subtract ',np.outer(evenK_new,evenK))
    MatA_even = mix_11-np.outer(evenK_new,evenK_new)
    MatC_even = mix_01-np.outer(evenK_new,evenK)
    print('MatA_even (lhs) = ',MatA_even)
    print('MatC_even (rhs) = ',MatC_even)
    print('\n')

    
    print('oddK = ',oddK)
    print('oddK_new = ', oddK_new)
    print('mix_11 = ', mix_11_odd)
    print('subtract ', np.outer(oddK_new,oddK_new))
    print('mix_01 = ', mix_01_odd)
    print('subtract ', np.outer(oddK_new,oddK))
    #TODO: cancellation bad, how to avoid?
    MatA_odd = mix_11_odd-np.outer(oddK_new,oddK_new)
    MatC_odd = mix_01_odd-np.outer(oddK_new,oddK)
    print('MatA_odd (lhs) = ',MatA_odd)
    print('MatC_odd (rhs) = ',MatC_odd)
    print('\n')

    
    return MatA_even, MatC_even, MatA_odd, MatC_odd


def getExponent(MatA,MatC,Nc):
    '''Get thermal or magnetic exponent based on output of RunMCRG
    and desired number of coupling constants to consider Nc'''
    LinRGMat = la.solve(MatA[0:Nc,0:Nc],MatC[0:Nc,0:Nc])
    lmbd = la.eigvals(LinRGMat);
    amplitude = np.absolute(lmbd)
    imax = np.argmax(amplitude)
    y = np.log(lmbd[imax])/np.log(b)
    return y

## Test for specific set of input parameters

'''# Interactive
L = int(input("Linear Dimension of Lattice: ")); 
nwarm = int(input("number of warm up Monte Carlo sweeps:")); 
nmeas = int(input("number of measurement Monte Carlo sweeps:"));
interval = int(input("interval between data measurements: "));
filename = input('file name for output data: ')
Kc = np.arccosh(3)/4; # Critical temperature Kc assumed to be known
K = Kc; h = 0; #Start on Critical manifold'''

# READ SETTINGS FROM FILE
lines = [line.rstrip('\n') for line in open('setting.txt')]

# Lattice and MC Parameters, Output File name
L = int(lines[0])
nwarm = int(lines[1]); 
nmeas = int(lines[2]);
interval = int(lines[3]);
filename = lines[5]
print(lines)
# Start on Critical manifold
Kc = np.arccosh(3)/4; # Critical temperature Kc assumed to be known
K = Kc; h = 0; 

# RG analysis setting
b = 2; 
layer = int(lines[4]); #How many layers of MCRG to do

# Derived constants
Ns = L*L; #total number of grid points
ndata = nmeas//interval
Pij = 1-np.exp(-2*K) # Wolff add probability
energy = np.zeros((nmeas+nwarm)//interval,dtype=float)
clustersize = np.zeros((nmeas+nwarm)//interval,dtype=int)

#Initialize 2d spin field
S = np.random.choice([-1,1],(L,L))

print("=========== SETTINGS FROM FILE ==================")
print("Linear Dimension of Lattice:", L )
print("How many layers of RG Transformations? ", layer )
print("Number of warm up Monte Carlo sweeps:", nwarm )
print("Number of measurement Monte Carlo sweeps:", nmeas )
print("Interval between data measurements:", interval)
print("Saving results to:","data/"+filename+'.txt')
print("================ RUNNING ======================")

## GO!
MatA_even, MatC_even, MatA_odd, MatC_odd = RunMCRG(K,h)

yt_arr = np.empty(7, dtype = complex);
yh_arr = np.empty(4, dtype = complex);

for i in np.arange(7):
    yt_arr[i] = getExponent(MatA_even,MatC_even,i+1)

for i in np.arange(4):
    yh_arr[i] = getExponent(MatA_odd,MatC_odd,i+1)

print("==================== RESULTS ======================")
print("mean cluster size as fraction of lattice size: ",np.mean(clustersize)/Ns)
print("y_t array = ", yt_arr)
print("y_h array = ", yh_arr)


#WRITE DATA TO TEXT FILE
f = open("data/"+filename+'.txt','a')
print("======================= SETTINGS =======================",file= f)
print("L = ", L, file = f)
print("K = ", K, file = f)
print("h = ", h, file = f)
print("b = ", b, file = f)
print("layer = ", layer, file = f)
print("nwarm = ", nwarm, file = f)
print("nmeas = ", nmeas, file = f)
print("interval = ", interval, file = f)
print("ndata = ", ndata, '\n', file = f)
print("======================== RESULTS ========================", file = f)
print("Mean Wolff cluster size = ",np.mean(clustersize)/Ns, '*', Ns, "\n",file = f)
print("y_t array = ", yt_arr,"\n",file = f)
print("y_h array = ", yh_arr,"\n",file = f)
print("======================= SANITY CHECK ====================", file = f)
print('MatA_even (lhs) = ',MatA_even, "\n",file = f)
print('MatC_even (rhs) = ',MatC_even, "\n",file = f)
print('MatA_odd (lhs) = ',MatA_odd, "\n",file = f)
print('MatC_odd (rhs) = ',MatC_odd, "\n",file = f)
print("=========================== END =========================", file = f)
print('\n\n\n\n\n',file = f)
f.close()


print("wrote run data to: data/"+filename+'.txt')



