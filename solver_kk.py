# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 10:45:46 2020

@author: pdavid
"""

from class1_domain import Grid 
#from class2 import Solver
from class3_solver import Solver_t
from class3_assembly import Assembly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Mult_vess_coupl


        

def plot_solution_vessel(sol, xlen, ylen,C):
    plt.figure()
    phi_tissue=sol[:(xlen*ylen)]
    phi_vessel=sol[(xlen*ylen):]
    plt.plot(phi_vessel, label='vessel')
    
    coupl=C.dot(phi_tissue)-np.identity(len(phi_vessel)).dot(phi_vessel)
    plt.plot(coupl, label='flux out')
    plt.legend()
    plt.show

    return(phi_vessel)		


def iterate_fweul(A,phi, inc_t,k):  
    """Function that we have to call to do one iteration with forward Euler
    The arguments are the matrix, the vector with the boundary conditions and the time increment"""
    inc=A.dot(phi)*inc_t
    phi_plus_one=phi+inc
    return(phi_plus_one)
        
def solve_lin_sys(A,phi, max_it, inc_t, xlen, ylen,k):
    it=0
    sol=np.empty([max_it+1,len(phi)])
    sol[0,:]=phi
    while it<max_it:
    #while self.it < max_it or err<1:
        sol[it+1,:]=iterate_fweul(A,sol[it,:],inc_t,k)
        it+=1
    return(sol)
        
            
def plot_solution(phi,xlen,ylen,X,Y):
    plt.figure()
    #To fix the amount of contour levels COUNTOUR LEVELS
    phi=phi[:(xlen*ylen)]
    limit=np.ceil(np.max(phi)-np.min(phi))
    breaks=np.linspace(0,np.max(phi),10)
    
    C=np.reshape(phi,(ylen,xlen))
    C=C
    plt.contourf(X,Y,C,breaks, cmap="Reds")
    plt.colorbar(ticks=breaks, orientation="vertical")
    print("minimum: ", np.min(phi))
    print("max: ", np.max(phi))
    
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.title("Heat line source")
    plt.savefig("solution.pdf")
    plt.show()
    
    plt.figure()
#==============================================================================
#     C=np.reshape(phi,(k.ylen,k.xlen))
#     t=C
#     plt.imshow(t[::-1,:], vmin=0, vmax=np.max(C))
#     plt.colorbar()
#     plt.show   
#==============================================================================




A=np.concatenate((k.a,k.B),axis=1)
B=np.concatenate((k.C,k.D),axis=1)
A=np.concatenate((A,B))

#A is the full matrix  
phi=np.concatenate((k.phi_tissue,k.phi_vessels))
    

sol=solve_lin_sys(A,phi,5000,inc_t,k.xlen,k.ylen,k)
m,_=sol.shape

for i in range(1,m,100):

    plot_solution(sol[i,:],k.xlen,k.ylen,k.X,k.Y)
    plot_solution_vessel(sol[i,:],k.xlen,k.ylen,k.C)

    






    