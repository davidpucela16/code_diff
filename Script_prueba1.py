#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:35:19 2020

@author: pdavid
"""


import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg
import pandas as pd
import time
import kk 


#SCRIPT BEGINS HERE!!
    
CN,CS,CE,CW=1,2,3,4
coeffs=(CN,CS,CE,CW)

ePe=10
Dv=Dt=1 #They are of the same order
L=30
Rm=10 #MAX RADIUS
Rv=1
K=ePe*Dv/(Rv**3*2)
Mt=1
inlet_c=1
epsilon=Rv/L

inc_r=0.1
inc_z=0.2
K_m=np.zeros(int(L/inc_z))
K_m[int(len(K_m)/3):-int(len(K_m)/3)]=5e-3

L_c=len(np.where(K_m!=0)[0])*inc_z


parameters={"Diff_vessel": Dv, "Diff_tissue": Dt, "Permeability": K_m, "inc_rho": inc_r, "inc_z": inc_z,
            "R_max": Rm, "R_vessel": Rv, "Length": L, "Tissue_consumption":Mt, "coeff_vel":K, "Inlet":inlet_c, "length":L}


vessel=kk.domain(coeffs, parameters, "vessel")
tissue=kk.domain(coeffs, parameters, "tissue")
    
geom={"zlen":vessel.zlen,"rlen":(vessel.rlen+tissue.rlen),"Z":np.append(vessel.Z,tissue.Z,axis=0),\
      "Rho":np.append(vessel.Rho,tissue.Rho,axis=0)}


new={"Rv":vessel.Rv, "vessel_r": vessel.r, "tissue_r":tissue.r, "z":tissue.z, "L_c":L_c }
geom.update(new)

start=time.time()
#Assembly of the block matrix
ass_tissue=kk.assembly_sparse(tissue, geom, coeffs)
tissue.Down=ass_tissue.Down
ass_vessel=kk.assembly_sparse(vessel, geom, coeffs)
vessel.Up=ass_vessel.Up
end=time.time()
print("Matrix was assembled in: {t}s".format(t=end-start))

A=sp.sparse.csc_matrix(sp.sparse.vstack([vessel.Up, tissue.Down]))

eDam=round(np.max(K_m)*Rv/Dv,4)
parameters["eDam"]=eDam
Dat=round(L_c**2*Mt/(parameters["Inlet"]*Dt),4)
parameters["Dat"]=Dat
epsilon=np.round(Rv/L_c, 2)
parameters["epsilon"]=epsilon

#Create class
k=kk.solver(A,vessel ,tissue,  parameters, geom)
k.set_tissue_consumption_quadratic(1)
k.set_Dirichlet_south(1)
k.set_Dirichlet_vessel(1)
k.set_Dirichlet_north(0)
k.limF=0.1
k.solve_sparse_SS(geom,"Dirichlet_south")


#Validation tissue plot
r=geom["tissue_r"]
a=geom["Rv"]
C_o=1
R=Rm
c2=-a*R**2/(a**2-R**2)
c1=a/(a**2-R**2)


def Euler(a,R, C_o, r):
    c1=a/(a**2-R**2)*C_o
    c2=-a*R**2/(a**2-R**2)*C_o
    return(c1*r+c2/r)

def Lapl(a, R, C_o, r):
    return(C_o*(np.log(r/R)/np.log(a/R)))



def comparison(a,r, R, Sim, C_o):
    plt.plot(Euler(a,R,C_o,r), r, '+r' ,label="Euler")
    plt.plot(Lapl(a,R,C_o,r),r,'bo', label="Laplacian")
    plt.plot(Sim, r, label="sim")
    plt.legend()

def get_tissue_dec(sol_SS, geom):
    phi=sol_SS
    C=phi.reshape((geom["rlen"], geom["zlen"]))
    phi_tissue=C[-geom["tissue_r"].size:,:]
    phi_vessel=C[:geom["vessel_r"].size,:]
    Sim=phi_tissue[:,int(geom["zlen"]/2)]
    return(Sim)


comparison(a,r, R, get_tissue_dec(k.sol_SS, geom), C_o)

