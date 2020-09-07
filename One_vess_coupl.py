# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:22:19 2020

@author: pdavid
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:03:57 2020

@author: pdavid
""" 

from class1_domain import Grid 
#from class2 import Solver
from class3_solver import Solver_t
from class3_assembly import Assembly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def dist(p1,p2):
    return(np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))

def unit_vec(p1,p2):
    k=np.array(p2)-np.array(p1)
    return(k/np.linalg.norm(k))


#Parameters

#geometrical/topological parameters:
dim=2
h=1
hx=hy=h
h_network=0.5
domain_x=5
domain_y=5
start_x=0
start_y=0
inc_t=0.1

parameters_geom={"inc_t":inc_t,"dim":dim,"h":h, "hx":hx, "hy":hy, "h_network":h_network,"domain_x":domain_x,"domain_y":domain_y, "start_x":start_x, "start_y":start_y}


#physical parameters:
Diff_tissue=5
Diff_blood=0.5
Permeability=10
linear_consumption=1

coord=np.array([[0.2,0.6],[0.7,0.55]])
coord[:,0]*=domain_x
coord[:,1]*=domain_y

velocity=np.random.random()*4
Network=pd.DataFrame([[coord[0],0],[coord[1],[0,1,2]]],columns=["coordinates","adjacency"])
Edges=pd.DataFrame([[(0,1),dist(coord[0],coord[1]),1,velocity, unit_vec(coord[0],coord[1])]],columns=["vertices","length","diameter","velocity","unit vector"])
number_edges=len(Edges)


p1=Grid(parameters_geom, Network, Edges)
s=p1.plot()  #here the function parametrize is included

new={"x":[np.min(p1.x),np.max(p1.x)], "y":[np.min(p1.y),np.max(p1.y)], "s":p1.s, "t":p1.t}
parameters_geom.update(new)
parameters_physical={"linear_consumption":linear_consumption,"D_tissue":Diff_tissue, "D_blood":Diff_blood, "Permeability":Permeability, "velocity":velocity}
parameters={}; parameters.update(parameters_geom); parameters.update(parameters_physical)


#IC
IC_vessels=np.zeros(len(p1.s))
IC_tissue=np.zeros(np.shape(p1.Cord[0]))

#BCs:
BCn=np.zeros(p1.x.shape)
BCs=np.zeros(p1.x.shape)
BCe=np.zeros(p1.y.shape)
BCw=np.zeros(p1.y.shape)

BC=[BCn,BCs,BCe,BCw]

BC_vessels=np.array([9,0])

conditions={"BC_tissue":BC, "BC_vessels":BC_vessels, "IC_tissue":IC_tissue, "IC_vessels":IC_vessels}
parameters.update(conditions)

k=Assembly(parameters) #initialization of the object
k.assembly()

l=np.zeros(4)

l[0],_=k.a.shape
l[1],_=k.B.shape
l[2],_=k.C.shape
l[3],_=k.D.shape

c=0
for i in l:
    if c==0:
        print("matrix a by rows")
        for j in range(int(i)):
            print(k.a[j,:].reshape(k.ylen,k.xlen))
            print()
    if c==1:
        print("matrix B by rows")
        for j in range(int(i)):
            print(k.B[j,:])
            print()
    if c==2:
        print("matrix C by rows")
        for j in range(int(i)):
            print(k.C[j,:])
            print()
    if c==3:
        print("matrix D by rows")
        for j in range(int(i)):
            print(k.D[j,:])
            print()
    c+=1    
    

def iterate_fweul(A,phi, inc_t):  
    """Function that we have to call to do one iteration with forward Euler
    The arguments are the matrix, the vector with the boundary conditions and the time increment"""
    inc=A.dot(phi)*inc_t
    phi_plus_one=phi+inc
    return(phi_plus_one)
        
def solve_lin_sys(A,phi, max_it, inc_t, xlen, ylen):
    it=0
    sol=np.empty([max_it+1,len(phi)])
    sol[0,:]=phi
    while it<max_it:
    #while self.it < max_it or err<1:
        sol[it+1,:]=iterate_fweul(A,sol[it,:],inc_t)
        it+=1
    return(sol)
        
            
def plot_solution(phi,xlen,ylen,X,Y):
    plt.figure
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
    return()

###Concatenation
    
A=np.concatenate((k.a,k.B),axis=1)
B=np.concatenate((k.C,k.D),axis=1)
A=np.concatenate((A,B))

#A is the full matrix  
phi=np.concatenate((k.phi_tissue,k.phi_vessels))
    

sol=solve_lin_sys(A,phi,50,inc_t,k.xlen,k.ylen)
m,_=sol.shape
for i in range(1,m):
    plot_solution(sol[i,:],k.xlen,k.ylen,k.X,k.Y)
    
    






# =============================================================================
# plt.figure()
# k.C=np.reshape(B[:k.len_tissue],(k.ylen,k.xlen))
# t=k.C
# plt.imshow(t[::-1,:], vmin=0, vmax=np.max(B[:k.len_tissue]))
# plt.colorbar()
# plt.show
# 
# plt.figure()
# a=B[-k.len_net:]
# plt.plot(a)
# plt.show()
# 
# =============================================================================
            
#==============================================================================
# k=Solver(p1.phi,p1.dim,p1.hx,p1.hy,p1.x,p1.y,p1.xlen,p1.ylen,p1.t,p1.X,p1.Y,Diff_tissue, Diff_blood, Permeability,h_network,IC_vessels, velocity)
# k.set_BC(BC)
# k.assembly()     
# 
# plt.figure()
# B=k.solve_linear_syst()
# k.plot_solution(B)
# plt.show()
# 
# plt.figure()
# k.C=np.reshape(B,(k.ylen,k.xlen))
# t=k.C
# plt.imshow(t[::-1,:], vmin=0, vmax=50)
# plt.colorbar()
# plt.show
#==============================================================================

