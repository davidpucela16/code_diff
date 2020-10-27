# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:31:05 2020

@author: pdavid
"""

from class1_domain import Grid 
#from class2 import Solver
from class3_solver import Solver_t
from class3_assembly import Assembly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class flow_solver():
    
    def __init__(self, Edges, Network, BCs):
        self.Network=Network
        self.InitP=self.boundary(Network, BCs)
        self.Edges=Edges
        
    def boundary(self, Network, BCs):
        """This function will add a column to the Network DataFrame with the values of the pressure 
        at the boundary nodes. If the node is not a boundary the value returned will be 0. Take care then
        not to assign a 0 pressure boundary condition. 
        
        The function will modify the variable self.Network and will return the pressure column as a np.array"""
        
        Pressure=np.zeros(Network.shape[0])   
        c=0
        for i in Network.loc[:,"boundary"]:
            if i!="No":
                Pressure[c]=BCs["Inlet"] if i=="Inlet" else BCs["Outlet"]
            c+=1
        self.Network["Boundary_P"]=Pressure
        return(Pressure)
    
    def pressure_matrix_assembly(self, phi):
        """This function will build the LINEAR matrix to solve the pressure problem"""
        conductance=self.Edges["conductance"]
        A=np.zeros([len(phi), len(phi)])
        
        for i in range(len(phi)): #i will be the value of the vertex we are writting the equations for 
            if self.Network.loc[i,"boundary"]=="No":
                adjacency_edge=self.Network.loc[i,"Edges"] #This list will have the edges that connect the vertex
                adjacency=self.Network.loc[i,"adjacency"]  #This vector will have the adjacent vertices 
                c=0 #This will help to retrouve the value of the edge
                for j in adjacency: #Therefore j will be the value of the neighboring vertex
                    A[i,j]-=conductance[adjacency_edge[c]] #Conductance of the specific edge
                    A[i,i]+=conductance[adjacency_edge[c]]
                    c+=1
            else:
                A[i,i]=1
        return(A)
        
    def get_pressure(self):
        """function that returns the pressure values at each vertex"""
        A=self.pressure_matrix_assembly(self.InitP)
        array=np.linalg.solve(A,self.InitP)
        return(array)
        
        
def funct(d,L):
    mu=1
    return(np.pi*d**4/(128*mu*L))

def dist(p1,p2):
    return(np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))

def unit_vec(p1,p2):
    k=np.array(p2)-np.array(p1)
    return(np.around(k/np.linalg.norm(k) , decimals= 2 ))

def from_Q_get_avgV(Edges, P):
    """Function that calculates the average velocity inside a vessel in function
    of the vessel diameter, conductance (info stored in the DF Edges), and pressure 
    difference between vertices (vertices stored in Edges and the pressure is given 
    in the vector P)"""
    velocity=np.zeros(len(Edges))
    for i in range(len(Edges)):
        cond, diam=Edges.loc[i,"conductance"],Edges.loc[i,"diameter"]     
        ver1,ver2=Edges.loc[i,"vertices"]
        velocity[i]=np.abs(ver1-ver2)*4*cond/(np.pi*diam**2)
    return(velocity)

def eff_vel(array):
        return(array)
        
def eff_diff(D_blood, len_array):
    return(np.zeros(len_array)+D_blood)

def eff_perm(perm, len_array):
    return(np.zeros(len_array)+perm)
#Parameters

#geometrical/topological parameters:
dim=2
h=0.5
hx=hy=h
h_network=0.25
domain_x=7
domain_y=7
start_x=0
start_y=0
inc_t=0.001
coeffs=(1,2,3,4)

parameters_geom={"inc_t":inc_t,"dim":dim,"h":h, "hx":hx, "hy":hy, "h_network":h_network, \
"domain_x":domain_x,"domain_y":domain_y, "start_x":start_x, "start_y":start_y, "coeffs":coeffs}


#physical parameters:
Diff_tissue=1
Diff_blood=0.5
Permeability=1
linear_consumption=0

coord=np.array([[0.3,0.3],[0.7,0.65],[0.9,0.9],[0.8,0.1]])
coord[:,0]*=domain_x
coord[:,1]*=domain_y

d=[1,0.4,0.6]
velocity=np.zeros(len(d))


Network=pd.DataFrame([[coord[0],1, "Inlet"],[coord[1],[0,2,3], "No"],[coord[2],1, "Outlet"], [coord[3],1, "Outlet"]],\
columns=["coordinates","adjacency", "boundary"])

Edges=pd.DataFrame([[(0,1),dist(coord[0],coord[1]),d[0],velocity[0], unit_vec(coord[0],coord[1]),\
funct(d[0],dist(coord[0],coord[1])) ]] ,columns=["vertices","length","diameter","velocity","unit vector", "conductance"])

number_edges=len(Edges)

Network["Edges"]=[0,[0,1,2], 1,2]

Edges=Edges.append(pd.DataFrame([[(1,2),dist(coord[1],coord[2]),d[1],velocity[1], unit_vec(coord[1],coord[2]), funct(d[1],dist(coord[1],coord[2])) ]],columns=["vertices","length","diameter","velocity","unit vector", "conductance"]), ignore_index=True)
Edges=Edges.append(pd.DataFrame([[(1,3),dist(coord[1],coord[3]),d[2],velocity[2], unit_vec(coord[1],coord[3]), funct(d[2],dist(coord[1],coord[2])) ]],columns=["vertices","length","diameter","velocity","unit vector", "conductance"]), ignore_index=True)







### THE SCRIPT BEGINS HERE, THE INFO BEFORE IS COMMONLY ALREADY OBTAINED FROM WHATEVER NETWORK DATA WE ARE USING

BCs={"Outlet":4, "Inlet":6}
k=flow_solver( Edges,Network, BCs)

Pressure_array=k.get_pressure()

Network["Pressure"]=Pressure_array
Edges["velocity"]=from_Q_get_avgV(Edges, Network["Pressure"])
velocity=Edges["velocity"]

Edges["eff_velocity"]=eff_vel(velocity)
Edges["eff_diff"]=eff_diff(Diff_blood,len(Edges))
Edges["eff_perm"]=eff_perm(Permeability,len(Edges))



p1=Grid(parameters_geom, Network, Edges)
s=p1.plot()  #here the function parametrize is included



source=p1.s
IC_vessels=np.zeros(len(source))
IC_vessels[0]=2
source["IC"]=IC_vessels
new={"x":[np.min(p1.x),np.max(p1.x)], "y":[np.min(p1.y),np.max(p1.y)], "s":source, "t":p1.t}
parameters_geom.update(new)
parameters_physical={"linear_consumption":linear_consumption,"D_tissue":Diff_tissue}
parameters={}; parameters.update(parameters_geom); parameters.update(parameters_physical)

#The bifurcations need to be encoded somehow, and be differentiated from the boundary nodes and from the other nodes
b=int(np.where(Network["Boundary_P"].values==0)[0]) #vertex which is not in the boundary, hence it must be a bifurcation

bif_edges=Network.loc[b,"Edges"]

bifurcation=2
boundary=1

source["bifurcation?"]=p1.vertices




k=Assembly(parameters, Edges)


