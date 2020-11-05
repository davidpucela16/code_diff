# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:31:05 2020

@author: pdavid
"""

from class1_domain import Grid 
#from class2 import Solver
#from class3_solver import Solver_t
from class3_assembly import Assembly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class flow_solver():
    
    def __init__(self, Edges, boundary, BCs, init, fin):
        self.boundary=boundary
        self.Edges=Edges
        self.init=init
        self.fin=fin 
        

    
    def pressure_matrix_assembly(self, phi):
        """This function will build the LINEAR matrix to solve the pressure problem"""
        conductance=self.Edges["conductance"]
        A=np.zeros([len(phi), len(phi)])
        
        for i in range(len(phi)): #i will be the value of the vertex we are writting the equations for 
            if self.boundary[i]==0:
                E0=np.where(self.init==i)[0] #edges the vertex i is the first vertex of
                Ef=np.where(self.fin==i)[0]  #edges the vertex i is the last vertex of
                adjacency_edge=np.append(E0,Ef) #This list will have the edges that connect the vertex
                adjacency=np.append(self.init[Ef], self.fin[E0])  #This vector will have the adjacent vertices 
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
        A=self.pressure_matrix_assembly(self.boundary)
        array=np.linalg.solve(A,self.boundary)
        return(array)

class data_visualizing():
    def __init__(self,parameters_geom, phi):
        hx,hy=parameters_geom["hx"], parameters_geom["hy"]
        self.xlen, self.ylen=parameters_geom["xlen"],parameters_geom["ylen"]
        self.X,self.Y=np.meshgrid(np.arange(hx/2,parameters_geom["domain_x"], hx),\
                                  np.arange(hx/2,parameters_geom["domain_y"], hy))
        
    def plot_contour_vessels(self, phi, time, pos_n):
        
        phi_tissue=phi[:(self.xlen*self.ylen)]
        breaks=np.linspace(0,np.max(phi_tissue),10)
        self.C=np.reshape(phi_tissue,(self.ylen,self.xlen))
        print("minimum: ", np.min(phi_tissue))
        print("max: ", np.max(phi_tissue))
        # Create two subplots and unpack the output array immediately
        f, (ax2, ax1) = plt.subplots(2, 1)
        ax1.set_ylabel("concentration")
        CS=ax1.contourf(self.X,self.Y,self.C,breaks, cmap="Reds")
        for i in range(len(init)):    
                ax1.plot([cordx[init[i]], cordx[fin[i]]], [cordy[init[i]], cordy[fin[i]]])
        ax1.set_title('Contour, time {j}'.format(j=time), fontsize=10)
        ax1.grid()
        cbar = f.colorbar(CS)
        c=0
        
        phi_vessel=phi[(k.xlen*k.ylen):]
        ax2.set_title("Average concentration along the axis", fontsize=10)
        for i in pos_n:
            s_vessel=phi_vessel[pos_n[c]]
            ax2.plot(s_vessel, label="vessel {j}".format(j=c))
            c+=1
        ax2.legend()
        f.subplots_adjust(hspace=0.40)
        plt.savefig("solution_time{t}.pdf".format(t=time))
        
def funct(d,L):
    mu=0.1
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
        P1,P2=Edges.loc[i,"vertices"]
        #Important to notice which is ver1 and ver2 cause that will indicate the positive sense of the effective velocity
        #If the velocity is positive, blood flows from lower index vertex to higher index vertex.
        velocity[i]=(P[0]-P[1])*4*cond/(np.pi*diam**2)
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
h=1

hx=hy=h
h_network=0.25
domain_x=15
domain_y=15
start_x=0
start_y=0
inc_t=0.01
coeffs=(1,2,3,4)
d=[1,0.4,0.6]

parameters_geom={"inc_t":inc_t,"dim":dim,"h":h, "hx":hx, "hy":hy, "h_network":h_network, \
"domain_x":domain_x,"domain_y":domain_y, "start_x":start_x, "start_y":start_y, "coeffs":coeffs, \
    "xlen":int(domain_x/hx), "ylen":int(domain_y/hy)}

#physical parameters:
    
#Pressure boundary conditions
BCs={"Outlet":1, "Inlet":20}
boundary=np.array([BCs["Inlet"], 0, BCs["Outlet"], BCs["Outlet"]])
    
Diff_tissue=1
Diff_blood=0.5
Permeability=0.2
linear_consumption=0.5
velocity=np.zeros(len(d))

#Topological parameters of the Network 
init=np.array([0,1,1])
fin=np.array([1,2,3])
cordx=np.array([0.3, 0.7,0.9,0.9])*domain_x
cordy=np.array([0.3, 0.55,0.9,0.2])*domain_y

cords=np.array([cordx,cordy])
edge=np.array([init,fin])



#Setting up of the data frames for later use
Network=pd.DataFrame([[[cordx[0],cordy[0]],1, boundary[0]],[[cordx[1],cordy[1]],[0,2,3], boundary[1]],[[cordx[2],cordy[2]],1, boundary[2]], [[cordx[3],cordy[3]],1, boundary[3]]],\
columns=["coordinates","adjacency", "boundary"])

Edges=pd.DataFrame([[(0,1),dist(Network.loc[init[0],"coordinates"],Network.loc[fin[0],"coordinates"]),d[0],velocity[0], unit_vec(Network.loc[init[0],"coordinates"],Network.loc[fin[0],"coordinates"]),\
funct(d[0],dist(Network.loc[init[0],"coordinates"],Network.loc[fin[0],"coordinates"])) ]] ,columns=["vertices","length","diameter","velocity","unit vector", "conductance"])

number_edges=len(Edges)

Edges=pd.DataFrame()
for i in range(len(init)):
    Edges=Edges.append(pd.DataFrame([[[init[i],fin[i]],dist(Network.loc[init[i],"coordinates"],Network.loc[fin[i],"coordinates"]),d[i],velocity[i], unit_vec(Network.loc[init[i],"coordinates"],Network.loc[fin[i],"coordinates"]),\
    funct(d[i],dist(Network.loc[init[i],"coordinates"],Network.loc[fin[i],"coordinates"])) ]] ,columns=["vertices","length","diameter","velocity","unit vector", "conductance"]), ignore_index=True)
        
        
        
        
### THE SCRIPT BEGINS HERE, THE INFO BEFORE IS COMMONLY ALREADY OBTAINED FROM WHATEVER NETWORK DATA WE ARE USING

#Flow solver


I=flow_solver(Edges,boundary, BCs, init, fin)

Pressure_array=I.get_pressure()

Network["Pressure"]=Pressure_array
Edges["velocity"]=from_Q_get_avgV(Edges, Network["Pressure"])
velocity=Edges["velocity"]

Edges["eff_velocity"]=eff_vel(velocity)
Edges["eff_diff"]=eff_diff(Diff_blood,len(Edges))
Edges["eff_perm"]=eff_perm(Permeability,len(Edges))



p1=Grid(parameters_geom, Network, Edges, cords, edge)
s=p1.plot()  #here the function parametrize is included



source=p1.s
IC_vessels=np.zeros(len(source))
IC_vessels[np.where(p1.s["Edge"]==-1)]=2 #Setting the Dirichlet bounary condition on the inlet vertex
source["IC"]=IC_vessels
new={"x":[np.min(p1.x),np.max(p1.x)], "y":[np.min(p1.y),np.max(p1.y)], "s":source, "t":p1.t}
parameters_geom.update(new)
parameters_physical={"linear_consumption":linear_consumption,"D_tissue":Diff_tissue}
parameters={}; parameters.update(parameters_geom); parameters.update(parameters_physical)

parameters["h_network"]=p1.h

parameters["IC_tissue"]=np.zeros([p1.xlen*p1.ylen])

#I need to give it a vector with the inlet and outlet boundary nodes
def encode_boundary_vessels(source, boundary, BCs):
    a=np.zeros(len(Network))
    c=0
    for i in source.values: 
        b=int(i[1]) #Edge value
        if b<0:
            vertex=-b-1
            if boundary[vertex]==0:
                a[c]=2
            elif boundary[vertex]==BCs["Inlet"]:
                a[c]=1
            elif boundary[vertex]==BCs["Outlet"]:
                a[c]=3
            c+=1
    return(a)

new={"boundary2":encode_boundary_vessels(source,boundary, BCs), "cordx": cordx, "cordy":cordy}
parameters.update(new)

k=Assembly(parameters, Edges, init, fin, parameters["boundary2"])
a=k.assembly()

vessels=k.get_vessel_network(init, fin, k.s)    
    
A=np.concatenate((k.a,k.B),axis=1)
B=np.concatenate((k.C,k.D),axis=1)
A=np.concatenate((A,B))

#A is the full matrix  
phi=np.concatenate((k.phi_tissue,k.phi_vessels))

pos_n=vessels["array_pos_network"]
vess={"cordx":cordx, "cordy":cordy, "init": init, "fin":fin}

def explicit_solver(phi, A, inc_t, iterations, frequency, pos_n):
    w=data_visualizing(parameters_geom, phi)
    
    for i in range(iterations):
        phi=(inc_t*A+np.identity(len(phi))).dot(phi)
        if i%frequency==0:
            w.plot_contour_vessels(phi, i*inc_t, pos_n)
            print(i*inc_t)
            print()
            
    return(phi)


phif=explicit_solver(phi, A, inc_t, 1000, 100, pos_n)
 

     
        
# =============================================================================
# for i in range(len(vessels)):
#     matrix=k.D[vessels.loc[i,"array_pos_network"]]
#     matrix=matrix[:,vessels.loc[i,"array_pos_network"]]
#     row=0
#     for j in matrix:
#         print("This is the Edge ", i, "Row ", row)
#         print(j)
#         print()
#         row+=1
#     #For vertices:
#     print()
# =============================================================================
    




# =============================================================================
# #Vessel solver
# def solver(iterations, D, phi, inc_t, pos):
#     sol=phi
#     for i in range(iterations):
#         inc=D.dot(sol)*inc_t
#         sol=(D*inc_t+np.identity(np.shape(D)[0]))*sol
#         if i%100==0:
#             plt.plot(sol[pos])
#             plt.show()
# =============================================================================

# =============================================================================

# 
# def plot_solution_vessel(sol, xlen, ylen, pos_n):
#     phi_tissue=sol[:(xlen*ylen)]
#     phi_vessel=sol[(xlen*ylen):]
# 
#     plt.xlabel("cell within the vessel")
#     plt.ylabel("average concentration")
#     plt.title("average concentration along the axis")
#     c=0
#     
#     for i in pos_n:
#         s_vessel=phi_vessel[pos_n[c]]
#   
#         plt.plot(s_vessel, label="vessel {j}".format(j=c))
#         c+=1
# 
#     plt.legend()
#     plt.show()
# 
#     return(phi_vessel)	
#         
#             
# def plot_solution(phi,xlen,ylen,X,Y,vess):
#     cordx=vess["cordx"]
#     cordy=vess["cordy"]
#     init=vess["init"]
#     fin=vess["fin"]
#     
#     plt.figure()
#     #To fix the amount of contour levels COUNTOUR LEVELS
#     phi=phi[:(xlen*ylen)]
#     limit=np.ceil(np.max(phi)-np.min(phi))
#     breaks=np.linspace(0,np.max(phi),10)
#     
#     C=np.reshape(phi,(ylen,xlen))
#     plt.contourf(X,Y,C,breaks, cmap="Reds")
#     for i in range(len(init)):    
#         plt.plot([cordx[init[i]], cordx[fin[i]]], [cordy[init[i]], cordy[fin[i]]])
#     print("minimum: ", np.min(phi))
#     print("max: ", np.max(phi))
#     
#     
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.grid()
#     plt.title("bifurcation")
#     plt.savefig("solution.pdf")
#     plt.show()
#     
#     plt.figure()
# 
#     C=np.reshape(phi,(k.ylen,k.xlen))
#     t=C
#     plt.imshow(t[::-1,:], vmin=0, vmax=np.max(C))
#     plt.colorbar()
#     plt.show   
# =============================================================================

# =============================================================================
# phi_tissue=phi[:(k.xlen*k.ylen)]
# breaks=np.linspace(0,np.max(phi_tissue),10)
# C=np.reshape(phi_tissue,(k.ylen,k.xlen))
# print("minimum: ", np.min(phi_tissue))
# print("max: ", np.max(phi_tissue))
# # Create two subplots and unpack the output array immediately
# f, (ax1, ax2) = plt.subplots(1, 2)
# CS=ax1.contourf(k.X,k.Y,C,breaks, cmap="Reds")
# for i in range(len(init)):    
#         ax1.plot([cordx[init[i]], cordx[fin[i]]], [cordy[init[i]], cordy[fin[i]]])
# ax1.set_title('Contour')
# ax1.grid()
# cbar = f.colorbar(CS)
# c=0
# 
# phi_vessel=phi[(k.xlen*k.ylen):]
# ax2.set_title("Average concentration along the axis")
# for i in pos_n:
#     s_vessel=phi_vessel[pos_n[c]]
#     ax2.plot(s_vessel, label="vessel {j}".format(j=c))
#     c+=1
# ax2.legend()
# =============================================================================
