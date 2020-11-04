# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:11:47 2020

@author: david
"""
###NEW VERSION GITHUB

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import random
from class1_domain import Grid 
import pandas as pd

class Assembly(Grid):
    
    
    def __init__(self, parameters, Edges, init, fin, boundary):
        
        
        self.boundary=boundary
        self.diam=Edges["diameter"]
        self.init=init
        self.fin=fin
        
        self.dim=parameters["dim"]
        self.hx,self.hy=parameters["hx"],parameters["hy"]
        self.x=np.arange(parameters["x"][0],parameters["x"][1]+self.hx,self.hx) 
        self.y=np.arange(parameters["y"][0],parameters["y"][1]+self.hy,self.hy)
        self.coeffs=parameters["coeffs"] 
        self.CN,self.CS,self.CE,self.CW=self.coeffs
        super().matrix(self.x,self.y)
        self.hx,self.hy=(parameters["hx"],parameters["hy"])
        self.h_network=parameters["h_network"]
        self.h_vertices=np.array([])
        self.K_eff=Edges["eff_perm"]
        self.Calc_Keff()
        self.h_tot=np.append(self.h_network, self.h_vertices)
        #You have to add the effective permeabilities for the different bifurcations and vertices
        self.D_tissue=parameters["D_tissue"]
        #For the initial simulation 
        self.c_0=1
        self.t=parameters["t"] #vector with the IDs of the tissue
        self.M=parameters["linear_consumption"]
        self.D_eff=Edges["eff_diff"]
        self.U_eff=Edges["eff_velocity"]
        
        
        self.cordx=parameters["cordx"]
        self.cordy=parameters["cordy"]
        
        
        #index of source, ind cell it is in, Edge it belongs to 
        source=parameters["s"]
        IC_vessels=source["IC"]
        self.s=np.array([np.arange(len(source)),source["ind cell"],source["Edge"], IC_vessels]).T
        self.s=self.s.astype(int)
        self.dfs=source
        self.num_cells_edge=((Edges["length"]/self.h_network).values).astype(int)-1
        
        
        
        
        #vessel
        #These two arrays should have as many values as edges in the network since the effective convective and dispersive 
        #coefficients are gonna be different in each vessel
        self.u_network=Edges["eff_velocity"] #effective velocity in the blood network 
        self.D_network=Edges["eff_diff"]
    
        
        
        self.IC_vessels=self.s[:,3]
        self.IC_tissue=np.ndarray.flatten(parameters["IC_tissue"])
        
    
    def Calc_Keff(self):
        self.K_eff=self.K_eff.values
        for i in np.arange(len(self.boundary)):
            edges=np.append(np.where(self.init==i), np.where(self.fin==i))
            factor=(self.diam[edges]**2).dot(self.h_network[edges])
            Upper=np.sum(self.diam[edges]**2*self.K_eff[edges]*self.h_network[edges])
            self.K_eff=np.append(self.K_eff,Upper/factor)
            Upper_2=np.sum(self.h_network[edges])/2
            self.h_vertices=np.append(self.h_vertices,Upper_2)
        return()
        
    def UDDiff(self, pos_net, e, pos_frw, *fac):
        """Upwind scheme for Downstream Diffusive flux"""
        factor=fac[0] if fac else (self.diam[e]**2)*self.h_network[e]
        v=(-self.D_eff[e]/self.h_network[e])*(self.diam[e]**2)/factor
        down=(self.D_eff[e]/self.h_network[e])*(self.diam[e]**2)/factor
        self.D[pos_net,pos_frw]+=down
        self.D[pos_net,pos_net]+=v
        print("from downstream diffusion: bif: {bif}, frwd ({pos_frw}): {forward}".format(bif=v, \
        pos_frw=pos_frw, forward=down))
        return()
        
    def UDConv(self, pos_net, e, *fac, **kk):
        """Upwind scheme for Downstream advective flux"""

        if "pos_fwd" in kk:
            pos_fwd=kk["pos_fwd"]
            factor=fac[0] if fac else (self.diam[e]**2)*self.h_network[e]
            down=np.max([-self.U_eff[e],0])*(self.diam[e]**2)/factor
            self.D[pos_net,pos_fwd]+=down
    
        else:
            e=self.s[pos_net,2] #if there is no downstream cell the edge is calculated with the 
                                #current cell
            factor=fac[0] if fac else (self.diam[e]**2)
        v=-np.max([self.U_eff[e],0])*(self.diam[e]**2)/factor
        self.D[pos_net,pos_net]+=v
        
        print("from downstream convection: bif: {bif}, frwd ({pos_frw}): {forward}".format(bif=v, \
        pos_frw=pos_fwd, forward=down))
        return()
        
#==============================================================================
#     def upwind_scheme_downstream_flux(self, pos_net,  **kwargs):
#         """When calling this function we need the position in the network (it must be thee). 
#         Then the optional arguments kwarts, which are:
#             """
#         #possible contents inside kargs => "diffusion", "convection", "pos_frw",   "factor"
#         j=kwargs.values()       
#         if "convection" in j:
#             UU=1
#         if "diffusion" in j:
#             DD=1        
#         if "pos_frw" in kwargs.keys(): #there is flux to downstream cell
#             pos_frw=kwargs["pos_frw"]
#             
#             factor=kwargs["factor"] if "factor" in kwargs.keys() else (2*self.diam[e]**2)
#             self.D[pos_net,pos_frw]+=(self.D_eff[e]/self.h_network[e]*DD+np.max([-self.U_eff[e],0])*UU)*(2*self.diam[e]**2)/factor
#         else: #no flux to downstream cell
#             e=self.s[pos_net,2]
#             factor=kwargs["factor"] if "factor" in kwargs.keys() else (2*self.diam[e]**2)         
#         
#         self.D[pos_net,pos_net]+=(-self.D_eff[e]/self.h_network[e]*DD-np.max([self.U_eff[e],0])*UU)*(2*self.diam[e]**2)/factor
#         
#         return()
#==============================================================================
    
    def UUDiff(self, pos_net, e, pos_bcw, *fac):
        """Upwind scheme for Upstream Diffusive flux"""

        factor=fac[0] if fac else (self.diam[e]**2)*self.h_network[e]
        v=(-self.D_eff[e]*self.diam[e]**2)/(factor*self.h_network[e])
        up=(self.D_eff[e]*self.diam[e]**2)/(factor*self.h_network[e])
        self.D[pos_net,pos_net]+=v
        self.D[pos_net,pos_bcw]+=up
        print("from upstream diffusion: bif: {bif}, bck ({pos_bcw}): {backward}".format(bif=v, \
        pos_bcw=pos_bcw, backward=up))

        return()
        
    def UUCnov(self, pos_net, e, *fac, **kk):
        """Upwind scheme for Upstream advective flux"""
        if "pos_bcw" in kk:
            pos_bcw=kk["pos_bcw"]
            factor=fac[0] if fac else (self.diam[e]**2)*self.h_network[e]
            up=np.max([self.U_eff[e],0])*(self.diam[e]**2)/factor
            self.D[pos_net,pos_bcw]+=up
        else:
            factor=fac[0] if fac else (self.diam[e]**2)
        
        v=(-np.max([-self.U_eff[e],0]))*(self.diam[e]**2)/factor
        self.D[pos_net,pos_net]+=v
        print("from upstream convection: bif: {bif}, bck ({pos_bcw}): {backward}".format(bif=v, \
        pos_bcw=pos_bcw, backward=up))
        return()   
    
    def Coupl_net(self,pos_net, pos_tiss):
        K_eff=self.K_eff[self.s[pos_net, 2]]
        self.C[pos_net, pos_tiss]+=K_eff
        self.D[pos_net, pos_net]+=-K_eff
        return()
    

    def bifurcation_law(self, vertex):
        #For intersections
        #Flux east
        a=np.where(self.init==vertex) #Edges that have this vertex as beginning
        b=np.where(self.fin==vertex) 
        self.n,self.b=a,b
        factor=2/(self.diam[np.append(a,b)]**2).dot(self.h_network[np.append(a,b)])

        #print("The edges connected to vertex {vertex} are {vec}".format(vertex=vertex, vec=np.append(a,b)))
        pos_net_vertex=vertex-len(self.cordx)

        #we search the first vertex in the source vector. They are ordered, so the position
        #will be the addition of the length of all the values in the previous vertices
        for i in a[0]: #Goes through each of the EDGES that have "vertex" as initial vertex 
        #Therefore downstream flux:
            pos_s=np.sum(self.num_cells_edge[:i]) #position of the first (second) surface of the vessel

            self.UDDiff(pos_net_vertex, i, pos_s,  factor)

            self.UDConv(pos_net_vertex, i, factor,  pos_fwd=pos_s)
            
        for i in b[0]:
            #Therefore upstream flux:
            pos_s=np.sum(self.num_cells_edge[:i+1])-1 #position of the last surface of the vessel
            self.UUDiff(pos_net_vertex,i, pos_s, factor)
            self.UUCnov(pos_net_vertex,i, factor, pos_bcw=pos_s)
            
        #Coupling
        self.D[pos_net_vertex, pos_net_vertex]-=self.K_eff[pos_net_vertex]
        self.C[pos_net_vertex, self.s[pos_net_vertex, 1]]+=self.K_eff[pos_net_vertex]
        return()
        
#==============================================================================
#     def boundary_set(self, vertex, boundary_vector, position_in_vector):
#         #If boundary[position_in_vector]=1 -> Inlet, =2 -> Bifurcation, -> =3 Vein outlet
#         if boundary_vector[position_in_vector]==1:
#             self.D[position_in_vector,position_in_vector]=1
#         elif boundary_vector[position_in_vector]==3:
#             e=np.where[self.fin==vertex] #edge
#             pos_bcw=np.sum(self.num_cells_edge[:e])-1 #final point in source vector
#             #Upstream lfux            
#             self.D[position_in_vector,position_in_vector]+=(-self.D_eff[e]/self.h_network[e]-np.max([-self.U_eff[e],0]))
#             self.D[position_in_vector,pos_bcw]+=(self.D_eff[e]/self.h_network[e]+np.max([self.U_eff[e],0]))
#             #Downstream flux
#             self.D[position_in_vector,position_in_vector]+=np.max([self.U_eff[e],0])
#         return()
#         
#             
#==============================================================================
    def flux_int(self, pos_tis, pos_net, pos_net_east, pos_net_west, e):
        self.UDDiff(pos_net, e, pos_net_east)
        self.UDConv(pos_net, e, pos_fwd=pos_net_east)
        self.UUDiff(pos_net, e, pos_net_west)
        self.UUCnov(pos_net, e, pos_bcw=pos_net_west)
        self.Coupl_net(pos_net,pos_tis)
        return()

    def assembly(self):
        #Matrix creation        
        self.len_tissue=self.IC_tissue.size
        self.len_net=self.IC_vessels.size        
        self.lenprob=self.len_tissue+self.len_net
        self.a=np.zeros([self.len_tissue,self.len_tissue])   #MAKE SPARSE AFTERWARDS
        self.B=np.zeros([self.len_tissue, self.len_net])
        self.C=np.zeros([self.len_net,self.len_tissue])
        self.D=np.zeros([self.len_net,self.len_net])
        self.phi_tissue=self.IC_tissue #Vector with the unknowns
        self.phi_vessels=self.IC_vessels
#==============================================================================
#            [a    B]   [        ] [phi_t]
#            [      ] * [unknowns]=[     ] 
#            [C    D]   [        ] [phi_v]
#==============================================================================
        
        #coeffs for the equation in blood -----> SOURCES (MATRICES B, C and D)
        hn=self.h_network  #discretization step of the network 

        c=0 #counter 
        z=0 #other counter
        #loop for the coupling and the source
        for i in self.s: #Loop going through each of the network (of vessels) cells
            p=i[0] #Identifier for the discretized cell of the vessel (the first column has the ID of the cell)
            pos_net=int(p) #the position of this cell in the matrix Value of the cell at the vascular level
            pos_tis=int(i[1]) #value of the FV (tissue) cell
            self.pos_net=pos_net
            e=i[2] #Number of the edge this cell belongs to 
            
            if e>=0:
                if c==0: #first cell (after vertex) of vessel 
                    pos_vertex=self.len_net-len(self.cordx)+self.init[e] #Position of the vertex in the source vector 
                    pos_net_west=pos_vertex
                    pos_net_east=pos_net+1
                    c+=1
                elif c==self.num_cells_edge[e]-1: #Last cell before bifurcation/boundary vertex
                    pos_vertex=self.len_net-len(self.cordx)+self.fin[e]
                    pos_net_east=pos_vertex
                    pos_net_west=pos_net-1
                    c=0
                else: #Intermediate vertex
                    pos_net_east=pos_net+1
                    pos_net_west=pos_net-1
                    c+=1
                
                self.flux_int(pos_tis, pos_net, pos_net_east, pos_net_west, e)
            
            else: #if e is lower than zero it is a vertex encoded as e=-vertex-1
                vertex=-e-1
                if self.boundary[vertex]==2: #it is a vertex, it does not belong to an edge
                    self.bifurcation_law(vertex)
                elif self.boundary[vertex]==3: #Outflow
                    e=int(np.where(self.fin==vertex)[0])
                    self.UDConv(pos_net, e, pos_fwd=pos_net_east)
                    self.UUDiff(pos_net, e, pos_net_west)
                    self.UUCnov(pos_net, e, pos_bcw=pos_net_west)
                    self.Coupl_net(pos_net,pos_tis)
                else: #self.boundary==1 that means inlet
                    self.D[pos_net, pos_net]=0
                    
            #c should be equal to pos_net=int(i[0])
            self.z=z
            self.c=c
            
            z+=1
            
            #This two blocks are not indented because even the boundary nodes will diffuse to the tissue
            self.a[pos_tis,pos_tis]-=self.K_eff[e]*self.h_tot[e]/(self.hx*self.hy)
            self.B[pos_tis, pos_net]+=self.K_eff[e]*self.h_tot[e]/(self.hx*self.hy) 
             
            
            
        D=self.D_tissue #tissue diffusion coefficient
        hx,hy=self.hx,self.hy
        """This loop will go over every value of phi[0]. This values represent the indices
        of the vector itself. It would have been the same to do for i in range(len(phi[0])).
        The purpose of the loop is go through every cell (phi[0]), and determine if there is 
        a """       
        
        for i in self.t.loc[:,"ind_cell"]:
            if self.t.loc[i,"BCenc"]%10!=self.CN:
                #There is diff flux north
                self.a[i,i+self.xlen]+=D/hy**2
                self.a[i,i]-=D/hy**2
            if self.t.loc[i,"BCenc"]%100//10!=self.CS:
                #There is diff flux north
                self.a[i,i-self.xlen]+=D/hy**2
                self.a[i,i]-=D/hy**2
            if self.t.loc[i,"BCenc"]%1000//100!=self.CE:
                #There is diff flux north
                self.a[i,i+1]+=D/hx**2
                self.a[i,i]-=D/hx**2
            if self.t.loc[i,"BCenc"]//1000!=self.CW:
                self.a[i,i-1]+=D/hx**2
                self.a[i,i]-=D/hx**2

        return(self.a)