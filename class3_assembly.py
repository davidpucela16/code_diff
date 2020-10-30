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
               
        self.dim=parameters["dim"]
        self.hx,self.hy=parameters["hx"],parameters["hy"]
        self.x=np.arange(parameters["x"][0],parameters["x"][1]+self.hx,self.hx) 
        self.y=np.arange(parameters["y"][0],parameters["y"][1]+self.hy,self.hy)
        self.coeffs=parameters["coeffs"] 
        self.CN,self.CS,self.CE,self.CW=self.coeffs
        super().matrix(self.x,self.y)
        self.hx,self.hy=(parameters["hx"],parameters["hy"])
        self.h_network=parameters["h_network"]
        self.K_eff=Edges["eff_perm"]
        self.D_tissue=parameters["D_tissue"]
        #For the initial simulation 
        self.c_0=1
        self.t=parameters["t"] #vector with the IDs of the tissue
        self.M=parameters["linear_consumption"]
        self.D_eff=Edges["eff_diff"]
        self.U_eff=Edges["eff_velocity"]
        
        
        #index of source, ind cell it is in, Edge it belongs to 
        source=parameters["s"]
        IC_vessels=source["IC"]
        self.s=np.array([np.arange(len(source)),source["ind cell"],source["Edge"], IC_vessels]).T
        self.s=self.s.astype(int)
        self.dfs=source
        self.num_cells_edge=((Edges["length"]/self.h_network).values).astype(int)-1
        
        self.boundary=boundary
        self.diam=Edges["diameter"]
        self.init=init
        self.fin=fin
        
        
        #vessel
        #These two arrays should have as many values as edges in the network since the effective convective and dispersive 
        #coefficients are gonna be different in each vessel
        self.u_network=Edges["eff_velocity"] #effective velocity in the blood network 
        self.D_network=Edges["eff_diff"]
    
        
        
        self.IC_vessels=self.s[:,3]
        self.IC_tissue=np.ndarray.flatten(parameters["IC_tissue"])

    def upwind_scheme_downstream_flux(self, pos_net, pos_frw, factor):
        e=self.s[pos_frw,2]
        self.D[pos_net,pos_net]+=(-self.D_eff[e]/self.h_network[e]-np.max([self.U_eff[e],0]))*(2*self.diam[e]**2)/factor
        self.D[pos_net,pos_frw]+=(self.D_eff[e]/self.h_network[e]+np.max([-self.U_eff[e],0]))*(2*self.diam[e]**2)/factor
        return()
    
    def upwind_scheme_upstream_flux(self, pos_net, pos_bcw, factor):
        e=self.s[pos_bcw,2]
        self.D[pos_net,pos_net]+=(-self.D_eff[e]/self.h_network[e]-np.max([-self.U_eff[e],0]))*(2*self.diam[e]**2)/factor
        self.D[pos_net,pos_bcw]+=(self.D_eff[e]/self.h_network[e]+np.max([self.U_eff[e],0]))*(2*self.diam[e]**2)/factor
        return()
        
    
    def bifurcation_law(self, vertex):
        #For intersections
        #Flux east
        a=np.where(self.init==vertex) #Edges that have this vertex as beginning
        b=np.where(self.fin==vertex) 
        factor=(self.Edges.loc[np.append(a,b),"diameter"]**2).dot(self.h_network[np.append(a,b)])
        #we search the first vertex in the source vector. They are ordered, so the position
        #will be the addition of the length of all the values in the previous vertices
        for i in np.array([a]): #Goes through each of the EDGES that have "vertex" as initial vertex 
            pos_s=np.sum(self.num_cells_edge[:i]) #position of the first (second) surface of the vessel
            pos_net_vertex=-1-vertex
            self.upwind_scheme_downstream_flux(pos_net_vertex, pos_s, factor)
            
        for i in np.array([b]):
            pos_s=np.sum(self.num_cells_edge[:i+1])-1 #position of the last surface of the vessel
            pos_net_vertex=-1-vertex
            self.upwind_scheme_upstream_flux(pos_net_vertex, pos_s, factor)
            
        return()
        
    def boundary_set(self, vertex, boundary_vector, position_in_vector):
        #If boundary[position_in_vector]=1 -> Inlet, =2 -> Bifurcation, -> =3 Vein outlet
        if boundary_vector[position_in_vector]==1:
            self.D[position_in_vector,position_in_vector]=1
        elif boundary_vector[position_in_vector]==3:
            e=np.where[self.fin==vertex] #edge
            pos_bcw=np.sum(self.num_cells_edge[:e])-1 #final point in source vector
            #Upstream lfux            
            self.D[position_in_vector,position_in_vector]+=(-self.D_eff[e]/self.h_network[e]-np.max([-self.U_eff[e],0]))
            self.D[position_in_vector,pos_bcw]+=(self.D_eff[e]/self.h_network[e]+np.max([self.U_eff[e],0]))
            #Downstream flux
            self.D[position_in_vector,position_in_vector]+=np.max([self.U_eff[e],0])
        return()
        
    def flux_int(self, pos_tis, pos_net, pos_net_east, pos_net_west, e):
        #Central difference Scheme!!!!!!!
        hn=self.h_network[e]
        #Diffusive:
        self.D[pos_net,pos_net]-=2/(hn**2)*self.Edges.loc[e,"eff_diff"]
        self.D[pos_net,pos_net_east]+=1/(hn**2)*self.Edges.loc[e,"eff_diff"]
        self.D[pos_net,pos_net_west]+=1/(hn**2)*self.Edges.loc[e,"eff_diff"]
        #Convective
        self.D[pos_net,pos_net_west]-=self.Edges.loc[e,"eff_velocity"]/(2*hn)
        self.D[pos_net,pos_net_east]+=self.Edges.loc[e,"eff_velocity"]/(2*hn)
        
        self.D[pos_net, pos_net]-=self.K_eff
        self.C[pos_net,pos_tis]+=self.K_eff
        
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
        #loop for the coupling and the source
        for i in self.s: #Loop going through each of the network (of vessels) cells
            p=i[0] #Identifier for the discretized cell of the vessel (the first column has the ID of the cell)
            pos_net=int(p) #the position of this cell in the matrix Value of the cell at the vascular level
            pos_tis=int(i[1]) #value of the FV (tissue) cell
            self.pos_net=pos_net
            e=i[2] #Number of the edge this cell belongs to 
            N=self.num_cells_edge[e] #numbers of cells this edge has
            if e>=0:
                if c==0: #first cell (after vertex) of vessel 
                    pos_vertex=self.len_net-len(self.cordx)+self.init[e] #Position of the vertex in the source vector 
                    pos_net_west=pos_vertex
                    pos_net_east=pos_net+1
                    c+=1
                elif c==N-1: #Last cell before bifurcation/boundary vertex
                    pos_vertex=self.len_net-len(self.cordx)+self.fin[e]
                    pos_net_east=pos_vertex
                    pos_net_west=pos_net-1
                    c=0
                else: #Intermediate vertex
                    pos_net_east=pos_net+1
                    pos_net_west=pos_net-1
                    c+=1
                
                self.flux_int(self, pos_tis, pos_net, pos_net_east, pos_net_west, e)
            
            else: #if e is lower than zero it is a vertex encoded as e=-vertex-1
                vertex=-e-1
                if self.boundary[e]==2: #it is a vertex, it does not belong to an edge
                    self.bifurcation_law(vertex)
                else:
                    self.boundary_set(vertex, self.boundary, c)
            #c should be equal to pos_net=int(i[0])
            c+=1
            
            #This two blocks are not indented because even the boundary nodes will diffuse to the tissue
            self.a[pos_tis,pos_tis]-=self.K_eff*hn[e]/(self.hx*self.hy)
            self.B[pos_tis, pos_net]+=self.K_eff*hn[e]/(self.hx*self.hy)  
            
            
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