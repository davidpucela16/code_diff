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
    
    
    def __init__(self, parameters, Edges):
               
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
        self.num_cells_edge=((Edges["length"]/h_network).values).astype(int)
        
        
        
        
        #vessel
        #These two arrays should have as many values as edges in the network since the effective convective and dispersive 
        #coefficients are gonna be different in each vessel
        self.u_network=Edges["eff_vel"] #effective velocity in the blood network 
        self.D_network=Edges["eff_diff"]
    
        
        
        self.IC_vessels=self.s[:,3]
        self.BC_vessels=parameters["BC_vessels"] #MODIFY LATER
        self.IC_tissue=np.ndarray.flatten(parameters["IC_tissue"])
        self.BCn=parameters["BC_tissue"][0]
        self.BCs=parameters["BC_tissue"][1]
        self.BCe=parameters["BC_tissue"][2]
        self.BCw=parameters["BC_tissue"][3]
        
    
    def bifurcation_law(self):
        return()
        
    def flux_int(self, pos_tis, pos_net, pos_net_east, pos_net_west, e):
        hn=self.h_network[e]
        #Diffusive:
        self.D[pos_net,pos_net]-=2/(hn**2)*self.Edges.loc[e,"eff_diff"]
        self.D[pos_net,pos_net_east]+=1/(hn**2)*self.Edges.loc[e,"eff_diff"]
        self.D[pos_net,pos_net_west]+=1/(hn**2)*self.Edges.loc[e,"eff_diff"]
        #Convective
        self.D[pos_net,pos_net_west]-=self.Edges.loc[e,"eff_velocity"]/(2*hn)
        self.D[pos_net,pos_net_east]+=self.Edges.loc[e,"eff_velocity"]/(2*hn)
        
        self.D[pos_net, pos_net]-=self.K_eff
        self.C[pos_net,pos_tis]+=self.K0
        
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
        
            #This two blocks are not indented because even the boundary nodes will diffuse to the tissue
            self.a[pos_tis,pos_tis]-=self.K0*hn[e]/(self.hx*self.hy)
            self.B[pos_tis, pos_net]+=self.K0*hn[e]/(self.hx*self.hy)  
            
            
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