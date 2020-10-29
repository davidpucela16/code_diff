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
        self.s=np.array([np.arange(len(source)),source["ind cell"],source["Edge"], IC_vessels, source["bifurcation?"]]).T
        self.s=self.s.astype(int)
        self.dfs=source
        self.L=np.max(self.s.shape)*self.h_network
        #Boundary info:
        self.boundary_vessel(source)
        
        
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
        Db=self.D_eff    #diffusion coefficient of blood
        Ub=self.U_eff

        
        
        #loop for the coupling and the source
        for i in self.s: #Loop going through each of the network (of vessels) cells
            p=i[0] #Identifier for the discretized cell of the vessel (the first column has the ID of the cell)
            pos_BD=int(p) #the position of this cell in the matrix Value of the cell at the vascular level
            pos_aC=int(i[1]) #value of the FV (tissue) cell
            self.i=i
            self.pos_BD=pos_BD
            n=i[4]
            e=i[2] #Number of the edge this cell belongs to 
            if n==1: #this belongs to the Bifurcation
                self.D[p,p]=1
                self.phi_vessels[p]=self.BC_vessels[1] #the gradient of concentration is fixed
            elif n
                self.D[p,p]+=1
                self.phi_vessels[p]=self.BC_vessels[0] #the flux for this unknown is fixed
                self.D[p,p-1]-=1
            elif n==-1 and Ub[e]>0:
                #There are both fluxes east and west, diffusive and convective
                #Flux east
                self.D[p,p-1]+=(1/hn[e])*(Ub[e]+Db[e]/hn[e]) #edge/vessel. The edge/vessel is given by the third column of the source term (self.s)
                self.D[p,p]-=Db/(hn[e]**2)
                #Flux west
                self.D[p,p+1]+=Db[e]/(hn[e]**2)#the coefficients depend strongly on the velocity, and the velocity is given by the 
                self.D[p,p]-=(Ub[e]/hn[e])+Db/(hn[e]**2)
                #Now, for the coupling part of the matrix (matrices B and C)
#==============================================================================
#           in case there is need for an imposed linear decay             
#              else:
#                 self.A[pos_A,pos_A]=1
#                 self.phi[pos_A]=(self.BC_vessels[1]-self.BC_vessels[0])*p/len(self.s) #the result for this unknown is fixed    
#==============================================================================
                
                #This two blocks are not indented because I do not want to change the coefficients of the matrix
                #if we are dealing with the boundary cells of the network
                
                self.D[p, p]-=self.K0
                self.C[p,pos_aC]+=self.K0
                
    
            #This two blocks are not indented because even the boundary nodes will diffuse to the tissue
            self.a[pos_aC,pos_aC]-=self.K0*hn/(self.hx*self.hy)
            self.B[pos_aC, p]+=self.K0*hn/(self.hx*self.hy)  #i[1] represents the FV cell of tissue
            
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