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

class Assembly(Grid):
    
    
    def __init__(self, parameters):
               
        self.dim=parameters["dim"]
        self.hx,self.hy=parameters["hx"],parameters["hy"]
        self.x=np.arange(parameters["x"][0],parameters["x"][1]+self.hx,self.hx) 
        self.y=np.arange(parameters["y"][0],parameters["y"][1]+self.hy,self.hy)
        super().matrix(self.x,self.y)
        self.hx,self.hy=(parameters["hx"],parameters["hy"])
        self.h_network=parameters["h_network"]
        self.K0=parameters["Permeability"]
        self.D_tissue=parameters["D_tissue"]
        #For the initial simulation 
        self.c_0=1
        self.t=parameters["t"] #vector with the IDs of the tissue
        self.M=parameters["linear_consumption"]
        
        #index of source, ind cell it is in, Edge it belongs to 
        source=parameters["s"]
        IC_vessels=parameters["IC_vessels"]
        self.s=np.array([np.arange(len(source)),source["ind cell"],source["Edge"], IC_vessels]).T
        self.s=self.s.astype(int)
        self.L=np.max(self.s.shape)*self.h_network
        self.Dirichlet=0
        
        
        #vessel
        self.u=parameters["velocity"]   #velocity in the blood network     
        self.D_blood=parameters["D_blood"]  
        
        self.IC_vessels=self.s[:,3]
        self.BC_vessels=parameters["BC_vessels"] #MODIFY LATER
        self.IC_tissue=np.ndarray.flatten(parameters["IC_tissue"])
        self.BCn=parameters["BC_tissue"][0]
        self.BCs=parameters["BC_tissue"][1]
        self.BCe=parameters["BC_tissue"][2]
        self.BCw=parameters["BC_tissue"][3]
        
    def is_on_boundary(self,i): #Will only work on a structured mesh
        xlen,ylen=self.xlen,self.ylen
        if i//xlen==0:
            #south
            return(2)
        elif i//xlen==ylen-1:
            #north
            return(1)
        elif i%xlen==0:
            #West
            return(4)
        elif i%xlen==xlen-1:
            #East
            return(3)  
        else:
            return(0)
        
    def Dirichlet_BC(self,k):
        """This function is a mess, it just checks if the cell i is in the boundary
        and if it is, it checks which one is it, and updates the values of the independent
        vector b accordingly"""
        i=int(k)
        #Auxiliary variables
        self.Dirichlet+=1
        o=self.is_on_boundary(i)
        if o: #information on boundary 
            self.a[i,i]=1
            xpos,ypos=(i%self.xlen,i//self.xlen)
            if o==1: #south
                self.phi_tissue[i]+=self.BCn[xpos]
            elif o==2: #north
                self.phi_tissue[i]+=self.BCs[xpos]
            elif o==3: #Eost
                self.phi_tissue[i]+=self.BCe[ypos]
            elif o==4: #West
                self.phi_tissue[i]+=self.BCw[ypos]  
            return(self.phi_tissue[i])
        
    def Flux_BC(self,k):
        """This function is a mess, it just checks if the cell i is in the boundary
        and if it is, it checks which one is it, and updates the values of the matrix and
        the flux"""
        i=int(k)
        #Auxiliary variables
        o=self.is_on_boundary(i)
        if o: #information on boundary 
            self.a[i,i]=1
            xpos,ypos=(i%self.xlen,i//self.xlen)
            if o==1: #north
                self.a[i,i-self.xlen]-=1
                self.phi_tissue[i]+=self.BCn[xpos]*self.hy
            elif o==2: #south
                self.a[i,i+self.xlen]-=1
                self.phi_tissue[i]+=self.BCs[xpos]*self.hy
            elif o==3: #East 
                self.a[i,i-1]-=1
                self.phi_tissue[i]+=self.BCe[ypos]*self.hx
            elif o==4: #West
                self.a[i,i+1]-=1
                self.phi_tissue[i]+=self.BCw[ypos]*self.hx
            return(self.phi_tissue[i])
            
    def is_on_boundary_1D(self, k, v_len):
        """this function just evaluates if the cell is not on the boundary (return(0)), if it 
        is at the beginning of the vessel (return(1), or at the end (return(2)))"""
        if k==0:
            return(1)
        elif k==v_len-1:
            return(2)
        else:
            return(0)
    
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
        Db=self.D_blood    #diffusion coefficient of blood
        fwd=Db/hn**2-self.u/(2*hn)   #i+1 coefficient for the discretized adv-diff equation
        bcw=Db/hn**2+self.u/(2*hn)   #i-1 coefficient for the discretized adv-diff equation
        cntr=-2*Db/(hn**2)    #i coefficient for the disc....
        
        (self.fwd, self.bcw, self.cntr)=(fwd,bcw,cntr)
        
        
        #loop for the coupling and the source
        for i in self.s: #Loop going through each of the network (of vessels) cells
            p=i[0] #Identifier for the discretized cell of the vessel (the first column has the ID of the cell)
            pos_BD=int(p) #the position of this cell in the matrix
            pos_aC=int(i[1])
            self.i=i
            self.pos_BD=pos_BD
            n=self.is_on_boundary_1D(p, self.len_net)
            if n==1: #this belongs to the initial boundary of the vessel
                self.D[p,p]=1
                self.phi_vessels[p]=self.BC_vessels[1] #the gradient of concentration is fixed
            elif p==self.len_net-1:
                self.D[p,p]+=1
                self.phi_vessels[p]=self.BC_vessels[0] #the flux for this unknown is fixed
                self.D[p,p-1]-=1
            else:
                self.D[p,p+1]=fwd #the coefficients depend strongly on the velocity, and the velocity is given by the 
                self.D[p,p-1]=bcw #edge/vessel. The edge/vessel is given by the third column of the source term (self.s)
                self.D[p,p]+=cntr
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
        #Factors:
        #linear, homogeneous diffusion coefficients:
        N=S=D/hx**2
        E=W=D/hx**2
        C=-4*D/hx**2-self.M
        """This loop will go over every value of phi[0]. This values represent the indices
        of the vector itself. It would have been the same to do for i in range(len(phi[0])).
        The purpose of the loop is go through every cell (phi[0]), and determine if there is 
        a """       
        for i in self.t: #t is the vector with the IDs of each FV cell
            i=int(i)
            #BCs:
            if self.is_on_boundary(i): #Check if the FV cell i is on the boundary
                                       #Will only work on a structured mesh 
                b=0
                self.Dirichlet_BC(i)
            else:
                #No boundary 
                b=1
                self.a[i,i]+=C
                self.a[i,i+1]+=E
                self.a[i,i-1]+=W
                self.a[i,i+self.xlen]+=S
                self.a[i,i-self.xlen]+=N

        return(self.a)