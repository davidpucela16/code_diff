# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 17:38:35 2020

@author: david
"""


#DIFF EQUATION
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import random
import pandas as pd

    

class Grid():
    
    def __init__(self,parameters, Network, Edges,  cords, edge):

        self.domain_x=parameters["domain_x"]
        self.domain_y=parameters["domain_y"]
        self.dim=parameters["dim"]
        self.hx=parameters["hx"]
        self.hy=parameters["hy"]
        self.h_network=parameters["h_network"]
        
        self.dim=parameters["dim"]
        self.coeffs=parameters["coeffs"]
        self.init=edge[0,:]
        self.fin=edge[1,:]
        self.cordx=cords[0,:]
        self.cordy=cords[1,:]        

        self.Edges=Edges
        self.Network=Network
        #The two vectors of coordinates that will be used throughout the problem
        self.x=np.arange(parameters["start_x"],parameters["start_x"]+self.domain_x+self.hx,self.hx) 
        self.y=np.arange(parameters["start_y"],parameters["start_y"]+self.domain_y+self.hy,self.hy) 
        x=self.x
        y=self.y
        
        self.h=np.zeros(len(Edges)) #Each edge will have a different discretization size so we store it in this vector
        

        #To initialize all the matrices: coordinates, cell centers, t, etc
        self.matrix(x,y)
        
        
        
    def matrix(self,x,y):
        """This is a very important function that you will come back to a lot of times.
        The 'empty' matrix is created here. The thing is that coordinate centers do not match 
        the grid coordinates. Therefore, different matrices will be created for this task.
        X and Y represent the coordinates of the lower cell face, therefore the matrix Coord represents
        the cell centers just by adding half of the discretization length
        
        The function is supposed to be call in __init__ because it is the foundation of the 
        work domain"""        
        
        #Grid
        self.X,self.Y=np.meshgrid(x[:-1],y[:-1])
        
        #Coordinates of the cell centers:
        self.Cord=np.array([self.X+0.5*self.hx,self.Y+0.5*self.hy])
        C=np.zeros(self.Cord.shape)   #Matrix to store the solution
        _,self.ylen,self.xlen=np.shape(self.Cord)        
        self.C=C 
        phi1=np.arange(C[0].size)  #Array carrying the ID of each cell
        self.t=pd.DataFrame(phi1, columns=["ind_cell"]) #Array carrying the ID of each cell
        
        #Boundary encryption 
        CN,CS,CE,CW=self.coeffs
        
        B=np.zeros([self.xlen, self.ylen])
        B[0,:]+=10*CS
        B[-1,:]+=CN
        B[:,0]+=1000*CW
        B[:,-1]+=100*CE
        
        B=np.ndarray.flatten(B)
        self.t["BCenc"]=B
        
        
        return()
        
    
    
    def parametrize(self):
        """This function will find the closes cell-center for each of the (discretized)
        point of the vessel line s
        L -> Length of the vessel
        h -> approximate size of cell edge
        Cord -> the matrix of the grid coordinates of the cell centers
        lamb -> vector used to parametrize the vessel
        t -> vector of sources. Each cell of each vessel will be here stored (hence every source term), with the label 
             of the edge (vessel) it belongs to and the cell where it is diffusing mass"""
        h_network=self.h_network   #Size of the discretizatin 
        Cord=self.Cord   #Matrix with cell centers
        self.Cordx=Cord[0,:,:]
        self.Cordy=Cord[1,:,:]             
        self.c=0     #counter
        L=self.Edges["length"] #vector with the lengths of each edge
        
        
        #Data frame for the source term. It will store at the index position the identifier of the source discrete "volume" (equivalent
        #to the cell number identifier for the FV mesh). The data stored will include the cell that segment is in and the edge it belongs to.        
        self.s=pd.DataFrame(columns=["ind cell","Edge"]) 
        #fig=plt.figure()
        #ax=fig.gca()
        #ax.set(xlim=(self.x[0],self.x[-1]),ylim=(self.y[0],self.y[-1]))
        for i in range(len(L)):  #loop that goes through each EDGE
            lamb=self.Edges.loc[i,"unit vector"]  #unit vector for this specific edge
            s=np.linspace(0,L[i],int(L[i]//h_network)) #parametrization coordinate for this vessel i 
            s=s[1:-1] #We discount the first and last surfaces because those are the vertex (bifurcation or boundary node).
            self.h[i]=s[1]-s[0] #real cellule size of the edge i
            p0=(self.cordx[self.init[i]], self.cordy[self.init[i]])    #initial point of the edge (i.e coordinates of the initial vertex of the edge)
            
            
            
            for j in s:
                px,py=p0+lamb*j #cartesian coordinates of the center of the segment (of size h_network) 
                mx=self.Cordx-px
                my=self.Cordy-py
                ind=np.argmin((mx**2+my**2)) #index of the cell this segment is in 
                #plt.plot(px,py,'ro')
                
                #This next variable is quite crucial. It is the Pd data frame that will be used
                #Later on for the solver. The variable is created in this function, hence
                #if this function were not to be called there would be no source
                self.s=self.s.append(pd.DataFrame([[ind,i]], columns=self.s.columns), ignore_index=True) 
        
        #plt.show()
                
        self.theta=np.unique(self.s["ind cell"])
        
        for i in range(len(self.cordx)):
            px,py=self.cordx[i],self.cordy[i]
            mx=self.Cordx-px
            my=self.Cordy-py
            ind=np.argmin((mx**2+my**2)) #index of the cell this segment is in 
            #plt.plot(px,py,'ro')
            
            #This next variable is quite crucial. It is the Pd data frame that will be used
            #Later on for the solver. The variable is created in this function, hence
            #if this function were not to be called there would be no source
            self.s=self.s.append(pd.DataFrame([[ind,-i-1]], columns=self.s.columns), ignore_index=True) 
            #The vertices are encoded as edge -1-numero of vertex. Therefore vertex 0 will be the edge -1, vertex 2-> edge -3 etc.
        
        return(self.s)
        
    def plot_vessels(self):
        b=self.b
        o,_,_=b.shape
        fig=plt.figure()
        ax=fig.gca()
        ax.set(xlim=(self.x[0],self.x[-1]),ylim=(self.y[0],self.y[-1]))

        for i in range(o):
            c=b[i]
            plt.plot([c[0,0],c[0,1]],[c[1,0],c[1,1]]) 
        plt.grid()
        plt.savefig("vessels1.pdf")
        plt.show()
        
        
    def plot(self):
        fig=plt.figure()
        ax=fig.gca()
        ax.set(xlim=(self.x[0],self.x[-1]),ylim=(self.y[0],self.y[-1]))
        ax.set_xticks(self.x)
        ax.set_yticks(self.y)
        self.parametrize()
        
        #theta=np.concatenate((self.theta,self.theta+1,self.theta-1,self.theta+self.xlen,self.theta-self.xlen))
        #self.theta=theta
        # Create a Rectangle patch
        xlen,ylen=self.xlen,self.ylen
        hx,hy=self.hx,self.hy
       
        
        #Test
        theta=self.theta
        v1=np.zeros(len(theta))
        v2=np.zeros(len(theta))
        c=0
        for i in theta:
            irow=int(i//xlen)
            icol=int(i%xlen)
            
            v1[c]=self.x[int(icol)]
            v2[c]=self.y[int(irow)]
            rect = mpatches.Rectangle((v1[c],v2[c]),hx,hy,linewidth=1,edgecolor='r',facecolor='r')
            # Add the patch to the Axes
            ax.add_patch(rect)
            c+=1
        plt.grid()
        plt.savefig("vessels.pdf")
        plt.show()
        #plt.plot(v1,v2)
        
# =============================================================================
#         for i in theta:
#             coord=(int(i//xlen),int(i%xlen)) #This is the index of the cell-center
#             rect = mpatches.Rectangle((self.x[coord[0]],self.y[coord[1]]),hx,hy,linewidth=1,edgecolor='r',facecolor='r')
#             # Add the patch to the Axes
#             ax.add_patch(rect)
# =============================================================================


        Edges=self.Edges
        Network=self.Network
        dim=self.dim
        b=np.empty([Edges["vertices"].size,dim,2])
        c=0
        for i in Edges["vertices"]:
            for j in range(2):    
                b[c,:,j]=Network["coordinates"][i[j]]
            c+=1
            
        
        fig=plt.figure()
        ax=fig.gca()
        ax.set(xlim=(self.x[0],self.x[-1]),ylim=(self.y[0],self.y[-1]))
        ax.set_xticks(self.x)
        ax.set_yticks(self.y)
        o,_,_=b.shape
        for i in range(o):
            c=b[i]
            plt.plot([c[0,0],c[0,1]],[c[1,0],c[1,1]]) 

        plt.grid()
        plt.savefig("figure.pdf")
        plt.show()
        
       
        
        
   
        




            
        
        
        
        
        

    
    
    




