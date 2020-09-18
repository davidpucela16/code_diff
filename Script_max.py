# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:14:42 2020

@author: david
"""


import numpy as np
import matplotlib.pyplot as plt 

N,S,E,W=1,2,3,4

Dv=1
Dt=1
K_m=2
inc_r=0.1
inc_z=0.1

L=2
Rm=1
Rv=0.5
K=10

parameters={"Diff_vessel": Dv, "Diff_tissue": Dt, "Permeability": K_m, "inc_rho": inc_r, "inc_z": inc_z,
            "R_max": Rm, "R_vessel": Rv, "Length": L}


z=np.arange(0,L,inc_z) 
r=np.arange(0,Rm,inc_r) 

w=K*(Rv**2-r**2)

Z,Rho=np.meshgrid(z,r) #Coordinatees

xlen, ylen=Z.shape

C=np.zeros(Z.shape)

C[0,:]+=10*S
C[-1,:]+=N
C[:,0]+=1000*W
C[:,-1]+=100*E

C[np.isclose(Rho,Rv)]+=5


phi=np.flatten(C)


def plot(z,r,w, inc_z, inc_r, xlen, ylen):
    fig=plt.figure()
    ax=fig.gca()
    ax.set(xlim=(z[0],z[-1]),ylim=(r[0],r[-1]))
    ax.set_xticks(z+inc_z)
    ax.set_yticks(r+inc_r)
    
    #theta=np.concatenate((self.theta,self.theta+1,self.theta-1,self.theta+self.xlen,self.theta-self.xlen))
    #self.theta=the
    # Create a Rectangle patch
    hx,hy=inc_z, inc_r
       
    
    #Test
    theta=np.isclose(C%10,5)
    v1=np.zeros(len(theta))
    v2=np.zeros(len(theta))
    c=0
    for i in Ztheta:
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





