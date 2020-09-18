# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:14:42 2020

@author: david
"""


import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches

def plot(z,r, inc_z, inc_r, xlen, ylen):
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
    theta=np.isclose(C%10,6)
    v1=np.zeros(len(theta))
    v2=np.zeros(len(theta))
    c=0
    
    v1=Z[theta]
    v2=Rho[theta]
    for i in range(len(v2)):
        rect = mpatches.Rectangle((v1[i],v2[i]+inc_r/2),hx,hy,linewidth=1,edgecolor='r',facecolor='r')
        # Add the patch to the Axes
        ax.add_patch(rect)
        c+=1
    plt.grid()
    plt.savefig("vessels.pdf")
    plt.show()
    #plt.plot(v1,v2)



CN,CS,CE,CW,MN,MS=1,2,3,4,6,7

Dv=1
Dt=1
K_m=2
inc_r=0.1
inc_z=0.1

lamb=0.5
L=2
Rm=1
Rv=0.5
K=10
Mt=2

parameters={"Diff_vessel": Dv, "Diff_tissue": Dt, "Permeability": K_m, "inc_rho": inc_r, "inc_z": inc_z,
            "R_max": Rm, "R_vessel": Rv, "Length": L, "Tissue_consumption":Mt}


z=np.arange(0,L,inc_z) 
r=np.arange(0,Rm,inc_r) 
Z,Rho=np.meshgrid(z,r) #Coordinatees
rlen, zlen=Z.shape

b=int(np.nonzero(np.isclose(Rv,r))[0])+1

Vel=K*(Rv**2-r[:b+1]**2)

new={"vel_prof": W}
parameters.update(new)

phi=np.ndarray.flatten(C)
phi = phi.astype(np.int)
phi_v=phi[:(b*zlen)]




C=np.zeros(Z.shape)

C[0,:]+=10*S
C[-1,:]+=N
C[:,0]+=1000*W
C[:,-1]+=100*E
C[Rv/inc_r,:]=MN
C[Rv/inc_r+1,:]=MS








plot(z,r, inc_z, inc_r, xlen, ylen)



#FOR TISSUE
#if north is not a boundary 
cntrl-=Dt*((r[i]+inc_r/2)/(inc_r**2*r[i]))
n+=Dt*((r[i]+inc_r/2)/(inc_r**2*r[i]))

#if south is not a boundary
cntrl-=Dt*((r[i]-inc_r/2)/(inc_r**2*r[i]))
s+=Dt*((r[i]-inc_r/2)/(inc_r**2*r[i]))

#if east is not a boundary
cntrl-=Dt/inc_z**2
e+=Dt/inc_z**2

#if west is not a boundary
cntrl-=Dt/inc_z**2
w+=Dt/inc_z**2


zlen,rlen=Z.shape
Rvi=int(Rv/inc_r+1)

phi=np.ndarray.flatten(C)

phi_v=phi[:(zlen*Rvi)]

c=0
A=np.zeros((len(phi),len(phi)))
for i in phi:
    Diff_flux_north,Diff_flux_south,Diff_flux_east,Diff_flux_west,Diff_flux_membrane,tissue_consumption, \
    Conv_flux_east, Conv_flux_West=1,1,1,1,0,Mt,1,1
    if i//1000==CW:
        Diff_flux_west=0
        Conv_flux_west=0
    elif (i%1000)//100==CE:
        Diff_flux_east=0
    elif (i%100)//10==CS:
        Diff_flux_south==0
    elif (i%10)==CN:
        Diff_flux_north=0
    elif (i%10)==MN:
        Diff_flux_membrane=1
    elif (i%10)==MS:
        Diff_flux_membrane=2
    elif c<Rvi*zlen:
        tissue_consumtion=0
        
    #radial coordinate (int)
    ir=c//zlen
    #axial coordinate (int)
    iz=c%zlen
    
    if radius<=Rv:
        Mt=0
        Vel=parameters["vel_prof"]
        D=parameters["Diff_vessel"]
    elif radiu>Rv:
        Mt=parameters["Tissue_consumption"]
        Vel=0
        D=parameters["Diff_tissue"]
    else:
        #membrane
        D=0
        Vel=0
        Mt=0
        
    N,S,E,W,cntrl=0,0,0,0,0
    if Diff_flux_north:
        cntrl-=(r[ir]+inc_r)*D/(r[ir]*inc_r**2)
        N+=(r[ir]+inc_r)*D/(r[ir]*inc_r**2)
    elif Diff_flux_south:
        cntrl-=(r[ir]-inc_r)*D/(r[ir]*inc_r**2)
        S+=(r[ir]-inc_r)*D/(r[ir]*inc_r**2)
    elif Diff_flux_east:
        cntrl-=D/inc_z**2
        E+=D/inc_z**2 
    elif Diff_flux_west:
        cntrl-=D/inc_z**2
        W+=D/inc_z**2 
    elif Conv_flux_east:
        cntrl-=W[ir]/(2*inc_z)
        E-=W[ir]/(2*inc_z)
    elif Conv_flux_west:
        cntrl+=W[ir]/(2*inc_z)
        W+=W[ir]/(2*inc_z)
    elif tissue_consumption:
        cntrl+=Mt
        
    A[c,c]=cntrl        
    A[c,c+1]=E  
    A[c,c-1]=W
    A[c,c+zlen]=N
    A[c,c-zlen]=S
    
    if Diff_flux_membrane:
        if Diff_flux_membrane==1:
            #membrane in the north, tissue
            A[c,c+zlen]=K_m*lamb
            A[c,c]-=K_m
        elif Diff_flux_membrane==2:
            #membrane in the south 
            A[c,c-zlen]=K_m
            A[c,c]-=K_m*lamb
    
    c+=1

    
    


