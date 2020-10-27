# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 20:41:14 2020
@author: david
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:14:42 2020
@author: david
"""


import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches


def plot(z,r, inc_z, inc_r,C,Z,Rho,Rv):
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
    theta=np.isclose(Rho+0.1,Rv,inc_r/2)
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

def plot_solution(solution, geom, iteration):
    zlen=geom["zlen"]
    rlen=geom["rlen"]
    Z=geom["Z"]
    Rho=geom["Rho"]
    print(Rho.shape)
    sol=np.reshape(solution,(rlen,zlen))
    C=sol
#==============================================================================
#     C=np.append(sol[::-1,:],sol,axis=0)
#     Z=np.append(Z[::-1,:],Z,axis=0)
#     Rho=np.append(Rho[::-1,:],Rho,axis=0)
#==============================================================================
    print(Rho.shape)
    plt.figure
    #To fix the amount of contour levels COUNTOUR LEVELS
    limit=np.ceil(np.max(sol)-np.min(sol))
    breaks=np.linspace(0,np.max(sol),10)

    
    plt.contourf(Z,Rho,C,breaks)
    plt.colorbar(ticks=breaks, orientation="vertical")

    
    print("minimum: ", np.min(sol))
    print("max: ", np.max(sol))
    
    
    plt.xlabel("z")
    plt.ylabel("r")
    plt.grid()
    plt.title("Coupled diffusion")
    plt.savefig("solution.pdf")
    plt.show()
    
#==============================================================================
#     plt.figure
#     plt.imshow(C)
#     plt.colorbar
#==============================================================================
    
    
    
    return()
    
    
class domain():
    def __init__(self,z,r, coeffs, parameteres, typ):
        self.typ=typ
        CN,CS,CE,CW=coeffs
        self.Z,self.Rho=np.meshgrid(z,r)
        rlen, zlen=self.Z.shape
        C=np.zeros(self.Z.shape)
        
        C[0,:]+=10*CS
        C[-1,:]+=CN
        C[:,0]+=1000*CW
        C[:,-1]+=100*CE
        phi=np.ndarray.flatten(C)
        
        self.C=C.astype(np.int)
        self.z=z
        self.r=r
        self.rlen=rlen
        self.zlen=zlen
        
        phi=np.ndarray.flatten(self.C)
        self.phi=phi
        
        K=parameters["coeff_vel"]
        Rv=parameters["R_vessel"]
        self.Rv=Rv
        self.vel=np.around(K*(Rv**2-self.r**2))
        self.vel[np.where(self.vel<0)]=0
        
        self.inc_r, self.inc_z=parameters["inc_rho"], parameters["inc_z"]
        
        if self.typ=="vessel":
            self.D=parameters["Diff_vessel"]
        elif self.typ=="tissue":
            self.D=parameters["Diff_tissue"]
        else:
            print("you introduced the type wrong but because I dont know yet how to do exceptions you have to re run the code yourself")
        
        
    
        
CN,CS,CE,CW=1,2,3,4
coeffs=(CN,CS,CE,CW)

Dv=2
Dt=1
K_m=1
inc_r=0.1
inc_z=0.1

lamb=0.5
L=5
Rm=6 #MAX RADIUS
Rv=2
K=0.1
Mt=0.5
inlet_c=1

parameters={"Diff_vessel": Dv, "Diff_tissue": Dt, "Permeability": K_m, "inc_rho": inc_r, "inc_z": inc_z,
            "R_max": Rm, "R_vessel": Rv, "Length": L, "Tissue_consumption":Mt, "coeff_vel":K, "inlet_concentration":inlet_c}


z=np.arange(0,L,inc_z) 
rv=np.arange(inc_r/2,Rv,inc_r) 
rt=np.arange(np.max(rv)+inc_r,Rm,inc_r) 
#total radius
r=np.append(rv,rt)

vessel=domain(z,rv,coeffs, parameters, "vessel")
tissue=domain(z,rt,coeffs, parameters, "tissue")


#new={"vel_prof": Vel}
#parameters.update(new)

#plot(z,r, inc_z, inc_r, len(z), len(r))



#   ASSEMBLY VESSEL MATRIX
def assembly(obj):
    c=0
    inc_r, inc_z=obj.inc_r, obj.inc_z
    
    main=np.zeros((len(obj.phi),len(obj.phi)))
    if obj.typ=="tissue":
        coupl=np.zeros((len(obj.phi),len(vessel.phi)))
    elif obj.typ=="vessel":
        coupl=np.zeros((len(vessel.phi),len(tissue.phi)))
    
    for i in obj.phi:
        D=obj.D
        Vel=obj.vel
        Diff_flux_north,Diff_flux_south,Diff_flux_east,Diff_flux_west, \
        Conv_flux_east, Conv_flux_west=1,1,1,1,1,1
        #axial coordinate (int)    
        iz=c%obj.zlen
        #radial coordinate (int)
        ir=c//obj.zlen
        
        if obj.typ=="tissue":
            irr=len(vessel.r)+ir #this is the actual value on the radius column 
        
        N,S,E,W,cntrl=0,0,0,0,0
        
        if (i%1000)//100==CE:
            Diff_flux_east=0
        if (i%100)//10==CS:
            Diff_flux_south=0
            if obj.typ=="tissue":
                coupl[c,vessel.zlen*(vessel.rlen-1)+iz]+=K_m*tissue.Rv/(tissue.r[ir]*inc_r)
                cntrl-=K_m*tissue.Rv/(tissue.r[ir]*inc_r)

        if (i%10)==CN:
            Diff_flux_north=0
            if obj.typ=="vessel":
                coupl[c,iz]+=K_m*vessel.Rv/(vessel.r[ir]*inc_r)
                cntrl-=K_m*vessel.Rv/(vessel.r[ir]*inc_r)
        if i//1000==CW and obj.typ=="vessel":
            Diff_flux_west=0
            Conv_flux_west=0
            Diff_flux_north=0
            Diff_flux_south=0
            Diff_flux_east=0
            Conv_flux_east=0
            

        if i//1000==CW and obj.typ=="tissue":
            Diff_flux_west=0
            
        
        if Diff_flux_north:
            cntrl-=(obj.r[ir]+inc_r/2)*D/(obj.r[ir]*inc_r**2)
            N=(obj.r[ir]+inc_r/2)*D/(obj.r[ir]*inc_r**2)
            main[c,c+obj.zlen]+=N
        if Diff_flux_south:
            cntrl-=(obj.r[ir]-inc_r/2)*D/(obj.r[ir]*inc_r**2)
            S=(obj.r[ir]-inc_r/2)*D/(obj.r[ir]*inc_r**2)
            main[c,c-obj.zlen]+=S
        if Diff_flux_east:
            cntrl-=D/inc_z**2
            E=D/inc_z**2 
            main[c,c+1]+=E
        if Diff_flux_west:
            cntrl-=D/inc_z**2
            W=D/inc_z**2 
            main[c,c-1]+=W
        if Conv_flux_east:
            cntrl-=Vel[ir]/(inc_z)
        if Conv_flux_west:
            W=Vel[ir]/(inc_z)
            main[c,c-1]+=W
    
        main[c,c]=cntrl        
        #Coupling
        c+=1
        
        obj.main=main
        obj.coupl=coupl
        
    if obj.typ=="tissue":
        return(np.append(coupl,main, axis=1))
    elif obj.typ=="vessel":
        return(np.append(main, coupl, axis=1))
    else:
        print("wrong type introduced")
    

# =============================================================================
# #Superuseful to visualize the matrix of coefficients
# for i in vessel.main:
#     print("")
#     n=i.reshape(vessel.rlen, vessel.zlen)[::-1,:]
#     n=np.around(n,decimals=1)
#     print(n)
# =============================================================================

assembly(tissue)
assembly(vessel)

Up=np.append(vessel.main, vessel.coupl, axis=1)
Down=np.append(tissue.coupl, tissue.main, axis=1)


A=np.append(Up, Down, axis=0)
geom={"zlen":vessel.zlen,"rlen":(vessel.rlen+tissue.rlen),"Z":np.append(vessel.Z,tissue.Z,axis=0),\
      "Rho":np.append(vessel.Rho,tissue.Rho,axis=0)}
 
 
#Set Dirichlet inlet BCs:
vessel.phi[:]=0
i=0
inlet=parameters["inlet_concentration"]
while i<len(vessel.phi):
    vessel.phi[i]=inlet
    i+=vessel.zlen

tissue.phi=np.zeros(len(tissue.phi))
phi_total=np.append(vessel.phi, tissue.phi)
    
# =============================================================================
# 
# for i in range(Rvi+1):
#     #Setting Dirichlet BC
#     o=zlen*i
#     A[o,:]=0
#     A[o,o]=1
#     phi[o]=2
#     
# 
# geom={"zlen":zlen,"rlen":rlen,"Z":Z,"Rho":Rho}
# =============================================================================

inc_t=0.001
def fwdEuler(A,phi,inc_t):
    inc_phi=A.dot(phi)*inc_t
    return(phi+inc_phi)

def iterate(iterations, A, phi, inc_t,n):
    sol=fwdEuler(A,phi,inc_t)
    i=0
        
    for i in range(iterations):
        if i in range(0,iterations,n):
            print(i)
            plot_solution(sol, geom, i)
        sol=fwdEuler(A,sol,inc_t)
    return(sol)
        

a=iterate(10000, A, phi_total, inc_t, 500)
    