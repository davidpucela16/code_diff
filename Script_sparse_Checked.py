

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:14:42 2020
@author: david
"""


import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import scipy as sp
import scipy.sparse.linalg

def assembly_sparse(obj, geom, coeffs):
    #   ASSEMBLY VESSEL MATRIX
    #def assembly_sparse(obj, geom):
    K_m=obj.K_m
    c=0
    inc_r, inc_z=obj.inc_r, obj.inc_z
    row_main=np.array([])
    col_main=np.array([])
    data_main=np.array([])
    row_c=np.array([])
    col_c=np.array([])
    data_c=np.array([])
    Rv=geom["Rv"]
    
    for i in np.ndarray.flatten(obj.C):
        #obj.phi is the one dimensional array with the information encoded for the fluxes
    
        D=obj.D
        Vel=obj.vel
        Diff_flux_north,Diff_flux_south,Diff_flux_east,Diff_flux_west, \
        Conv_flux_east, Conv_flux_west=1,1,1,1,1,1
        #axial coordinate (int)    
        iz=c%obj.zlen
        #radial coordinate (int)
        ir=c//obj.zlen
        v_r=geom["vessel_r"]
        t_r=geom["tissue_r"]
        z=geom["z"]
        lenves=len(v_r)*len(z)
        lentis=len(t_r)*len(z)
        
        if obj.typ=="tissue":
            irr=len(v_r)+ir #this is the actual value on the radius column 
        
        N,S,E,W,cntrl=0,0,0,0,0
        coupl_south, coupl_north=0,0
        
        if (i%1000)//100==CE:
            Diff_flux_east=0
        if (i%100)//10==CS:
            Diff_flux_south=0
            if obj.typ=="tissue":
                #this is the flux that goes through the membrane to the tissue
                coupl_south+=K_m[iz]*Rv/(t_r[ir]*inc_r)
                cntrl-=K_m[iz]*Rv/(t_r[ir]*inc_r)
    
        if (i%10)==CN:
            Diff_flux_north=0
            if obj.typ=="vessel":
                coupl_north+=K_m[iz]*Rv/(v_r[ir]*inc_r)
                cntrl-=K_m[iz]*Rv/(v_r[ir]*inc_r)
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
            N+=(obj.r[ir]+inc_r/2)*D/(obj.r[ir]*inc_r**2)
            
        if Diff_flux_south:
            cntrl-=(obj.r[ir]-inc_r/2)*D/(obj.r[ir]*inc_r**2)
            S+=(obj.r[ir]-inc_r/2)*D/(obj.r[ir]*inc_r**2)
            
        if Diff_flux_east:
            cntrl-=D/inc_z**2
            E+=D/inc_z**2
            
        if Diff_flux_west:
            cntrl-=D/inc_z**2
            W+=D/inc_z**2 
            
        if Conv_flux_east:
            cntrl-=Vel[ir]/(inc_z)
        if Conv_flux_west:
            W+=Vel[ir]/(inc_z)
            
        if E:
            row_main=np.append(row_main, c) 
            col_main=np.append(col_main, c+1)
            data_main=np.append(data_main, E)
        if W:
            row_main=np.append(row_main, c) 
            col_main=np.append(col_main, c-1)
            data_main=np.append(data_main, W)
        if N:
            row_main=np.append(row_main, c) 
            col_main=np.append(col_main, c+obj.zlen)
            data_main=np.append(data_main, N)
        if S:
            row_main=np.append(row_main, c) 
            col_main=np.append(col_main, c-obj.zlen)
            data_main=np.append(data_main, S)
        row_main=np.append(row_main, c) 
        col_main=np.append(col_main, c)
        data_main=np.append(data_main, cntrl)
        
        
        #Coupling
        
        
        if coupl_south:
            #the object must be the tissue
            row_c=np.append(row_c, c) 
            col_c=np.append(col_c, len(z)*(len(v_r)-1)+iz) 
            data_c=np.append(data_c, coupl_south)
        if coupl_north:
            #the object must be the vessel
            row_c=np.append(row_c, c) 
            col_c=np.append(col_c, iz) 
            data_c=np.append(data_c, coupl_north)
            
            
            
        c+=1     
        
    if obj.typ=="tissue":
        coupl=sp.sparse.csc_matrix((data_c, (row_c, col_c)), shape=(len(obj.phi),lenves), dtype='f')
        main=sp.sparse.csc_matrix((data_main, (row_main, col_main)), shape=(len(obj.phi), len(obj.phi)), dtype='f')
        obj.Down=sp.sparse.hstack([coupl, main])
        return(obj.Down)
        
    elif obj.typ=="vessel":
        coupl=sp.sparse.csc_matrix((data_c, (row_c, col_c)), shape=(len(obj.phi),lentis), dtype='f')
        main=sp.sparse.csc_matrix((data_main, (row_main, col_main)), shape=(len(obj.phi), len(obj.phi)), dtype='f')
        obj.Up=sp.sparse.hstack([main,coupl])
        return(obj.Up)

def plot_solution(solution, geom, t, Rv,K_m, *namefig, **l):
    
    zlen=geom["zlen"]
    rlen=geom["rlen"]
    Z=geom["Z"]
    Rho=geom["Rho"]
    
    sol=np.reshape(solution,(rlen,zlen))
    C=sol
    C=np.append(sol[::-1,:],sol,axis=0)
    Z=np.append(Z[::-1,:],Z,axis=0)
    Rho=np.append(-Rho[::-1,:],Rho,axis=0)
    
    #To fix the amount of contour levels COUNTOUR LEVELS
    lim=l["limF"] if l else np.max(sol)/2
    beg=l["lim0"] if l else 0
    breaks=np.linspace(beg,lim,10)
    
    print("minimum: ", np.min(sol))
    print("max: ", np.max(sol))
    
    #array positions where the permeability is not actually 0
    coupl_s=np.where(K_m!=0)
    
    fig=plt.figure(figsize=(15,15))
    gs=gridspec.GridSpec(3,2)
    ax=fig.add_subplot(gs[0,:])
    fig.suptitle("Contour and C and C_avg through the centerline", fontsize=14, fontweight="bold")
    eDam=np.max(K_m)*Rv/Dv
    Dat=(L*2/3)**2*Mt/(parameters["Inlet"]*Dt)
    epsilon=Rv/L
    ax.set_title("$\epsilon Da_m=$ {i},$Da_t=$ {j}, $\epsilon$= {ep}, t={o} s".format(i=eDam, j=Dat, ep=epsilon, o=t))    
    CS=ax.contourf(Z,Rho,C,breaks, levels=breaks)
    ax.plot([0,z[-1]],[Rv, Rv], 'r-')
    ax.plot([0,z[-1]],[-Rv, -Rv], 'r-')
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(CS, ticks=breaks, orientation="vertical", format='%.0e')
    cbar.ax.set_ylabel('concentration')
    ax.set_xlabel("z")
    ax.set_ylabel("r")
    ax.grid()
    

    kk=data_proc(solution, vessel, tissue)
    csa=kk.cross_section_average()[coupl_s]
    ax2=fig.add_subplot(gs[1,0])
    ax2.plot(vessel.z[coupl_s], csa/np.max(csa))
    ax2.set_ylabel("conc")
    ax2.set_xlabel("s")
   
    
    ax3=fig.add_subplot(gs[1,1])
    ax3.plot(vessel.z[coupl_s], kk.get_tissue_wall()[coupl_s]/np.max(csa))
    ax3.set_xlabel("s")
    
    ax4=fig.add_subplot(gs[2,:])
    pos=np.linspace(0,zlen-1,9).astype(int)
    r=geom["Rho"][:,1]
    pos_tissue=[np.where(r>Rv)][0][0]
    for i in pos:
        ax4.plot(sol[pos_tissue,i], r[pos_tissue], label='z={o}'.format(o=(i/zlen-0.33)*3))
    leg = ax4.legend();
    ax4.set_xlabel("concentration")
    ax4.set_ylabel("R")
    
    figname=namefig[0] if namefig else "solution_{time}s.pdf".format(time=t)
    print(figname)
    plt.savefig(figname)
    plt.show()
    

    

    plt.show()
    
    
    
# =============================================================================
#     vess=np.where(np.abs(Rho[:,0])<Rv)[0]
#     breaks2=np.linspace(0,1,10)
#     fig2=plt.figure()
#     gs2=gridspec.GridSpec(2,2)
#     ax4=fig2.add_subplot(gs2[0,:])
#     ax4.plot(vessel.z, kk.cross_section_average())
#     ax4.set_ylabel("conc")
#     ax4.set_xlabel("s") 
#     ax5=fig2.add_subplot(gs2[1,:])
#     CS2=ax5.contourf(Z[vess,:],Rho[vess, :],C[vess, :])
#     cbar = fig2.colorbar(CS2, ticks=breaks2, orientation="vertical")
#     ax5.set_ylabel("R")
#     ax5.set_xlabel("s") 
#     plt.show()
# =============================================================================



    
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
        C=np.zeros(self.Z.shape, dtype='f')
        self.K_m=parameters["Permeability"]
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
        self.vel=K*(Rv**2-self.r**2)
        self.vel[np.where(self.vel<0)]=0
        
        self.inc_r, self.inc_z=parameters["inc_rho"], parameters["inc_z"]
        
        if self.typ=="vessel":
            self.D=parameters["Diff_vessel"]
        elif self.typ=="tissue":
            self.D=parameters["Diff_tissue"]
        else:
            print("you introduced the type wrong but because I dont know yet how to do exceptions you have to re run the code yourself")
        
        
class solve_sys():
    
    def __init__(self, A, vessel, tissue, parameters, geom):

        self.A=A
        self.zlen=vessel.zlen
        self.v_rlen=vessel.rlen
        self.lent=len(tissue.phi)
        
        self.inlet=parameters["Inlet"]
        self.Mt=parameters["Tissue_consumption"]
        self.phit=tissue.phi
        self.phiv=vessel.phi
        self.phi_total=np.append(vessel.phi, self.phit)
        self.lim0=0
        self.limF=0.4
        self.Rv=parameters["R_vessel"] 
        self.geom=geom
        self.K_m=vessel.K_m
        self.it=0
        
    def set_Dirichlet_tissue_SS_sparse(self):
        zlen=self.zlen
        v_rlen=self.v_rlen
        
        a=np.array([np.arange(0, zlen*v_rlen,zlen), np.arange(0,zlen*v_rlen, zlen)])
        
        self.A[a[0,:],:]=0
        self.A[a[0], a[1]]=1
        phiv=np.zeros(len(self.phiv))
        phiv[a[0]]=self.inlet
        phit=np.zeros(len(self.phit))
        
        lenprob=len(phiv)+len(phit)
        b=np.concatenate((np.arange(zlen*v_rlen, lenprob,zlen), np.arange(zlen*(v_rlen+1)-1, lenprob,zlen),np.arange(lenprob-zlen,lenprob)))
        
        self.A[b,:]=0
        self.A[b,b]=1
        phit=np.zeros(len(phit))
        
        self.phi_total=np.concatenate((phiv, phit)).astype('f')
        
        #LINEAR CONSUMTION
        
        c=np.arange(v_rlen*zlen, lenprob)
        self.A[c,c]-=Mt
        
        self.A.eliminate_zeros()
        
    def set_Dirichlet(self, value):
        east=np.arange(self.v_rlen*self.zlen, len(self.phi_total),self.zlen)
        west=np.arange(self.v_rlen*self.zlen+self.zlen-1, len(self.phi_total), self.zlen)
        north=np.arange(len(self.phi_total)-self.zlen, len(self.phi_total))
        array_positions=np.concatenate((east,west, north))
        self.A[array_positions,:]=0
        self.A[array_positions, array_positions]=1
        self.phi_total[array_positions]=value


    def solve_sparse_SS(self, *string):
        self.sol_SS=sp.sparse.linalg.spsolve(self.A, self.phi_total)

        plot_solution(self.sol_SS, geom, 9999, self.Rv,self.K_m, string[0], lim0=self.lim0, limF=self.limF)
        
    def SS(self):
        #Set Dirichlet inlet BCs:
        self.phiv[:]=0 #vector where the encryption for the boundary conditions were written 
        A=self.A
        positions=np.arange(0, self.v_rlen*self.zlen, self.zlen)
        A[positions, :]=0
        A[positions, positions]=1
        self.phi_total[positions]=self.inlet
            
        #set linear consumtion
        for i in range(self.lent):
            k=self.v_rlen*self.zlen
            A[k+i,k+i]-=Mt
        
        a=np.linalg.solve(A, self.phi_total)
        self.phi_SS=a
    
    def iterate(self, A, phi, inc_t):
        return(inc_t*A.dot(phi)+phi)
    
    def call_plot_solution(self, phi, time, *string):
        plot_solution(phi, self.geom, time,  self.Rv, self.K_m,string[0],  lim0=self.lim0, limF=self.limF)
        
    def forward_Euler(self, inc_t, sim_time,freq, phi_initial):
        """freq is how many times per second we want a plot to appear"""
        
        ##I THINK THERE IS NOT TISSUE CONSUMTION SET HERE
        self.it=0
        inc_phi=self.A.dot(phi_initial)*inc_t
        phi=phi_initial+inc_phi
        self.it+=1
        m=int(1/(freq*inc_t))
        for i in np.arange(int(sim_time//inc_t))+1:
            phi=self.iterate(self.A, phi, inc_t)
            if m and not self.it%m:
                t=self.it*inc_t
                string="Solution_eDam{eDam}_Dat{Dat}_t={s}s.pdf".format(eDam=eDam, Dat=Dat, s=t)
                print(string)
                self.call_plot_solution(phi, t, string)
            self.it+=1
        self.phi_fwE=phi
        return(phi)
    

    
class data_proc():
    def __init__(self,phi, vessel, tissue):
        """phi should be the solution, that now we are gonna process"""
        self.v=phi[:vessel.rlen*vessel.zlen]
        self.t=phi[vessel.rlen*vessel.zlen:] 
        self.v_r=vessel.r
        self.z=vessel.z
        self.t_r=tissue.r
        
    def cross_section_average(self):
        self.v=self.v.reshape(len(self.v_r), len(self.z))
        average=np.zeros(len(self.z))
        for i in range(len(self.z)):
            average[i]=np.sum(self.v[:,i])
        
        self.average=average/len(self.v_r)
        return(self.average)
    
    def get_tissue_wall(self):
        self.C_t=self.t.reshape(len(self.t_r), len(self.z))
        return(self.C_t[0,:])   


    

# =============================================================================
# #Superuseful to visualize the matrix of coefficients
# for i in vessel.main:
#     print("")
#     n=i.reshape(vessel.rlen, vessel.zlen)[::-1,:]
#     n=np.around(n,decimals=1)
#     print(n)
# =============================================================================    

def plot_tissue(a, tissue, vessel):
    
    b=a[vessel.rlen*vessel.zlen:]
    tis=b.reshape(tissue.rlen, tissue.zlen)
    
    f=plt.figure()
    plt.contourf(tissue.Z, tissue.Rho, tis)
    plt.colorbar()
    plt.xlabel("S through centerline")
    plt.ylabel("r")
    plt.title("Concentration contour in the tissue")
    
#SCRIPT BEGINS HERE!!
    
CN,CS,CE,CW=1,2,3,4
coeffs=(CN,CS,CE,CW)
        
L=15
Rm=10 #MAX RADIUS
Rv=0.5
K=13
Mt=1e-4
inlet_c=1
epsilon=Rv/L

Dv=Dt=1 #They are of the same order


inc_r=0.05
inc_z=0.1
K_m=np.zeros(int(L/inc_z))
K_m[int(len(K_m)/3):-int(len(K_m)/3)]=5e-1


parameters={"Diff_vessel": Dv, "Diff_tissue": Dt, "Permeability": K_m, "inc_rho": inc_r, "inc_z": inc_z,
            "R_max": Rm, "R_vessel": Rv, "Length": L, "Tissue_consumption":Mt, "coeff_vel":K, "Inlet":inlet_c}


z=np.arange(0,L,inc_z) 
rv=np.arange(inc_r/2,Rv,inc_r) 
rt=np.arange(np.max(rv)+inc_r,Rm,inc_r) 
#total radius
r=np.append(rv,rt)

vessel=domain(z,rv,coeffs, parameters, "vessel")
tissue=domain(z,rt,coeffs, parameters, "tissue")

geom={"zlen":vessel.zlen,"rlen":(vessel.rlen+tissue.rlen),"Z":np.append(vessel.Z,tissue.Z,axis=0),\
      "Rho":np.append(vessel.Rho,tissue.Rho,axis=0)}


new={"Rv":vessel.Rv, "vessel_r": vessel.r, "tissue_r":tissue.r, "z":tissue.z }
geom.update(new)

assembly_sparse(tissue, geom, coeffs)
assembly_sparse(vessel, geom, coeffs)

A=sp.sparse.csc_matrix(sp.sparse.vstack([vessel.Up, tissue.Down]))

k=solve_sys(A,vessel ,tissue,  parameters, geom)
k.set_Dirichlet_tissue_SS_sparse()
eDam=np.max(K_m)*Rv/Dv
Dat=L**2*Mt/(parameters["Inlet"]*Dt)
epsilon=np.round(Rv/L, 2)
string="$\epsilon$ $Da_m$={i},$Da_t$= {j}, $\epsilon$= {ep}".format(i=eDam, j=Dat, ep=epsilon)
k.solve_sparse_SS("Prueba")




#==============================================================================
# ##THIS FUNCTION WAS USED TO DO THE ASSEMBLY OF THE MATRIX WITHOUT SPARSENESS
# def assembly_matrix_A(tissue, vessel):
#     assembly(tissue, vessel, tissue)
#     assembly(vessel, vessel, tissue)
#     
#     Up=np.append(vessel.main, vessel.coupl, axis=1)
#     Down=np.append(tissue.coupl, tissue.main, axis=1)
#     
#     
#     A=np.append(Up, Down, axis=0)
#     return(A)
# 
# A=assembly_matrix_A(tissue, vessel)
#  
# #Set Dirichlet inlet BCs:
# vessel.phi[:]=0 #vector where the encryption for the boundary conditions were written 
# i=0
# inlet=parameters["Inlet"]
# while i<len(vessel.phi):
#     vessel.phi[i]=inlet
#     i+=vessel.zlen
# 
# tissue.phi=np.zeros(len(tissue.phi))
# phi_total=np.append(vessel.phi, tissue.phi)
#     
# k=solve_sys(A, vessel, tissue, parameters, geom)
# k.set_Dirichlet(0)
# k.SS()
# plot_solution(k.phi_SS, geom, 9999, Rv, K_m, string, lim0=0, limF=0.1)
# 
#==============================================================================