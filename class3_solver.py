import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import random



class Solver_t():
    
    
    
    def __init__(self, t, dim, hx, hy, x, y, xlen, ylen, source, X, Y,D_tissue,D_blood, K0, h_network, velocity, IC_vessels):
               
        self.dim=dim
        self.ylen,self.xlen=X.shape
        self.x=x
        self.y=y
        self.X=X
        self.Y=Y
        self.hx,self.hy=hx,hy
        self.h_network=h_network
        self.K0=K0
        self.D_tissue=D_tissue
        #For the initial simulation 
        self.c_0=1
        self.t=t #vector with the IDs of the tissue
        
        #index of source, ind cell it is in, Edge it belongs to 
        self.s=np.array([np.arange(len(source)),source["ind cell"],source["Edge"], IC_vessels]).T
        self.s=self.s.astype(int)
        self.L=np.max(self.s.shape)*h_network
        self.Dirichlet=0
        
        #vessel
        self.u=velocity   #velocity in the blood network     
        self.D_blood=D_blood  
        
        
            
#==============================================================================
#     def solve_linear_syst(self):
#         sol=0
#         err=1
#         self.it=0
#         while self.it<0.1:
#             sol_ant=sol
#             sol=np.linalg.solve(self.assembly(),self.b)
#             err_tot=np.sum(sol-sol_ant)
#             err=err_tot/len(err_tot)
#             self.it+=1
#             self.err=err
#         return(sol)
#==============================================================================
        
        
    def iterate_fweul(self, phi,inc_t):
        inc=self.A.dot(phi)*inc_t
        phi_plus_one=phi+inc
        return(phi_plus_one)
        
    def solve_lin_sys(self, max_it, inc_t):
        self.it=0
        sol=np.empty([max_it+1,self.lenprob])
        sol[0,:]=self.phi
        self. avg_err=10
        while self.it<max_it:
        #while self.it < max_it or err<1:
            sol[self.it+1,:]=self.iterate_fweul(sol[self.it,:],inc_t)
            err=np.sqrt((sol[self.it+1,:]-sol[self.it,:])**2)            
            self.avg_err=np.mean(err)            
            self.it+=1
        return(sol)
        
    def solve_lin_sys_SS(self):
        sol=np.linalg.solve(self.A,self.phi)
        return(sol)
            
            
    def plot_solution(self,phi):
        plt.figure
        #To fix the amount of contour levels COUNTOUR LEVELS
        phi=phi[:self.len_tissue]
        limit=np.ceil(np.max(phi)-np.min(phi))
        breaks=np.linspace(0,np.max(phi),10)
        
        C=np.reshape(phi,(self.ylen,self.xlen))
        self.C=C
        plt.contourf(self.X,self.Y,C,breaks, cmap="Reds")
        plt.colorbar(ticks=breaks, orientation="vertical")
        print("minimum: ", np.min(phi))
        print("max: ", np.max(phi))
        
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid()
        plt.title("Heat line source")
        plt.savefig("solution.pdf")
        return()
        
    def get_values(self,x,y):
        xpos=np.where(np.abs((self.x-x))==np.min(np.abs(self.x-x)))
        ypos=np.where((np.abs(self.y-y))==np.min(np.abs(self.y-y)))
        return(xpos,ypos)
        xpos=int(xpos[0][0])
        ypos=int(ypos[0][0])
        print("solution: ",self.C[xpos,ypos])
        return()
        
        
 
          

            
        
        