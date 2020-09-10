# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:24:51 2020

@author: pdavid
"""

import numpy as np
import matplotlib.pyplot as plt


w=0.5
h=0.001
r=np.arange(-0.2-h,0.2,h)

def lap_cyl(r,y,h):
    frw=(h**-2)+1/(r*h*2)
    bck=(h**-2)-1/(r*h*2)
    cntr=np.zeros(len(r))+(h**-2)*-2
    
    BCK=np.append([0,0],r)*np.append([0,0],bck)
    FRW=np.append(r,[0,0])*np.append(frw,[0,0])
    CNTR=np.append(0,np.append(cntr,0))*np.append(0,np.append(r,0))
    
    lap=BCK+FRW+CNTR
    return(lap)

y=1/np.abs(r)
y2=np.exp(r)/np.abs(r)
y[np.where(y>1000)]=1000
y2[np.where(y2>1000)]=1000

plt.figure()
plt.plot(r,y)
plt.show()
plt.figure()
plt.plot(r,y2)
plt.show()

plt.figure()
plt.plot(r,y2-y)
plt.show()


M1=0.5
x=np.arange(0.01,1,0.01)
y1=M1*(x/(1+x))

u=np.arange(0.1,10,0.01)
p=0
s=u
for i in u:
    y2=i*r
    c=0
    for i in r:
        s[p]+=(y2[c]-y1[c])**2
    p+=1
        




y2=M2*x

plt.figure()
plt.plot(x,y, label="Michaelis Menten")
plt.plot(x,y2, label="linear")
plt.legend()





rv=0.05
rho=[0.1,0.15,0.2,0.25]

x=np.arange(0,0.25,0.01)






