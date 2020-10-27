# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:24:51 2020

@author: pdavid
"""

import numpy as np
import matplotlib.pyplot as plt


class kk():
    def __init__(self, velocity):
        self.vel=self.effective(velocity)
        
    
    def effective(self, vell):
        return(np.array(vell)**2)
        
        
        
        
tr=kk([0,5,4])

print(tr.vel)





