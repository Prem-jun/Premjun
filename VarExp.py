#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 11:23:26 2022

@author: premjunsawang
"""

from numpy import random
import numpy as np
import myfun, math
import scipy.stats as stats
import matplotlib.pyplot as plt
import statistics
numint=3
numpt = 1000
# x = random.normal(loc=1, scale=4, size=(numpt, 1))
mean = np.array([0,0,0])
cov = np.array([[1, 0,0], [0,10, 0],[0,0,100]])
nVar = len(mean)
x = np.random.multivariate_normal(mean, cov, numpt)
xx = np.zeros_like(x)
for i in range(numint):
    idx_del = random.randint(0,len(x)-1) 
    xx[i]=x[idx_del]
    numpt-=1
    x=np.delete(x,idx_del,axis = 0)

xx_cen = np.mean(xx[0:numint],axis=1)
xx_std = np.sqrt(np.var(xx[0:numint],axis=1))
ni = numint
std_dif = np.zeros_like(x)
for i in range(len(x)-1):
    idx_del = random.randint(0,len(x)-1)
    xx_cen, ni = myfun.update_center(xx_cen,x[idx_del],ni)
    xx_std = myfun.update_sd(np.array([x[idx_del]]),xx_cen,3,xx_std,ni-1,math,np)
    x=np.delete(x,idx_del,axis = 0)
    chi2_left = stats.chi2.ppf(0.95,df=ni-1)
    chi2_right = stats.chi2.ppf(0.05,df=ni-1)
    var_temp = ((ni-1)*xx_std**2)
    var_lower = var_temp/chi2_left
    var_upper = var_temp/chi2_right
    var_mean = (var_lower+var_upper)/2
    std_dif[i,] = np.abs(xx_std-np.sqrt(var_mean))
# mean = np.mean(x)
# xx =np.zeros(numint)
# for i in range(numint):
#     idx_del = random.randint(0,len(x)-1) 
#     xx[i]=x[idx_del]
#     numpt-=1
#     x=np.delete(x,idx_del)
# xx_cen = np.mean(xx)
# xx_cen = np.array([xx_cen])
# xx_std = np.sqrt(np.var(xx))
# xx_std = np.array([xx_std])
# ni = numint
# std_dif = np.zeros(len(x))
# for i in range(len(x)-1):
#     idx_del = random.randint(0,len(x)-1)
#     xx_cen, ni = myfun.update_center(xx_cen,x[idx_del],ni)
#     xx_std = myfun.update_sd(np.array([[x[idx_del]]]),xx_cen,1,xx_std,3,math,np)
#     x=np.delete(x,idx_del)
#     chi2_left = stats.chi2.ppf(0.95,df=ni-1)
#     chi2_right = stats.chi2.ppf(0.05,df=ni-1)
#     var_lower = ((ni-1)*xx_std[0,0]**2)/chi2_left
#     var_upper = ((ni-1)*xx_std[0,0]**2)/chi2_right
#     var_mean = (var_lower+var_upper)/2
#     std_dif[i] = np.abs(xx_std-np.sqrt(var_mean))
    
# a = np.array(range(0,len(std_dif)))
# plt.plot(a, std_dif)    
