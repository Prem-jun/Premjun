# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:24:34 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 22:03:33 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 22:16:05 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:28:32 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:27:49 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 22:25:51 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 16:07:58 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:39:07 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:28:03 2021

@author: ASUS
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:03:19 2021

@author: premjunsawang
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 22:39:34 2021

@author: ASUS
This version designed for classification problem
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:06:04 2021

@author: premjunsawang
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 20:22:01 2021

@author: ASUS
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 23:29:16 2021

@author: premjunsawang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:35:18 2021

@author: premjunsawang
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:59:14 2021

@author: ASUS
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import math
import myfun
from scipy.stats import t
import pandas as pd
import numpy as np
from random import randint
from numpy import linalg as LA
from sklearn import datasets
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MC_class:  # Class for creating the new neuron.
    def __init__(self, center,ni,rad,tmp_s,tmp_center,cl):  
        self.center = center  
        self.ni = ni
        self.tmp_s = tmp_s
        self.tmp_center = tmp_center
        self.cl = cl
        # self.acc = acc



class_select = [0,1,2] # if class_select == -1 means data without class label.
tar = 1
features = [0,3] # the selected features for running
InitNum = 1 # initial number of neuron
minnSampl =  5 # minimum number of learned samples 
alpha = 0.05
minncov = 0 # minimum number of covered data for plotting. 

######## Starting program part ###############
MC_cl = []
pres_data_c = np.array([])
pres_data_nc = np.array([])
trained_cl = np.array([0])
A=np.reshape([[0]*InitNum*InitNum],(InitNum,InitNum))
stdise = 1
plot = 1
nseed = 1
if sum(class_select) < 0:
    # pts.csv is the generated data from click mouse by r programing. 
    df = pd.read_csv('D:/KKU/Chula Postdoc/HCM/pts.csv') 
    olddata = df
else:
    iris = datasets.load_iris()
    for i in range(len(class_select)):
        if i == 0:
           TestData = iris.data[iris.target == class_select[i] ,:]  
           df = pd.DataFrame(TestData[:,features])
           if tar == 1:
               df_tar = iris.target[iris.target == class_select[i]]
        else:
            TestData = iris.data[iris.target == class_select[i] ,:]
            frame=[df,pd.DataFrame(TestData[:,features])]
            df = pd.concat(frame)
            df = df.set_index(np.arange(len(df)))
            if tar ==  1:                
                df_tar = np.concatenate([df_tar,iris.target[iris.target == class_select[i]]])
    olddata = df
olddata_tar = df_tar
# olddata = np.array(olddata)
olddata = np.array([])
for i in range(len(df)):
    if df_tar[i] == 2 or df_tar[i] == 1:
        df_tar[i] = 0
nSampl = len(df)
nVar = len(df.columns)
# Setting Initial values of parameters
init_center = np.array([[99999.9]*nVar]*InitNum)
init_ni = np.array([0]*InitNum)
init_rad = np.array([-1.0]*InitNum)
init_tmp_s = np.array([[0.0]*nVar]*InitNum)
init_Lvec = np.array([[0.0]*nVar]*InitNum)
init_L = np.array([0.0]*InitNum)
init_tmp_center = np.array([[99999.9]*nVar]*InitNum)
init_cl = np.array([-99]*InitNum)
# MC_cl.append( MC_class(init_center,init_ni,init_rad,init_tmp_s,init_center,init_cl) )
MC_cl = MC_class(init_center,init_ni,init_rad,init_tmp_s,init_center,init_cl)
while nSampl > 0:
    irand = randint(1,nSampl)-1 # randomly select a data point.
    xvec = np.array([df.iloc[irand,:]])
    nSampl-=1
    df=df.drop(df.index[irand])
    df_tar = np.delete(df_tar,irand) 
    df=df.set_index(np.arange(nSampl))
    if plot == 1:
        myfun.plot2DSphClusterProb(xvec,olddata,MC_cl,pres_data_c,pres_data_nc,plt,np,minncov)
        if len(olddata) == 0:
           olddata = np.array(xvec)
        else:
           olddata = np.concatenate((xvec,olddata))
    tmp_n_sph = sum(MC_cl.ni)
    if tmp_n_sph > 0:
        MC_cl,A,con_update,xvec, pres_data_c, update = myfun.Learn_MC(xvec,pres_data_c,MC_cl,A,trained_cl,nVar,stdise,alpha,t,np,math,LA)
        if update == 1:
            myfun.plot2DSphClusterProb(xvec,olddata,MC_cl,pres_data_c,pres_data_nc,plt,np,minncov)
            s=1
        else:
            n_pres_data_c = len(pres_data_c)
            if n_pres_data_c >= minnSampl:
                ch_list = np.zeros(n_pres_data_c)
                for i in range(n_pres_data_c):
                    min_ind, ch_in = myfun.check_closestsphere_in(pres_data_c[i],MC_cl.center,MC_cl.rad,stdise,np,LA)
                    ch_list[i] = min_ind
                mc_close = np.unique(ch_list)
                for i in range(len(mc_close)):
                    if sum(ch_list == mc_close[i]) == minnSampl:
                        pres_data_c_tmp = pres_data_c[ch_list == mc_close[i]]
                        pres_data_c = np.delete(pres_data_c,np.where(ch_list == mc_close[i])[0],0)                    
                        MC_cl,A, pres_data_c_tmp,ch = myfun.Learn_MC_pres(pres_data_c_tmp,MC_cl,A,minnSampl,nVar,trained_cl[0],stdise,alpha,t,np,math,LA)
                        if ch > 0 :
                           if len(pres_data_c_tmp)>0:
                               pres_data_c = np.concatenate((pres_data_c,pres_data_c_tmp))
                           myfun.plot2DSphClusterProb(xvec,olddata,MC_cl,pres_data_c,pres_data_nc,plt,np,minncov)
        ass_clust,n_clust = myfun.ClustAssign(A,np)                       
    else: # for empty neuron
        # keep the unlearned data in pres_data_c
        if len(pres_data_c) == 0:
            pres_data_c = xvec
        else:                
            pres_data_c = np.concatenate((xvec,pres_data_c))
        xvec = np.delete(xvec,0,0)
        # pres_data_c = np.concatenate((xvec,pres_data_c))
        # xvec = np.delete(xvec,0,0)
        if len(pres_data_c) >= minnSampl:
            # Create a central neuron.
            pres_data_c,tmp_dist, MC_cl,A, ch = myfun.Central_MC(pres_data_c,nVar,trained_cl,MC_cl,A,stdise,alpha,t,np,math,LA)
            if ch == 1: # center neuron and the arounding neurons were generated.
                if plot == 1:
                    # myfun.plot2DSphClusterProb(xvec,olddata,MC_cl,pres_data_c,pres_data_nc,plt,np,minncov)
                    s=1
                tmp_id, tmp_dist = myfun.check_datasphere_in(pres_data_c,MC_cl.center[-1],np.array([MC_cl.rad[-1]*2]),stdise,np,LA) 
                idx_cenmc = len(MC_cl.ni)-1 # Specify the index of the created micro cluster.
                tmp_n = len(pres_data_c)
                if tmp_id[0].size > 0:
                   MC_cl,A,con_update, pres_data_c = myfun.Around_MC(tmp_dist,pres_data_c,nVar,MC_cl,A,trained_cl,stdise,alpha,t,idx_cenmc,np,math,LA)
                   if len(pres_data_c) < tmp_n and plot == 1:
                       myfun.plot2DSphClusterProb(xvec,olddata,MC_cl,pres_data_c,pres_data_nc,plt,np,minncov)
                       s=1
            elif ch == 2:
                idx_cenmc = len(MC_cl.ni)-1
                MC_cl,A,con_update, pres_data_c = myfun.Around_MC(tmp_dist,pres_data_c,nVar,MC_cl,A,trained_cl,stdise,alpha,t,idx_cenmc,np,math,LA)
                if plot == 1:
                    myfun.plot2DSphClusterProb(xvec,olddata,MC_cl,pres_data_c,pres_data_nc,plt,np,minncov)
                    s=1
            ass_clust,n_clust = myfun.ClustAssign(A,np)
    
    