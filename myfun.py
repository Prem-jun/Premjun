def ClustAssign(A,np):
    n_MC = len(A)
    ass_Clust = np.zeros(n_MC) # assigend cluster
    n_clust = 0
    ch = 1
    idx_cl = 0
    while ch:
        idx_nonClust = np.where(ass_Clust==0)[0]
        if len(idx_nonClust)>0:
            idx_cl +=1
            tmp_con = np.zeros(n_MC)
            # idx_con = np.where(A[idx_nonClust[0],]>0)[0]
            current_con = A[idx_nonClust[0],]
            if sum(current_con)==0:
                ass_Clust[idx_nonClust[0]]=idx_cl
            else:
                if sum(tmp_con==current_con)==n_MC:
                    ch = 0
                else:
                    ch1 = 1
                    tmp_con = np.logical_or(tmp_con, current_con)
                    while ch1:
                       idx_con = np.where(tmp_con>0)[0]
                       for i in idx_con:                   
                           tmp_con = np.logical_or(tmp_con, A[i,]) 
                       if sum(tmp_con==current_con)==n_MC:
                           ch1=0
                           ass_Clust[tmp_con>0]=idx_cl
                       else:
                           current_con = tmp_con
        else:
            ch = 0
    n_clust = len(np.unique(ass_Clust))
    return ass_Clust, n_clust    
        

def ConUpdate(idx_active,MC,A,old_rad,np,LA): # Update the connection matrix A
    n_MC = len(A) # Get #neurons
    con_update = np.zeros(n_MC)
    if idx_active == -1:
        # idx_active = len(A[0])-1
        idx_active = n_MC-1
    if n_MC > 1:
        for k in range(n_MC):
            if k != idx_active:
                con = ConCheck(MC.center[idx_active],MC.center[k],MC.rad[idx_active],MC.rad[k],np,LA)
                if A[idx_active][k]!=con:
                    con_update[k] = 1
                    A[idx_active][k] = con
                    A[k][idx_active] = con
    return A, con_update
def ConCheck(center1,center2,rad1,rad2,np,LA): # Check condition
    con = 0
    dist_cen = distsph(center1,center2,np,LA) # distance between center1 and center2
    if dist_cen <= 2*rad1 or dist_cen <= 2*rad2:
        con = 1
    # elif dist_cen <= 3*rad1+rad2 or dist_cen <= 3*rad2+rad1: 
    elif dist_cen <= 3*rad1 or dist_cen <= 3*rad2:     
        con = 1
    return con

def ConMat4ConMC(A,idx1,idx2,np):
    n_MC = len(A)
    con_update = np.zeros(n_MC)
    A[idx1][idx2] = 1
    A[idx2][idx1] = 1
    con_update[idx2] = 1
    return A,con_update

def ConMat4AddMC(MC,A,np): # Connection matrix for adding the new mc.
    if len(MC.ni)-1 == len(A):
        A = np.concatenate((A,np.array([np.zeros(len(A))])))
        A = np.concatenate((A,np.zeros(len(A)).reshape(len(A),1)),1)
        # A = np.concatenate((A,np.array([np.zeros(len(A[0]))])),0)
    return A

def MergeProcess(MC,A,idx_current,stdise,alpha,t,np,math,LA):
    merge = 0
    idx_merge = -99
    if len(MC.ni) >= 2:
        # idx_current = -1
        closest_id, ch_in2 = check_closestcent4merge(idx_current,MC.center,MC.rad,stdise,np,LA)
        if ch_in2 == 1:
             MC,id_del,idx_merge = MergeMC(MC,idx_current,closest_id,alpha,t,np,math)
             merge = 1
        if merge == 1:
            A = np.delete(A,id_del,0)
            A = np.delete(A,id_del,1)
            A = ConMat4AddMC(MC,A,np)
            idx_active = -1
            old_rad = 0
            A,con_update = ConUpdate(idx_active,MC,A,old_rad,np,LA)
    return merge,MC,A,idx_merge


def check_closestcent4merge(idx_cenn,centervec,rad,stdise,np,LA):
    # This function is created for checking the merge condition
    # ncen=len(centervec)
    ch_in=np.zeros(1)
    if stdise == 1:
        # compute the distance of centervec[idx_cenn] wrt the radias of each neuron
       tmp_dist = dist_sphere_stdise(centervec[idx_cenn],centervec,rad,np,LA)
       # compute the distance of centervec wrt the radias of centervec[idx_cenn]
       tmp_dist2 = dist_sphere_stdise(centervec,centervec[idx_cenn],rad[idx_cenn],np,LA)
    else:
       tmp_dist = dist_sphere(centervec[idx_cenn],centervec,rad,np,LA)
       tmp_dist = dist_sphere(centervec,centervec[idx_cenn],rad[idx_cenn],np,LA)
       
       
    rad2 = rad[tmp_dist==-1] # Set rad2 equal to rad of the centervec[idx_cenn]
    tmp_dist[tmp_dist==-1]=float('inf') # Set the distance of itselt to inf value.
    min_ind=np.argmin(tmp_dist )
    if min(tmp_dist) <=0: # the centervec[idx_cenn] is in the others.
        
        # decision making for selecting a merge indice.
        tmp_rad = rad[min_ind]
        if tmp_rad > 2*rad2:
           ch_in[0]=1
    
    tmp_dist2[tmp_dist2==-1]=float('inf') # Set the distance of itselt to inf value.              
    
    if min(tmp_dist2) <=0: # if the other centers is in centervec[idx_cenn]
      min_ind2=np.argmin(tmp_dist2 )
      tmp_rad2 = rad[min_ind2] 
      if rad2 > 2*tmp_rad2:
           ch_in[0]=1
           min_ind = min_ind2
       
    return min_ind, ch_in

def MergeVar(var1,var2,num1,num2,xbar1,xbar2):
    varpoo = ((num1-1)*var1 + (num2-1)*var2 + ((num1*num2)/(num1+num2))*(xbar1-xbar2)**2)/(num1+num2-1)
    return varpoo

def MergeCen(cen1,num1,cen2,num2):
    ncen1 = num1*cen1
    ncen2 = num2*cen2
    num = num1+num2
    return (ncen1+ncen2)/num

def MergeMC(MC,idi,idj,alpha,t,np,math):
    tmp_cen = MergeCen(MC.center[idi],MC.ni[idi],MC.center[idj],MC.ni[idj])
    
    ndim = len(tmp_cen)
    tmp_s = np.zeros(ndim)
    for i in range(ndim):
        tmp_s[i] = math.sqrt(MergeVar(MC.tmp_s[idi][i],MC.tmp_s[idj][i],MC.ni[idi],MC.ni[idj],MC.center[idi][i],MC.center[idj][i]))
    
    MC.center = np.concatenate((MC.center,np.array([tmp_cen])))
    MC.ni = np.concatenate((MC.ni,np.array([MC.ni[idi]+MC.ni[idj]])))
    if MC.cl[idi] >= 0:
       MC.cl = np.concatenate((MC.cl,np.array([MC.cl[idi]])))
    elif MC.cl[idj] > 0:
        MC.cl = np.concatenate((MC.cl,np.array([MC.cl[idj]])))
    
    MC.tmp_center = np.concatenate((MC.tmp_center,np.array([tmp_cen])))
    MC.tmp_s = np.concatenate((MC.tmp_s,np.array([tmp_s])))
    tmp_rad = conf_rad(alpha,MC.ni[idi]+MC.ni[idj],tmp_s,t,np,math)
    MC.rad = np.concatenate((MC.rad,tmp_rad))
    id_del = np.array([idi,idj])
    MC.center = np.delete(MC.center,id_del,0)
    MC.ni = np.delete(MC.ni,id_del,0)
    MC.cl = np.delete(MC.cl,id_del,0)
    MC.tmp_s = np.delete(MC.tmp_s,id_del,0)
    MC.rad = np.delete(MC.rad,id_del,0)
    MC.tmp_center = np.delete(MC.tmp_center,id_del,0)
    idx_merge = -1
    return MC, id_del,idx_merge    
    
def GenUnkDat(target,unk_tar,nseed):
    from random import seed, sample
    import numpy as np
    import math
    seed(nseed)
    ndat = len(target)
    sequence = [i for i in range(ndat)]
    unk_tarn = math.floor(unk_tar*ndat)
    unk_ind = sample(sequence, unk_tarn)
    dat_tar = np.zeros(ndat)
    dat_tar[unk_ind] = -1

    for i in range(ndat):
        if dat_tar[i] == 0:
            dat_tar[i] = target[i]
    return dat_tar

def Measure_Perf(pres_data_ncl_test,pres_data_ncl_test_lab,trained_cl,MC_cl,stdise,np,LA):
    ndat = len(pres_data_ncl_test)
    acc_ch = np.zeros(ndat)
    unk_count = 0.0 
    nk_count = 0.0
    xvec_predictall = np.zeros(ndat)
    # nlearn = len(trained_cl)
    # unk_cl = 0
    for i in range(ndat):
        xvec_test = pres_data_ncl_test[i]
        mindist_cl, xvec_predict, idx_cl, idx_MC = Predict_MC(xvec_test,trained_cl,MC_cl,stdise,np,LA)
        xvec_predictall[i] = xvec_predict
        if xvec_predict >= 0:
            nk_count +=1
            if xvec_predict == pres_data_ncl_test_lab[i]:
                acc_ch[i] = 1.0
        else:
            unk_count +=1
    if nk_count > 0:        
        nk_acc = sum(acc_ch)/nk_count
    else:
        nk_acc = -1.0
    unk_rad = unk_count/ndat 
    
     
        # if xvec_predict >= 0:
        #     if xvec_predict == pres_data_ncl_test_lab[i]:
        #         acc_ch[i] = 1.0
        # else:
        #     for k in range(nlearn):
        #         if k > nlearn:
        #             unk_cl = 1
        #         else:
                    
            # count = 0
            # for k in range(len(MC_cl)):
            #     if MC_cl[k].cl[0]>= 0:
            #         count +=1
            # if count < nlearn:                            
            #     for k in range(len(MC_cl)):
            #         if  pres_data_ncl_test_lab[i] != MC_cl[k].cl[0] and MC_cl[k].cl[0] >= 0:
            #             countn = 1
            #             break
            #     if countn > 0:
            #         acc_ch[i] = 1.0
            # else:
            #     for k in range(len(MC_cl)):
            #         if  pres_data_ncl_test_lab[i] != MC_cl[k].cl[0] and MC_cl[k].cl[0] < 0:
            #             countn = 1
            #             break
            #     if countn > 0:
            #         acc_ch[i] = 1.0
                
            # # for k in range(len(trained_cl)):
            # #     if MC_cl[k].ni[0] > 0:
            # #         count += 1
            # # if count < len(trained_cl):
            # #     acc_ch[i] = 1.0
            # # if sum(pres_data_ncl_test_lab[i] == trained_cl)== 0 : 
            #     # acc_ch[i] = 1.0
    return xvec_predictall, nk_acc, unk_rad, ndat    
        # if len(Err_all) == 0:
        #     Err_all = np.array([sum(acc_ch)/ndat])
        # else:
        #     Err_all = np.concatenate((Err_all, np.array([sum(acc_ch)/ndat])))
        

def Learn_MC_Unk(xvec,pres_data_ncl,pres_data_cl,MC_cl,trained_cl,nVar,stdise,alpha,t,np,math,LA):
    mindist_cl, xvec_predict, idx_cl, idx_MC = Predict_MC(xvec,trained_cl,MC_cl,stdise,np,LA)
    if xvec_predict > 0:
       if mindist_cl[idx_cl] <= MC_cl[idx_cl].rad[idx_MC]:
           MC_cl[idx_cl],xvec, pres_data_cl[idx_cl], update = Learn_MC(xvec,pres_data_cl[idx_cl],MC_cl[idx_cl],trained_cl[idx_cl],nVar,stdise,alpha,t,np,math,LA)
       elif mindist_cl[idx_cl] <= 2.0*MC_cl[idx_cl].rad[idx_MC]:                    
           pres_data_cl[idx_cl] = np.concatenate((xvec,pres_data_cl[idx_cl])) 
           tmp_id, tmp_dist = check_datasphere_in(pres_data_cl[idx_cl],MC_cl[idx_cl].center[idx_MC],np.array([MC_cl[idx_cl].rad[idx_MC]*2]),stdise,np,LA)    
           MC_cl[idx_cl], pres_data_cl[idx_cl] = Around_MC(tmp_dist,pres_data_cl[idx_cl],nVar,MC_cl[idx_cl],trained_cl[idx_cl],stdise,alpha,t,idx_MC,np,math,LA)
    else:
        if len(pres_data_ncl) == 0:
           pres_data_ncl=np.array(xvec)
           xvec = np.delete(xvec,0,0)           
        else:
            pres_data_ncl = np.concatenate((xvec,pres_data_ncl))
            xvec = np.delete(xvec,0,0)
    return xvec,xvec_predict, pres_data_ncl, pres_data_cl, MC_cl        
                   
def Learn_MC_UnkPres(pres_data_ncl,pres_data_cl,MC_cl,trained_cl,nVar,stdise,alpha,t,np,math,LA):
    ndat = len(pres_data_ncl)
    pres_ncl_predict = np.zeros(ndat)
    for i in range(ndat):
        xvec = pres_data_ncl[i]
        mindist_cl, xvec_predict, idx_cl, idx_MC = Predict_MC(xvec,trained_cl,MC_cl,stdise,np,LA)
        pres_ncl_predict[i] = xvec_predict 
        if xvec_predict > 0:
           if mindist_cl[idx_cl] <= MC_cl[idx_cl].rad[idx_MC]: 
               MC_cl[idx_cl],xvec, pres_data_cl[idx_cl], update = Learn_MC(xvec,pres_data_cl[idx_cl],MC_cl[idx_cl],trained_cl[idx_cl],nVar,stdise,alpha,t,np,math,LA)
                                  
           elif mindist_cl[idx_cl] <= 2.0*MC_cl[idx_cl].rad[idx_MC]:  
               if len(pres_data_cl[idx_cl]) == 0:
                   pres_data_cl[idx_cl] = np.array([xvec])
               else:
                   pres_data_cl[idx_cl] = np.concatenate((xvec,pres_data_cl[idx_cl])) 
               tmp_id, tmp_dist = check_datasphere_in(pres_data_cl[idx_cl],MC_cl[idx_cl].center[idx_MC],np.array([MC_cl[idx_cl].rad[idx_MC]*2]),stdise,np,LA)    
               MC_cl[idx_cl], pres_data_cl[idx_cl] = Around_MC(tmp_dist,pres_data_cl[idx_cl],nVar,MC_cl[idx_cl],trained_cl[idx_cl],stdise,alpha,t,idx_MC,np,math,LA)
    return MC_cl,pres_ncl_predict, pres_data_ncl, pres_data_cl       
            
def plot2DSphClusterProb(xvec,olddata,MC_cl,pres_data_c,pres_data_nc,plt,np,minncov):
    data_color = ['rx']
    sph_color = ['co']
    pres_color = ['ko']
    plt.plot(xvec[:,0],xvec[:,1],'go')
    if len(olddata) > 0:
       plt.plot(olddata[:,0],olddata[:,1],data_color[0])
    if len(pres_data_c) > 0:
        plt.plot(pres_data_c[:,0],pres_data_c[:,1],pres_color[0])
    if MC_cl.ni[0] > 0:
       for k in range(len(MC_cl.ni)):
         if MC_cl.cl[k] == -1:
            datapoints = dat_cycle(MC_cl.center[k],MC_cl.rad[k],30,np)
            plt.plot(datapoints[:,0],datapoints[:,1],'ko') 
            plt.plot(MC_cl.center[k,0],MC_cl.center[k,1],'ko')
         else: 
            if MC_cl.ni[k] > minncov:
                datapoints = dat_cycle(MC_cl.center[k],MC_cl.rad[k],30,np)
                plt.plot(datapoints[:,0],datapoints[:,1],sph_color[0]) 
                plt.plot(MC_cl.center[k,0],MC_cl.center[k,1],sph_color[0]) 
    plt.show()

def plot2DSphCluster3(xvec,olddata,olddata_tar,MC_cl,pres_data_cl,pres_data_ncl,trained_cl,plt,np,minncov):
    data_color = ['cx','bx','rx']
    sph_color = ['co','bo','ro']
    pres_color = ['ko','ko','ko']
    uniq_old, ind_old = np.unique(olddata_tar, return_index=True)
    # uniq_pres, ind_pres = np.unique(pres_data_tar, return_index=True)
    # uniq_sph, ind_sph = np.unique(MC_cl['cl'], return_index=True)
    plt.plot(xvec[:,0],xvec[:,1],'go')    
    if len(olddata)>0:
        for i in range(len(uniq_old)):
            plt.plot(olddata[olddata_tar == uniq_old[i],0], olddata[olddata_tar == uniq_old[i],1],data_color[i])
    if len(pres_data_ncl) > 0:
        plt.plot(pres_data_ncl[:,0],pres_data_ncl[:,1],'ro')
    for i in range(len(trained_cl)):
        if len(pres_data_cl[i]) > 0:
            plt.plot(pres_data_cl[i][:,0],pres_data_cl[i][:,1],pres_color[i])
            
        if MC_cl[i].ni[0] > 0:
            for k in range(len(MC_cl[i].ni)):
                if MC_cl[i].cl[k] == -1:
                    datapoints = dat_cycle(MC_cl[i].center[k],MC_cl[i].rad[k],30,np)
                    plt.plot(datapoints[:,0],datapoints[:,1],'ko') 
                    plt.plot(MC_cl[i].center[k,0],MC_cl[i].center[k,1],'ko')
                else:
                    if MC_cl[i].ni[k] > minncov:
                        datapoints = dat_cycle(MC_cl[i].center[k],MC_cl[i].rad[k],30,np)
                        plt.plot(datapoints[:,0],datapoints[:,1],sph_color[i]) 
                        plt.plot(MC_cl[i].center[k,0],MC_cl[i].center[k,1],sph_color[i])
                
    plt.show()

def Learn_MC_pres(pres_data,MC,A,minnSampl,nVar,trained_cl,stdise,alpha,t,np,math,LA):
    # This function is to create the set of microcluster for dealin with
    # the data in the preserved data set.
    pres_data,tmp_dist, MC,A, ch = Central_MC(pres_data,nVar,trained_cl,MC,A,stdise,alpha,t,np,math,LA)
    # myfun.plot2DSphCluster2(olddata,olddata_tar,MC_cl,pres_data_cl,trained_cl,plt,np,minncov)
    if ch == 1 : # Create a central MC with a specified class label.
        tmp_id, tmp_dist = check_datasphere_in(pres_data,MC.center[-1],np.array([MC.rad[-1]*2]),stdise,np,LA) 
        idx_cenmc = len(MC.ni)-1 # Specify the index of the created micro cluster.
        if tmp_id[0].size > 0:
            MC,A,con_update, pres_data = Around_MC(tmp_dist,pres_data,nVar,MC,A,trained_cl,stdise,alpha,t,idx_cenmc,np,math,LA)
            # myfun.plot2DSphCluster2(olddata,olddata_tar,MC_cl,pres_data_cl,trained_cl,plt,np,minncov) 
    elif ch == 2:
        idx_cenmc = len(MC.ni)-1
        MC,A,con_update, pres_data = Around_MC(tmp_dist,pres_data,nVar,MC,A,trained_cl,stdise,alpha,t,idx_cenmc,np,math,LA)                   
    return MC,A, pres_data, ch

def Learn_MC(xvec,pres_data,MC,A,trained_cl,nVar,stdise,alpha,t,np,math,LA):
#   This function is used to handle an incoming datum xvec and the uncovered data in 
# pres_data set and unknown label dataset name pres_data_ncl.
    # Step1: Find the closest MC.
    # Step2: If xvec is in the closest MC, then updating this MC with xvec by 
    #       the SphCoverIn function and go to Step5:. Otherwise, do Step3.
    # Step3:    If xvec is nearby the closest MC, then updating MC to cover xvec.
    # Step4:    If there are some data in press_data is the range of 2 times of radius,
    #       then create the arounding MC to cover them.
    # Step5: If xvec is used for update process, then for each MC, create the new 
    #      MC to covere the left data in pres_data.
    # Step6: Return MC, xvec, pres_data and the update status.

# ---------- > Step1: Find the closest MC.
    min_ind, ch_in = check_closestsphere_in(xvec,MC.center,MC.rad,stdise,np,LA)
    update = 0
# ---------- > Step2:
    if ch_in == 1: # if xvec is inside MC, then updating the MC.                
        MC,A,con_update, xvec, ch1 = SphCoverIn(MC,A,min_ind,trained_cl,xvec,nVar,stdise,alpha,t,math,np,LA)
        if ch1 == 1: # the neuron min_ind is updated           
            # if len(MC.ni) >= 2:
            #     closest_id, ch_in2 = check_closestcent4merge(min_ind,MC.center,MC.rad,stdise,np,LA)
            #     if ch_in2 == 1:
            #          MC = MergeMC(MC,min_ind,closest_id,alpha,t,np,math)
            idx_current = min_ind
            merge, MC,A,idx_merge = MergeProcess(MC,A,idx_current,stdise,alpha,t,np,math,LA)
            
        update = 1
        
    else:
# ---------- > Step3:    
        MC,A,con_update, xvec, ch2 = SphCoverNear(MC,A,min_ind,min_ind,xvec,nVar,stdise,alpha,t,math,np,LA)
        if ch2 == 0: # if xvec is around the MC, then combining xvec to pres_data. 
            if xvec.ndim == 1:
                xvec = np.array([xvec])
            pres_data = np.concatenate((xvec,pres_data)) 
# ---------- > Step4:            
            tmp_id, tmp_dist = check_datasphere_in(pres_data,MC.center[min_ind],np.array([MC.rad[min_ind]*2]),stdise,np,LA) 
            if tmp_id[0].size > 0:
                MC,A,con_update, pres_data = Around_MC(tmp_dist,pres_data,nVar,MC,A,trained_cl,stdise,alpha,t,min_ind,np,math,LA)                
                idx_current = -1
                merge, MC,A,idx_merge = MergeProcess(MC,A,idx_current,stdise,alpha,t,np,math,LA)
               
                # if len(MC.ni) >= 2:
                    # closest_id, ch_in2 = check_closestcent4merge(idx_current,MC.center,MC.rad,stdise,np,LA)
                    # if ch_in2 == 1:
                    #      MC = MergeMC(MC,idx_current,closest_id,alpha,t,np,math)
                
                update = 1
            
        else:
            # if len(MC.ni) >= 2:
            #     closest_id, ch_in2 = check_closestcent4merge(min_ind,MC.center,MC.rad,stdise,np,LA)
            #     if ch_in2 == 1:
            #         MC = MergeMC(MC,min_ind,closest_id,alpha,t,np,math)
            idx_current = -1
            merge, MC,A,idx_merge = MergeProcess(MC,A,idx_current,stdise,alpha,t,np,math,LA)
            
            update = 1
    if update == 1: # if xvec is used to update the MC, then check the pres_data is used or not by the updating MC.               
        for i in range(len(MC.ni)):
            if len(pres_data) > 0:                        
                tmp_id, tmp_dist = check_datasphere_in(pres_data,MC.center[i],np.array([MC.rad[i]*2]),stdise,np,LA)    
                if tmp_id[0].size > 0 :
                   MC,A,con_update, pres_data = Around_MC(tmp_dist,pres_data,nVar,MC,A,trained_cl,stdise,alpha,t,i,np,math,LA)
                   # idx_current = -1
                   # merge, MC = MergeProcess(MC,idx_current,stdise,alpha,t,np,math,LA)
                   # # if len(MC.ni) >= 2:
                   # #      idx_current = -1
                   # #      closest_id, ch_in2 = check_closestcent4merge(idx_current,MC.center,MC.rad,stdise,np,LA)
                   # #     if ch_in2 == 1:
                   # #        MC = MergeMC(MC,idx_current,closest_id,alpha,t,np,math)                   
            else:
                break
    return MC,A,con_update,xvec, pres_data, update

def Acc_MC(xvec,xvec_tar,MC_cl,trained_cl,np,LA):
    numdat = len(xvec_tar)
    predict_cl = np.zeros(numdat)
    acc_ch = np.zeross(numdat)
    for i in range(numdat):
        mindist_cl, predict_val = Predict_MC(xvec,trained_cl,MC_cl,np,LA)
        predict_cl[i] = predict_val
        if predict_cl[i] == xvec_tar[i]:
            acc_ch = 1
        elif predict_cl[i] == -1:
            acc_ch = -1
        else:
            acc_ch = 0
    return predict_cl, acc_ch

def Predict_MC(xvec,trained_cl,MC_cl,stdise,np,LA):
    numcl = len(trained_cl)
    mindist_cl = np.zeros(numcl)
    minind_cl = np.zeros(numcl)
    minind_cl = minind_cl.astype(int)
    for i in range(numcl):
        if sum(MC_cl[i].ni) > 0:
            mindist, minind = check_closestsphere_dist(xvec,MC_cl[i].center,MC_cl[i].rad,stdise,np,LA)
            mindist_cl[i] = mindist
            minind_cl[i] = minind
        else:
            mindist_cl[i] = float('inf')
            minind_cl[i] = -1
    idx_mindist = np.argmin(mindist_cl)
    if mindist_cl[idx_mindist] < 2.0*MC_cl[idx_mindist].rad[minind_cl[idx_mindist]]:        
        predict_cl = trained_cl[idx_mindist]
    else:
        predict_cl = -1
    if predict_cl == -1:
        idx_MC = -1
        idx_cl = -1
    else:
        idx_MC = minind_cl[idx_mindist]
        idx_cl = idx_mindist
    return mindist_cl, predict_cl,idx_cl,idx_MC        


def Around_MC(tmp_dist,pres_data,nVar,MC,A,trained_cl,stdise,alpha,t,idx_cenmc,np,math,LA):
    # tmp_dist is the distance of 2 times radius of the central MC.
    #Alg.:
    # Step 1: find the closest data.
    # Step 2: create the around MC from the closest one.
    # Step 3: delete the closest one.
    # Step 4: if the created around MC cover the left data in pres_data, then update this one. 
    # Step 5: if the created around MC cover the near data, then update this one again.
    # Step 6: compute the distance of 2 times padius and do Steps 1-5 until no data are covered.
    count = 1
    idx_central = idx_cenmc # The central MC's index.
    while count:
        # Step 1: # find the closest data used as the new center.
        idx_close = np.argmin(tmp_dist) 
        # Step 2:
        MC = add_new_sph2(pres_data[idx_close],MC.rad[idx_central],MC,pres_data[idx_close],MC.tmp_s[idx_central],np)
        A = ConMat4AddMC(MC,A,np)
        if MC.ni[idx_central]>1:
            A,con_update = ConMat4ConMC(A,idx_central,-1,np) 
            A,con_update = ConUpdate(-1,MC,A,MC.rad[-1],np,LA)
        MC.cl = np.append(MC.cl,np.array([trained_cl])) # setting the class label to the new one.                                
        # Step 3: 
        pres_data = np.delete(pres_data,idx_close,0) # put away the trained data.
        # Step 4:
        if len(pres_data) > 0:
            idx_ch = -1
            MC,A,con_update, pres_data, ch1 = SphCoverIn(MC,A,idx_ch,trained_cl,pres_data,nVar,stdise,alpha,t,math,np,LA)
            if ch1 == 1:
                 idx_current = idx_ch
                 merge, MC,A,idx_merge = MergeProcess(MC,A,idx_current,stdise,alpha,t,np,math,LA)
                 
        else:
            count = 0
        
        if len(pres_data) > 0:
            CovStat = 1
            # Step 5:
            while CovStat:
                MC,A,con_update, pres_data, CovStat = SphCoverNear(MC,A,idx_ch,idx_central,pres_data,nVar,stdise,alpha,t,math,np,LA)    
                if CovStat == 1:
                    idx_current = idx_ch
                    merge, MC,A, idx_merge = MergeProcess(MC,A,idx_current,stdise,alpha,t,np,math,LA)
                    
            # Step 6:
            if len(pres_data) > 0:
               tmp_id, tmp_dist = check_datasphere_in(pres_data,MC.center[idx_central],np.array([MC.rad[idx_central]*2]),stdise,np,LA) 
               if tmp_id[0].size == 0:
                   count = 0
            else:
                count = 0
        else:
            count = 0
    return MC,A,con_update, pres_data
    
def Central_MC(pres_data,nVar,trained_cl,MC,A,stdise,alpha,t,np,math,LA):
    # This function is to create the central MC based on the data in pres_data.
    # Step 1: Compute the mean, sd, n and radius of pres_data.
    # Step 2: If there are some covered data wrt. the mean and sd, then creating the new MC. 
    # Step 3: Update parameter of the new MC by the set of covered data in Step2:   
    # Step 4: If there is no covered data, then setting the class of the central MC as -1.
    
# -----> Step1:
    tmp_center = np.mean(pres_data,axis = 0)
    tmp_s = np.std(pres_data,axis = 0)
    tmp_n = len(pres_data) # number of preserved data.
    tmp_rad = conf_rad(alpha,tmp_n,tmp_s,t,np,math)
    
# -----> Step2:
    tmp_id, tmp_dist = check_datasphere_in(pres_data,tmp_center,tmp_rad,stdise,np,LA)
    ch = 0
    if tmp_id[0].size>0:
        ch = 1
        if tmp_id[0].size == 1:
            tmp_center =  pres_data[tmp_id[0]]
            # pres_data = np.delete(pres_data,tmp_id[0],0)
            # tmp_dist = np.delete(tmp_dist,tmp_id[0])
        else:
            tmp_center = np.mean(pres_data[tmp_id[0]],axis = 0)
        MC = add_new_sph2(tmp_center,tmp_rad,MC,tmp_center,tmp_s,np)   
        # idx_cenmc = len(A)-1
        A = ConMat4AddMC(MC,A,np)   
        if len(MC.ni) == 1 :
            MC.cl = np.array([trained_cl])
        else:
            if trained_cl.ndim == 0:
                # trained_cl = np.array([trained_cl])
                MC.cl=np.append(MC.cl,trained_cl)
            # elif trained_cl.ndim == 1:
            #     trained_cl = np.array([trained_cl])
            # MC.cl = np.concatenate((MC.cl,trained_cl))
        count = 1
# -----> Step 3:
        while count:
            idx_ch = -1
            MC,A,con_update, pres_data, ch1 = SphCoverIn(MC,A,idx_ch,trained_cl,pres_data,nVar,stdise,alpha,t,math,np,LA)            
            if ch1==1:
                if len(A)>1:
                      # idx_active = idx_ch
                      # A = ConUpdate(idx_active,MC,A,np,LA)                
                      idx_current = idx_ch
                      merge, MC,A,idx_merge = MergeProcess(MC,A,idx_current,stdise,alpha,t,np,math,LA)
                      
            if len(pres_data) > 0:
                MC,A,con_update, pres_data, ch2 = SphCoverNear(MC,A,idx_ch,idx_ch,pres_data,nVar,stdise,alpha,t,math,np,LA)
                if ch2==1:
                    if len(A)>1:
                        # con = ConCheck(MC.cl[idx_cenmc],MC.cl[idx_ch],MC.rad[idx_cenmc],MC.rad[idx_ch],np,LA)
                        idx_current = idx_ch
                        merge, MC,A,idx_merge = MergeProcess(MC,A,idx_current,stdise,alpha,t,np,math,LA)
                           
                if len(pres_data) > 0:
                    tmp_id, tmp_dist = check_datasphere_in(pres_data,MC.center[idx_ch],np.array([MC.rad[idx_ch]]),stdise,np,LA)
                    if tmp_id[0].size == 0:
                        count = 0
                else:
                    count = 0
            else:
                count = 0
    else:
# -----> Step 4:
        tmp_id, tmp_dist = check_datasphere_in(pres_data,tmp_center,tmp_rad*2,stdise,np,LA)
        if tmp_id[0].size>0:
            ch = 2
            MC = add_new_sph2(tmp_center,tmp_rad,MC,tmp_center,tmp_s,np) 
            # idx_cenmc = len(A)-1
            A = ConMat4AddMC(MC,A,np) 
            if len(MC.ni) == 1 :
                MC.cl = np.array([-1])
            else:
                MC.cl = np.concatenate((MC.cl,np.array([-1])))
    return pres_data,tmp_dist, MC,A, ch 
            
def SphCoverIn(MC,A,idx_ch,trained_cl,pres_data,nVar,stdise,alpha,t,math,np,LA):
#   This function is used to update MC if there are some data in 'pres_data' 
# being inside the sphere.
    # 'idx_ch' is the index of the active MC.
    # 'trained_cl' is the class label of MC.
    # Alg:
    # Step1: Find the data inside the MC's 'idx_ch' index.
    # Step2: If there are data inside this MC, then update MC and change 
    #         the minus class label of MC to the normal class labels.
    # Step3: Update center and sd 
   
    if pres_data.ndim == 1:
        pres_data = np.array([pres_data])
        
#-----> Step1:
    tmp_rad = np.array([MC.rad[idx_ch]]) # tmp_rad keep the radius of the old mc.
    tmp_id2, tmp_dist2 = check_datasphere_in(pres_data,MC.center[idx_ch],tmp_rad,stdise,np,LA)
    
#-----> Step2: 
    old_rad = MC.rad[idx_ch]
    ch = 0 # there is no data in sphere.
    if tmp_id2[0].size > 0: # there are data in the sphere.
        if MC.cl[idx_ch] == -1: # change class label.
            MC.cl[idx_ch] = np.array(trained_cl)
        ch = 1
        tmp_center2 = MC.center[idx_ch]                        
        tmp_s2 = MC.tmp_s[idx_ch]
        tmp_ni2 = MC.ni[idx_ch]    
        
        for i in range(tmp_id2[0].size):
            tmp_pres = pres_data[tmp_id2[0][i]]
            tmp_center2, tmp_ni2 = update_center(tmp_center2,tmp_pres,tmp_ni2)
            tmp_s2 = update_sd(tmp_pres,tmp_center2,nVar,tmp_s2,tmp_ni2,math,np)
        MC.tmp_s[idx_ch] = tmp_s2
        MC.center[idx_ch] = tmp_center2
        min_covered  = 4
        
        if tmp_ni2 >= min_covered:   
                                          
            # MC.rad[idx_ch] = conf_rad(alpha,tmp_ni2,tmp_s2,t,np,math)
            tmp_rad = conf_rad(alpha,tmp_ni2,tmp_s2,t,np,math) #compute the new radias.
            tmp_id3, tmp_dist3 = check_datasphere_in(pres_data[tmp_id2[0]],MC.center[idx_ch],tmp_rad,stdise,np,LA)
            if tmp_id3[0].size == 0: # the data inside the old but outside the update one.
                MC.rad[idx_ch] = tmp_rad
            else:
                if any(tmp_dist3[tmp_id3[0]]>-0.5) and any(tmp_dist2[tmp_id2[0]])> -0.5:
                    MC.rad[idx_ch] = tmp_rad
            # if tmp_id3[0].size == tmp_id2[0].size and tmp_dist3>-0.5 and tmp_dist2> -0.5:  # If the new radias cover the point so updating it to this new one.
            #     MC.rad[idx_ch] = tmp_rad
               
                
                
        MC.ni[idx_ch] = tmp_ni2
        pres_data = np.delete(pres_data,tmp_id2[0],0)
    A,con_update = ConUpdate(idx_ch,MC,A,old_rad,np,LA)
    return MC,A,con_update,pres_data, ch

def SphCoverNear(MC,A,idx_ch,idx_central,pres_data,nVar,stdise,alpha,t,math,np,LA):
    # pres_data is the set of data points.
    # Step1: Compute temporary center named tmp_center.
    # Step2: If the data is covered by the temporary center, then
    #        update center and sd.  
    # Step3: If #covered data is more than min value, then do Step4:  
    # Step4:    If the updated radius covered all covered data, then updating 
    #           radius. Otherwise, not updating. 
    # Step5: Otherwise, set the radius equal to the central MC's radius.   
    
    tmp_ch = np.zeros(len(pres_data))
     # tmp_ni = minnSampl-1
    tmp_ni = MC.ni[idx_ch]
    tmp_center_fix = MC.center[idx_ch]
    tmp_rad_fix = MC.rad[idx_ch]
    tmp_rad_fix = np.array([tmp_rad_fix])
    CovStat = 0                        
    for k in range(len(tmp_ch)):
    # -----> Step1:                        
        tmp_center, tmp_ni2 = update_center(tmp_center_fix,pres_data[k],tmp_ni)
    
    # -----> Step2:    
        tmp_id, tmp_dist = check_datasphere_in(pres_data[k],tmp_center,tmp_rad_fix,stdise,np,LA)                                                    
        if tmp_id[0].size > 0:
            tmp_ch[k] = 1                                
            MC.center[idx_ch], tmp_ni2 = update_center(MC.center[idx_ch],pres_data[k],tmp_ni+sum(tmp_ch))
            MC.tmp_s[idx_ch] = update_sd(pres_data[k],MC.center[idx_ch],nVar,MC.tmp_s[idx_ch],tmp_ni+sum(tmp_ch),math,np) 
    old_rad = MC.rad[idx_ch]
    if sum(tmp_ch) > 0:
        MC.ni[idx_ch] = tmp_ni+sum(tmp_ch)
    # -----> Step3:    
        if MC.ni[idx_ch] >3:
            tmp_rad = conf_rad(alpha,MC.ni[idx_ch],MC.tmp_s[idx_ch],t,np,math)
            tmp_id3, tmp_dist3 = check_datasphere_in(pres_data[tmp_ch == 1],MC.center[idx_ch],tmp_rad,stdise,np,LA)
    # -----> Step4:
            if tmp_id3[0].size == sum(tmp_ch):
                # old_rad = MC.rad[idx_ch]
                MC.rad[idx_ch] = tmp_rad
                # A,con_update = ConUpdate(idx_ch,MC,A,old_rad,np,LA) 
        else:
            MC.rad[idx_ch] = MC.rad[idx_central]
        pres_data = np.delete(pres_data,tmp_ch == 1,0)
        CovStat = 1
    A,con_update = ConUpdate(idx_ch,MC,A,old_rad,np,LA)    
    return MC,A,con_update, pres_data, CovStat    

def plot2DSphCluster2(olddata,olddata_tar,MC_cl,pres_data_cl,trained_cl,plt,np,minncov):
    data_color = ['cx','bx','rx']
    sph_color = ['co','bo','ro']
    pres_color = ['ko','ko','ko']
    uniq_old, ind_old = np.unique(olddata_tar, return_index=True)
    # uniq_pres, ind_pres = np.unique(pres_data_tar, return_index=True)
    # uniq_sph, ind_sph = np.unique(MC_cl['cl'], return_index=True)
    
    if len(olddata)>0:
        for i in range(len(uniq_old)):
            plt.plot(olddata[olddata_tar == uniq_old[i],0], olddata[olddata_tar == uniq_old[i],1],data_color[i])
    for i in range(len(trained_cl)):
        if len(pres_data_cl[i]) > 0:
            plt.plot(pres_data_cl[i][:,0],pres_data_cl[i][:,1],pres_color[i])
            
        if MC_cl[i].ni[0] > 0:
            for k in range(len(MC_cl[i].ni)):
                if MC_cl[i].cl[k] == -1:
                    datapoints = dat_cycle(MC_cl[i].center[k],MC_cl[i].rad[k],30,np)
                    plt.plot(datapoints[:,0],datapoints[:,1],'ko') 
                    plt.plot(MC_cl[i].center[k,0],MC_cl[i].center[k,1],'ko')
                else:
                    if MC_cl[i].ni[k] > minncov:
                        datapoints = dat_cycle(MC_cl[i].center[k],MC_cl[i].rad[k],30,np)
                        plt.plot(datapoints[:,0],datapoints[:,1],sph_color[i]) 
                        plt.plot(MC_cl[i].center[k,0],MC_cl[i].center[k,1],sph_color[i])
                
    plt.show()


def plot2DSphCluster(olddata,olddata_tar,MC,pres_data,pres_data_tar,plt,np):
    data_color = ['cx','bx','rx']
    sph_color = ['co','bo','ro']
    pres_color = ['ko','ko','ko']
    uniq_old, ind_old = np.unique(olddata_tar, return_index=True)
    uniq_pres, ind_pres = np.unique(pres_data_tar, return_index=True)
    uniq_sph, ind_sph = np.unique(MC['cl'], return_index=True)
    
    if len(olddata)>0:
        for i in range(len(uniq_old)):
            plt.plot(olddata[olddata_tar == uniq_old[i],0], olddata[olddata_tar == uniq_old[i],1],data_color[i])
    if len(pres_data)>0:
        for i in range(len(uniq_pres)):
            plt.plot(pres_data[pres_data_tar == uniq_pres[i],0], pres_data[pres_data_tar == uniq_pres[i],1],pres_color[i])
    if MC['nSph'] > 0:
        for i in range(MC['nSph']):
            if MC['cl'][i] == -1:
                datapoints = dat_cycle(MC['center'][i],MC['rad'][i],30,np)
                plt.plot(datapoints[:,0],datapoints[:,1],'kx') 
                plt.plot(MC['center'][i,0],MC['center'][i,1],'kx')
            else:
                for k in range(len(sph_color)):
                    if MC['cl'][i] == k:                          
                        datapoints = dat_cycle(MC['center'][i],MC['rad'][i],30,np)
                        plt.plot(datapoints[:,0],datapoints[:,1],sph_color[k]) 
                        plt.plot(MC['center'][i,0],MC['center'][i,1],sph_color[k])
    
    # if len(olddata)>0:
    #     plt.plot(olddata[:,0], olddata[:,1],'co')
    # if len(pres_data) > 0:
    #     plt.plot(pres_data[:,0], pres_data[:,1],'ko')
    # if MC['nSph'] > 0:
    #     for i in range(MC['nSph']):
    #         datapoints = dat_cycle(MC['center'][i],MC['rad'][i],30,np)
    #         plt.plot(datapoints[:,0],datapoints[:,1],'yo') 
    #         plt.plot(MC['center'][i,0],MC['center'][i,1],'bx')
    plt.show()

def conf_rad(alpha,tmp_n,tmp_s,t,np,math): # Compute radius of a sphere.
    degree_f = tmp_n-1 # degree of freedom.
    tmp_t = t.ppf(alpha, degree_f) # compute t- value
    tmp_se = np.mean(tmp_s) # compute mean of average
    tmp_rad = np.array([tmp_t*(tmp_se/math.sqrt(tmp_n))]) #compute radias
    # tmp_rad = np.array([math.sqrt(tmp_se**2/(1+abs(tmp_t)*math.sqrt(2/(tmp_n-1))))])
    return abs(tmp_rad)

def update_sd(xvec,center_new,nVar,ss,ni,math,np):
    if center_new.ndim == 1:
       center_new = np.array([center_new])
    if ss.ndim == 1:
        ss = np.array([ss])
    if xvec.ndim == 1:
        xvec = np.array([xvec])
    s_new = np.array([[0.0]*nVar]*1)
    for idx, x in np.ndenumerate(center_new):     
        tmp_val= math.sqrt(((ni-1)*(ss[0,idx[1]]**2)+((ni+1)/ni)*(x-xvec[0,idx[1]])**2)/ni)
        s_new[0,idx[1]] = tmp_val
    return s_new
    
def update_center(center,xvec,ni):
    centern = (ni*center + xvec)/(ni+1)
    ni = ni+1
    return centern, ni
    
def dat2feat_linear(data,Lvec,LA,np):
    OthoLvec = np.array([-Lvec[1],Lvec[0]])
    AT = np.array([Lvec,OthoLvec])
    data = np.transpose(data)
    dataF = np.matmul(AT,data)
    dataF = np.transpose(dataF)
    return dataF
    
def dat_rotate(data,Lvec,LA,np):
    # data = AT(Odata)
    # data is the point in feature space
    # LA.inv()
    OthoLvec = np.array([-Lvec[1],Lvec[0]])
    AT = np.array([Lvec,OthoLvec])
    ATI = LA.inv(AT)
    data =  np.transpose(data)
    Odata = np.matmul(ATI,data)
    Odata = np.transpose(Odata)
    return Odata
    

def dat_rectang(center,rad,L,npoints,np):
    xstart = center[0][0]-L
    xend = center[0][0]+L
    xrange = abs(xstart - xend)
    xsteps = (xrange)/(npoints-1)
    ystart = center[0][1] - rad
    yend = center[0][1] + rad
    yrange = abs(ystart - yend)
    ysteps = (yrange)/(npoints-1)
    dat = np.array([[0.0]*2]*(npoints*4))
    for i in np.arange(npoints):
        dat[i,0] = xstart + i*xsteps
        dat[npoints+i,0] = dat[i,0]
        
        dat[2*npoints+i,1] = ystart + i*ysteps
        dat[3*npoints+i,1] = dat[2*npoints+i,1]
        
        dat[i,1] = center[0][1] + rad
        dat[npoints+i,1] = center[0][1] - rad
        # tmp_val = center[0] + L
        dat[2*npoints+i,0] = center[0][0] + L
        dat[3*npoints+i,0] = center[0][0] - L
        
    return dat
                
def dat_cycle(center,rad,npoints,np):
    if center.ndim == 1:
        center = np.array([center])
    import math
    xstart = center[0][0]-rad
    xend = center[0][0]+rad
    range1 = xend-xstart
    steps = (range1)/(npoints-1)
    dat = np.array([[0.0]*2]*(npoints*2))
    for i in np.arange(npoints):
        dat[i,0] = xstart + i*steps
        dat[npoints+i,0] = dat[i,0]
        if abs(rad**2-(dat[i,0]-center[0][0])**2)<0.0001:
            tmp_val = 0
        else:
            tmp_val = math.sqrt(rad**2-(dat[i,0]-center[0][0])**2)
        
        dat[i,1] = tmp_val + center[0][1] 
        dat[npoints+i,1] = -tmp_val + center[0][1]
        
    # dat[npoints,1]=xend
    return dat

def distpara(xvec,jvec,unitvec,np):
    # xvec is numpy 1-array (a data vector x)
    # jvec is numpy 2-Darray (center vectors j) 
    # unitvec is numpy 1 array ( eigen vectors w.r.t. jvec j)
    # np is numpy
    delta_xj=xvec-jvec # 2-D array
    if delta_xj.ndim == 1:
        delta_xj = np.array([delta_xj])    
        
    return abs(np.apply_along_axis(sum,1,delta_xj*unitvec))

def distpara_sph(centersph,centercyl,cylLvec,np):
    delta_sphcyl=centercyl-centersph # 2-D array
    if delta_sphcyl.ndim == 1:
        delta_sphcyl = np.array([delta_sphcyl*cylLvec])
    
    return abs(np.apply_along_axis(sum,1,delta_sphcyl*cylLvec))

def distprep(xvec,jvec,unitvec,np,LA):
    #jvec is cylider center vector j.
    #kvec is shpere center vector k
    # np is numpy
    # LA is linalg
    ncen = len(unitvec)
    tmp_dot1=np.array([[0.0]*len(unitvec[0])]*ncen)
    delta_xj=xvec-jvec
    if delta_xj.ndim == 1:
        delta_xj=np.array([delta_xj])
    tmp_dot = np.apply_along_axis(sum,1,delta_xj*unitvec)
    for i in np.arange(ncen):
        tmp_dot1[i] = tmp_dot[i]*unitvec[i]
    # delvec=delta_xj-(np.apply_along_axis(sum,1,delta_xj*unitvec)*unitvec)
    
    delvec = delta_xj-tmp_dot1
    return np.apply_along_axis(LA.norm,1,delvec)

def distprep_sph(centersph,centercyl,cylLvec,np,LA):
    delta_sphcyl=centercyl-centersph
    if delta_sphcyl.ndim == 1:
        delta_sphcyl = np.array([delta_sphcyl])
    delvec=delta_sphcyl-(np.apply_along_axis(sum,1,delta_sphcyl*cylLvec)*cylLvec)
    return np.apply_along_axis(LA.norm,1,delvec)

def dist_sphere(xvec,jvec,rad,np,LA):        
    delvec=xvec-jvec
    if delvec.ndim == 1:
        delvec = np.array([delvec])
    if rad.ndim == 0 or len(rad) == 1:
        tmp_dist=np.apply_along_axis(LA.norm,1,delvec)/rad
    else:
        tmp_dist=np.apply_along_axis(LA.norm,1,delvec/rad)
    return tmp_dist

def dist_sphere_stdise(xvec,jvec,rad,np,LA):        
    delvec=xvec-jvec
    if rad.ndim == 0:
       delvec=delvec/rad
    elif len(rad) == 1:
        delvec=delvec/rad
    else:
        for i in range(len(delvec)):
            delvec[i] = delvec[i]/rad[i]
       # delvec = delvec/rad
    if delvec.ndim == 1:
        delvec = np.array([delvec])
    tmp_dist=np.apply_along_axis(LA.norm,1,delvec)
    tmp_dist = tmp_dist - 1
    return tmp_dist
   
def check_datasphere_in(data,centervec,rad,stdise,np,LA):
    if data.ndim == 1:
        data = np.array([data])
    if stdise == 1:
        tmp_dist = dist_sphere_stdise(data,centervec,rad,np,LA)
        return np.where(tmp_dist <= 0), tmp_dist
    else:
        tmp_dist = dist_sphere(data,centervec,rad,np,LA)
        return np.where(tmp_dist <= rad), tmp_dist
    
def check_closestsphere_dist(xvec,centervec,rad,stdise,np,LA):
    # ncen=len(centervec)
    if stdise == 1:
        tmp_dist = dist_sphere_stdise(xvec,centervec,rad,np,LA)
    else:
        tmp_dist = dist_sphere(xvec,centervec,np,LA)
    min_ind = np.argmin(tmp_dist)
    min_dist =  np.min(tmp_dist)
    # if centervec.ndim == 1:
    #     tmp_rad = rad
    #     distcomp = tmp_dist
    #     min_ind = 0
    # else:
    #     tmp_rad = rad[min_ind]
    #     distcomp = tmp_dist[min_ind] 
    # if  distcomp <= tmp_rad:
    #    ch_in[0]=1
       
    return min_dist, min_ind

def check_closestsphere_in(xvec,centervec,rad,stdise,np,LA):
    # ncen=len(centervec)
    ch_in=np.zeros(1)
    if stdise == 1:
       tmp_dist = dist_sphere_stdise(xvec,centervec,rad,np,LA)
    else:
       tmp_dist = dist_sphere(xvec,centervec,rad,np,LA)
    min_ind=np.argmin(tmp_dist)
    if centervec.ndim == 1:
        tmp_rad = rad
        distcomp = tmp_dist
        min_ind = 0
    else:
        tmp_rad = rad[min_ind]
        distcomp = tmp_dist[min_ind] 
    if stdise == 1:
        if  distcomp <= 0:
            ch_in[0]=1
    else:
        if  distcomp <= tmp_rad:
           ch_in[0]=1
       
    return min_ind, ch_in
        
def check_closestcylind_in(xvec,centervec,Lvec,rad,L,np,LA):
    # ncen=len(centervec)
    ch_in = np.zeros(1)
    tmp_distprep = distprep(xvec,centervec,Lvec,np,LA)
    min_indprep = np.argmin(tmp_distprep) # Get the closest cylinder.
    tmp_distpara = distpara(xvec,centervec,Lvec,np) 
    if tmp_distpara[min_indprep] <= L[min_indprep] and tmp_distprep[min_indprep] <= rad:
        ch_in[0]=1
        
    # for i in np.arange(ncen):
    #     if tmp_distpara[i] <= 2*rad[i] and tmp_distprep[i] <= L[i]+rad[i]:
    #         ch_in[i]=1
            
    return min_indprep, ch_in 

def check_overlab_sphcyl(centersph,centercyl,cylLvec,cylL,rad,np,LA):
    ncyl=len(centercyl)
    ch_overlab = np.zeros(ncyl)
    tmp_distpara = distpara_sph(centersph,centercyl,cylLvec,np)
    tmp_distprep = distprep_sph(centersph,centercyl,cylLvec,np,LA)
    for i in np.arange(ncyl):
        if tmp_distpara[i] <= cylL[i]+rad and tmp_distprep[i] <= 2*rad:
            ch_overlab[i]=1
            
    return ch_overlab
            
def check_closestoverlab_sphcyl(centersph,centercyl,cylLvec,cylL,rad,np,LA):
    ch_overlab = np.zeros(1)
    tmp_distpara = distpara_sph(centersph,centercyl,cylLvec,np)
    tmp_distprep = distprep_sph(centersph,centercyl,cylLvec,np,LA)
    min_indprep = np.argmin(tmp_distprep) # Get the closest cylinder.
    if tmp_distpara[min_indprep] <= cylL[min_indprep]+rad and tmp_distprep[min_indprep] <= 2*rad:
        ch_overlab[0] = 1
    
    return min_indprep, ch_overlab

def update_connected(ind_Sph_k,MC,A,np,LA):
    # The function is an update matrix connection A based on
    for i in np.arange(MC['nMC']):
        if i != ind_Sph_k and A[i,ind_Sph_k] == 0:
            if i<MC['nSph']:
                if dist_sphere(MC['center'][ind_Sph_k],MC['center'][i],np,LA)  < 2*MC['rad'][0]:
                    A[i,ind_Sph_k]=1
                    A[ind_Sph_k,i]=1
            else:
                tmp_distpara = distpara_sph(MC['center'][ind_Sph_k],MC['center'][i-MC['nSph']],MC['Lvec'][i-MC['nSph']],np) 
                tmp_distprep_sph = distprep_sph(MC['center'][ind_Sph_k],MC['center'][i-MC['nSph']],MC['Lvec'][i-MC['nSph']],np,LA)
                if tmp_distpara <= MC['L'][i-MC['nSph']] + MC['rad'][0] and tmp_distprep_sph <= 2*MC['rad'][0] :
                    A[i,ind_Sph_k]=1
                    A[ind_Sph_k,i]=1
    return A    

def add_new_sph(xvec,Init_rad,MC,A,np):
    if xvec.ndim==1:
        xvec = np.array([xvec])
    if MC['nSph']==0:
        MC['center'][0]=xvec[0]
        MC['ni'][0]=1
        MC['rad'][0]=Init_rad        
    else:
        nClyn = MC['nMC']-MC['nSph']
        if nClyn == 0:            
            MC['center'] = np.concatenate((MC['center'],xvec))
            # MC['Lvec']=np.concatenate((MC['Lvec'],[[99999]*len(xvec[0])]))
            MC['ni'] = np.append(MC['ni'], 1)
            MC['rad'] = np.append(MC['rad'], Init_rad)                            
            A = np.concatenate((A,np.array([np.zeros(MC['nMC'])])))
            A = np.concatenate((A,np.array([np.zeros(MC['nMC']+1)]).reshape(MC['nMC']+1,1)),1)
        else:
            tmp_locat = MC['nSph']
            if xvec.ndim == 1:
               np.array([xvec]) 
            MC['center'] = np.insert(MC['center'],tmp_locat,xvec,axis = 0)
            MC['ni'] = np.insert(MC['ni'],tmp_locat,1,axis = 0)
            MC['rad']=np.append(MC['rad'], Init_rad)
            A = np.insert(A,tmp_locat,np.array([np.zeros(MC['nMC'])]),axis=0)
            A = np.insert(A,tmp_locat,np.array([np.zeros(MC['nMC']+1)]),axis=1)
            # np.insert
        # MC['L']=np.append(MC['L'], 0)
        # MC['rad']=np.append(MC['rad'], Init_rad)                               
    MC['nMC']+=1
    MC['nSph']+=1        
    return MC, A   

def add_new_sph2(xvec,Init_rad,MC,tmp_center,tmp_s,np):
    if xvec.ndim == 1:
        xvec = np.array([xvec])
    if tmp_center.ndim == 1:
        tmp_center = np.array([tmp_center])
    if tmp_s.ndim == 1:
        tmp_s = np.array([tmp_s])
    if sum(MC.ni) == 0 :
        MC.center = xvec
        MC.ni = np.array([1])
        MC.rad = Init_rad
        MC.tmp_center = tmp_center
        MC.tmp_s = tmp_s
    else:
        MC.center = np.concatenate((MC.center,xvec))
        MC.ni = np.append(MC.ni, 1)
        MC.rad = np.append(MC.rad, Init_rad)
        MC.tmp_center = np.concatenate((MC.tmp_center,tmp_center))
        MC.tmp_s = np.concatenate((MC.tmp_s,tmp_s))
    # if MC['nSph'] == 0:
    #     MC['center'][0] = xvec[0]
    #     MC['ni'][0] = 1
    #     MC['rad'][0] = Init_rad
    #     MC['tmp_center'][0] = tmp_center
    #     MC['tmp_s'][0] = tmp_s
    # else:
    #     MC['center'] = np.concatenate((MC['center'],xvec))
    #     MC['ni'] = np.append(MC['ni'], 1)
    #     MC['rad'] = np.append(MC['rad'], Init_rad)
    #     MC['tmp_center'] = np.concatenate((MC['tmp_center'],tmp_center))
    #     MC['tmp_s'] = np.concatenate((MC['tmp_s'],tmp_s))
    # MC['nMC']+=1
    # MC['nSph']+=1
    return MC
                    
def distsph(xvec,jvec,np,LA): # Compute center distances from a data point 
# xvec to the another set of data points jvec
    # jvec is a set of vectors
    delta_xj=xvec-jvec
    if delta_xj.ndim==1:
        return LA.norm(delta_xj)
    else:
        return np.apply_along_axis(LA.norm,1,delta_xj)
    
def distsph_stdise(xvec,jvec,rad,np,LA):
    # jvec is a set of vectors
    delta_xj=xvec-jvec
    if delta_xj.ndim==1:
        return LA.norm(delta_xj)/rad
    else:
        return np.apply_along_axis(LA.norm,1,delta_xj)/rad    

def mc_remove(MC,relist,np):
    # remove the set of MC as given by the list relist(remove list)
    MC['center']=np.delete(MC['center'],relist,0)
    MC['ni']=np.delete(MC['ni'],relist,0)
    MC['rad']=np.delete(MC['rad'],relist,0)
    numrem = len(relist)    
    MC['nSph']-=numrem
    MC['nMC']-=numrem
    # MC['Lvec']=np.delete(MC['Lvec'],relist,0)
    # MC['L']=np.delete(MC['L'],relist,0)
    return MC
    
def cons(xvec,Init_rad,MC,A,nump,np,LA):
    # xdim=len(xvec)
    if MC['nMC']-MC['nSph']>0: # if the cylinder set is not empty.
        # find the closet the cylinder micro cluster. 
        tmp_dist=distprep(xvec,MC['center'][MC['nSph']:(MC['nMC'])],MC['Lvec'][0:(MC['nMC']-MC['nSph'])],np,LA) # Line 1 Alg.1
        min_ind=np.argmin(tmp_dist) # Get the index of the closet cylider.
        tmp_distpara=distpara(xvec,MC['center'][MC['nSph']+min_ind],MC['Lvec'][min_ind],np)        
        if (tmp_dist[min_ind]<= MC['rad'][0]) and (tmp_distpara <= MC['L'][min_ind]): # Line 2:Satisfies the conditions in Eq.4
            MC['ni'][MC['nSph']+min_ind]+=1 # update cylinder parameter. Line 3 Alg.1
            return MC, A # Line 4. Alg 1
    if MC['nSph']>0: # Check the sphere cluster exists,       
        # tmp_dist = distsph(xvec,MC['center'][0:MC['nSph']],np,LA) # Line 6 Alg. 1 
        # min_ind=np.argmin(tmp_dist) # Get the closest sphere Lines 7-13. 
        min_ind,ch_in = check_closestsphere_in(xvec,MC['center'][0:MC['nSph']],MC['rad'][0:MC['nSph']],np,LA)        
        if ch_in==1: # Line 7.
            MC['center'][min_ind] = ((MC['ni'][min_ind]/(MC['ni'][min_ind]+1))*MC['center'][min_ind]) + xvec/(MC['ni'][min_ind]+1)  # Line 8
            MC['ni'][min_ind]+=1 #(Line 8)
            if MC['ni'][min_ind] > nump: # Line 9
                nCylin=MC['nMC']-MC['nSph']
                if nCylin>0: # There exist the cylinder micro cluster
                    min_indprep, ch_in = check_closestcylind_in(MC['center'][min_ind],MC['center'][MC['nSph']:MC['nMC']],\
                                                                MC['Lvec'][0:nCylin],MC['rad'][0],MC['L'][0:nCylin],np,LA)
                    # tmp_distpara = distprep(MC['center'][min_ind],MC['center'][MC['nSph']:MC['nMC']],\
                    #                       MC['Lvec'][0:(MC['nMC']-MC['nSph'])],np,LA) # Line 16
                    # min_indpara = np.argmin(tmp_distpara) # Get the closest cylinder.
                    # tmp_distprep = distprep(MC['center'][min_ind],MC['center'][MC['nSph']+min_indpara],\
                    #                       MC['Lvec'][min_indpara],np,LA) 
                    # cond1 = tmp_distpara[min_indpara]<= MC['L'][min_indpara]+MC['rad']   
                    # cond2 = tmp_distprep <= 2*MC['rad']
                    # if (cond1 and cond2): # merge conditions met. Check the conditions for merge. Line 17
                    if ch_in==1: # merge sphere and cylinder conditions met    
                        MC['ni'][MC['nSph']+min_indprep] = MC['ni'][MC['nSph'] + min_indprep] + MC['ni'][min_ind] # Line 18
                        tmp_delta = MC['center'][min_ind] - MC['center'][MC['nSph']+min_indprep] # Line 18  
                        MC['center'][MC['nSph']+min_indprep] = MC['center'][MC['nSph']+min_indprep] +  \
                            (MC['L'][min_indprep]-MC['L'][min_indprep])*\
                                np.sign(np.apply_along_axis(LA.norm,1,tmp_delta*MC['Lvec'][min_indprep]))*\
                                    MC['Lvec'][min_indprep] # Update cylinder center                       
                        MC['L'][min_indprep] = (tmp_distpara[min_indprep]+MC['L'][min_indprep]+MC['rad'])/2 # Line 18    
                        A = update_connected(min_ind,MC,A,np,LA)
                        A[:,MC['nSph']+min_indprep]=np.logical_or(A[:,min_ind], A[:,MC['nSph']+min_indprep])
                        A[MC['nSph']+min_indprep,:]=np.logical_or(A[min_ind,:], A[MC['nSph']+min_indprep,:])
                        A=np.delete(A,min_ind,0)
                        A=np.delete(A,min_ind,1)
                        MC=mc_remove(MC,min_ind,np)    
                        MC['nSph']-=1
                        MC['nMC']-=1
                        return MC, A # Line 21
                    
                # tmp_dist=distsph(MC['center'][min_ind],MC['center'][0:MC['nSph']],np,LA)
                
                tmp_dist = np.zeros(MC['nSph'])
                count=0                
                for i in np.arange(MC['nSph']):
                    if MC['ni'][i] > nump:
                       tmp_dist[i] = distsph(MC['center'][min_ind],MC['center'][i],np,LA)
                       if tmp_dist[i] < 2*MC['rad'][0]:
                           count+=1
                    else:
                       tmp_dist[i]=-1
                if count>1:
                    A = update_connected(min_ind,MC,A,np,LA)
                    tmp_merge = np.logical_and(tmp_dist < 2*MC['rad'][0],tmp_dist >= 0) 
                    ind_merge = np.where(tmp_merge == True)
                    tmp_cenmerge = MC['center'][ind_merge]
                    tmp_center = np.array([np.mean(tmp_cenmerge,0)])
                    tmp_ni = np.sum(MC['ni'][ind_merge]) # get scalar value of summation.
                    if count == 2:
                        tmp_dif = tmp_cenmerge[0] - tmp_cenmerge[1]
                        tmp_centerdiff = tmp_cenmerge - tmp_center
                        tmp_Lvec = np.array([tmp_dif])/LA.norm(tmp_dif)
                    else:                                            
                        tmp_centerdiff = tmp_cenmerge - tmp_center
                        from sklearn.decomposition import PCA
                        pca_center = PCA(n_components=1)
                        pca_center.fit(tmp_centerdiff)
                        tmp_Lvec=pca_center.components_
                        
                    tmp_dot=abs(np.apply_along_axis(sum,1,tmp_centerdiff*tmp_Lvec))
                    MC['nMC']+=1 # Create the new cylinder micro cluster.
                    MC['center']=np.concatenate((MC['center'],tmp_center))
                    MC['ni']=np.append(MC['ni'], tmp_ni)
                    if nCylin==0:
                        MC['Lvec'][0]= tmp_Lvec
                        MC['L'][0]=max(tmp_dot)
                    else:
                        MC['Lvec']=np.concatenate((MC['Lvec'],tmp_Lvec))
                        MC['L']=np.append(MC['L'],max(tmp_dot))   
                    
                    A=np.concatenate((A,np.array([np.zeros(MC['nMC']-1)])))
                    A=np.concatenate((A,np.array([np.zeros(MC['nMC'])]).reshape(MC['nMC'],1)),1)
                    for i in np.arange(count): 
                        A[:,MC['nMC']-1]=np.logical_or(A[:,ind_merge[0][i]], A[:,MC['nMC']-1])
                        A[MC['nMC']-1,:]=np.logical_or(A[ind_merge[0][i],:], A[MC['nMC']-1,:])
                    A[MC['nMC']-1,MC['nMC']-1] = 0    
                    # MC['nMC']-=count
                    A = np.delete(A,ind_merge,0)
                    A = np.delete(A,ind_merge,1)
                    MC = mc_remove(MC,ind_merge[0],np)
                    return MC, A # Line 29    
                else:
                    return MC, A
                                                                                  
            return MC,A
        
        MC, A =add_new_sph(xvec,Init_rad,MC,A,np)  
        return MC,A
    else:
        MC, A =add_new_sph(xvec,Init_rad,MC,A,np)
        return MC, A
         
       
           
                
                                      
                
                    
                               
                      
                 
            
            
        
        
    
    