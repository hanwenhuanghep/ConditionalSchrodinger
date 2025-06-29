import numpy as np

def m1(cond):
    target = cond[:, 0]**2 + np.exp(cond[:, 1] + cond[:, 2]/4) + np.cos(cond[:, 3] + cond[:, 4]) + 0.1*np.random.randn(cond.shape[0])
    return target[:, None]

def m2(cond):
    z = np.random.randn(cond.shape[0])
    target = cond[:, 0]**2 + np.exp(cond[:, 1] + cond[:, 2]/4) + cond[:, 3] - cond[:, 4] + (1 + cond[:, 1]**2 + cond[:, 4]**2) / 2 * z
    return target[:, None]

def m3(cond):
    choices = np.random.rand(cond.shape[0], 1)
    target = 0.25*np.random.randn(cond.shape[0], 1) + cond * np.where(choices>0.5, 1, -1)
    return target

def m4(cond):
    target = np.tanh(cond[:, 0]) + np.random.gamma(1,0.3,cond.shape[0])
    return target[:, None]

def m5(cond):
    z=np.sqrt(0.05)*np.random.randn(cond.shape[0])
    target = np.tanh(cond[:, 0]+z)
    return target[:, None]

def m6(cond):
    z=np.random.gamma(1,0.3,cond.shape[0])
    target = z*np.tanh(cond[:, 0])
    return target[:, None]

def m7(cond):
    prefact = 5+cond[:,0]**2/3+cond[:,1]**2+cond[:,2]**2+cond[:,3]+cond[:,4]    
    choices = np.random.rand(cond.shape[0])
    nmean=np.where(choices>0.5,2,-2)
    target = prefact*np.exp(0.5*(np.random.randn(cond.shape[0])+nmean)) 
    return target[:, None]

def compute_target_mean_and_sd(data_name, cond):
    if data_name == 'm1':
        mean = cond[:, 0]**2 + np.exp(cond[:, 1] + cond[:, 2]/4) + np.cos(cond[:, 3] + cond[:, 4])
        sd = np.ones_like(mean)
    elif data_name == 'm2':
        mean = cond[:, 0]**2 + np.exp(cond[:, 1] + cond[:, 2]/4) + cond[:, 3] - cond[:, 4]
        sd = (1 + cond[:, 1]**2 + cond[:, 4]**2) / 2
    elif data_name == 'm3':
        # https://math.stackexchange.com/questions/3689141/calculating-the-mean-and-standard-deviation-of-a-gaussian-mixture-model-of-two-c
        sd = np.sqrt(cond**2 + 0.25**2).squeeze()
        mean = np.zeros_like(sd)
    elif data_name == 'm4':
        mean = np.tanh(cond[:,0])
        sd = 0.3*np.ones_like(mean)
    elif data_name == 'm5':
        length=cond.shape[0]
        nMC=1000
        mean = np.empty_like(cond.shape[0])
        sd = np.empty_like(cond.shape[0])
        conds=np.repeat(cond, nMC, axis=0)
        targets=m5(conds)
        for j in range(length):
            mean[j] = np.mean(targets[j*nMC:(j+1)*nMC,:])
            sd[j] = np.std(targets[j*nMC:(j+1)*nMC,:])
        mean=mean.flatten()
        sd=sd.flatten()
    elif data_name == 'm6':
        mean = 0.3*np.tanh(cond[:,0])
        sd = 0.3*np.tanh(cond[:,0])
    elif data_name == 'm7':
        prefact = 5+cond[:,0]**2/3+cond[:,1]**2+cond[:,2]**2+cond[:,3]+cond[:,4]            
        mean0 = 0.5*prefact*(np.exp(9/8)+np.exp(-7/8))
        var=0.5*prefact**2*(np.exp(9/2)+np.exp(-7/2))-mean0**2
        sd0 = np.sqrt(var)
        length=cond.shape[0]
        nMC=5000
        mean = np.empty_like(mean0)
        sd = np.empty_like(sd0)
        conds=np.repeat(cond, nMC, axis=0)
        targets=m7(conds)
        for j in range(length):
            mean[j] = np.mean(targets[j*nMC:(j+1)*nMC,0])
            sd[j] = np.std(targets[j*nMC:(j+1)*nMC,0])
        mean=mean.flatten()
        sd=sd.flatten()
        #print(np.max(np.abs(mean-mean0)))
        #print(np.max(np.abs(sd-sd0)))
    else:
        raise ValueError(f'unsupported data_name{data_name}')
    return mean, sd

def get_target_fn(data_name):
    return dict(m1=m1, m2=m2, m3=m3,m4=m4,m5=m5,m6=m6,m7=m7)[data_name]
    