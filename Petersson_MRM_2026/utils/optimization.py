"""
File contains:
 - Function crlb: remastered function from Oscar to minmize the crlb
 - code to run the optimization and save the results
"""
#%% 
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize, Bounds
from models import sIVIM_jacobian, check_regime, SIVIM_REGIME, INTERMEDIATE_REGIME
from ivim.seq.sde import calc_c, G_from_b, MONOPOLAR, BIPOLAR
from ivim.constants import y as gamma
import time

def crlb(D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], regime: str, 
         bmax: float = 1000, bthr: float = 0, 
         fitK: bool = False, K: npt.NDArray[np.float64] | None = None, SNR: float = 100,
         Dstar: npt.NDArray[np.float64] | None = None, usr_input: dict | None = None, nb_total: int = 14,
         T2d: npt.NDArray[np.float64] | None = None,T2p: npt.NDArray[np.float64] | None = None,
         H: npt.NDArray[np.float64] | None = None, Covterm: bool = False):
    """
    Optimize b-values (and possibly c-values) using Cramer-Rao lower bounds optmization.
    Arguments:
        D:           diffusion coefficients to optimize over [mm2/s]
        f:           perfusion fractions to optimize over (same size as D)
        regime:      IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        bmax:        (optional) the largest b-value that can be returned by the optimization
        fitK:        (optional) if True, optimize with the intention to be able to fit K in addition to D and f
        K:           (optional) kurtosis coefficients to optimize over if fitK and for bias term if minbias
        SNR:         (optional) expected SNR level at b = 0 to be used to scale the influence of the bias term
    ---- sIVIM regime ----
        bthr:        (optional) the smallest non-zero b-value that can be returned by the optimization
    """


    def cost(x, n0 = 0):
        """ 
        x: vector with b-values and possibly fractions 
        n0: number of b = 0 acquisitions (only relevant for regime = 'no') 
        """
        
        nte = 2
        nb = n0 + (x.size-n0-nte) // 2 # x: [b1,...,bn, a0,a1,..,an, te0,te1]=2*n+2+1 # total number of b-values
        b = np.zeros(nb)
        b[n0:] = x[:nb-n0]
        a = x[nb-n0:-nte]
        TE = x[-nte:]

        b = np.tile(b,nte)
        a = np.tile(a,nte)
        TE = np.repeat(TE,nb)

        # EP: Calculate cvalues and k & T if bias_regime == intermediate model:
        r = np.roots([2/3, usr_input['t_180']+usr_input['t_rise'],0,-max(b)*1e6/(gamma**2*usr_input['Gmax']**2)])
        delta = r[(r.real>=0)*(r.imag == 0)][0].real
        Delta = delta + usr_input['t_180']+usr_input['t_rise']

        S0 = np.ones_like(D)
        if regime == SIVIM_REGIME:
            if fitK:
                J = sIVIM_jacobian(b, D, f, S0 = S0, K = K, TE = TE, T2d = T2d, T2p = T2p,Covterm=Covterm,H=H)
            else:
                J = sIVIM_jacobian(b, D, f, S0 , TE = TE, T2d = T2d, T2p = T2p,Covterm=Covterm,H=H)
        # EP: handle steps that make F non-invertible:
        # EP: changed from a -> a*nb to get amount of each b
        # EP: changed from a*nb -> a*14
        F = ((a*nb_total)[np.newaxis,np.newaxis,:]*J.transpose(0,2,1))@J

        try:
            Finv = np.linalg.inv(F)
        except:
            print('Unable to compute inv(F)')
            return np.inf
        
        Finv = Finv/SNR**2

        # C = np.sum(np.sqrt(Finv[:, 0, 0])/D + np.sqrt(Finv[:, 1, 1])/f)
        C = np.sum(np.sqrt(Finv[:, 1, 1])/f) #+ np.sum(np.sqrt(Finv[:, 3, 3])/T2d) + np.sum(np.sqrt(Finv[:, 4, 4])/T2p) 
        
        if fitK:
            idxK = 3 
            C += np.sum(np.sqrt(Finv[:, idxK, idxK])/K)

        return C

    check_regime(regime)

    nb = 4 + fitK - (regime == SIVIM_REGIME)  # total number of b-values
    na = 4 + fitK - (regime == SIVIM_REGIME)
    n0 = (regime == SIVIM_REGIME)
    bmin = bthr*((regime == SIVIM_REGIME)) #+ (regime == SBALLISTIC_REGIME))
    nte = 2

    mincost = np.inf
    
    lb = bmin * np.ones(nb-n0+na+nte) # lower bound for bvalues (and initialization for the rest)
    ub = bmax * np.ones(nb-n0+na+nte) # upper bound for bvalues 
    lb[nb-n0:] = 0.01 # lower bound for fraction of bvalue
    ub[nb-n0:] = 1.0 # upper bound for fraction of bvalues
    lb[-nte:] = 1e-3 # lower bound for echo times
    ub[-nte:] = 10

    bounds = Bounds(lb, ub, keep_feasible = np.full_like(lb, True))

    def TE_low_constraint(x):
        r = np.roots([2/3, usr_input['t_180']+usr_input['t_rise'],0,-max(x[:nb-n0])*1e6/(gamma**2*usr_input['Gmax']**2)])
        delta = r[(r.real>=0)*(r.imag == 0)][0].real
        return np.min(x[-nte:]) - (usr_input['t_180']+usr_input['t_epi']+usr_input['t_rise']*2+2*delta)

    constraints = ({'type':'eq',   'fun':lambda x: np.sum(x[nb-n0:nb-n0+na]) - 1}, # sum(a) = 1
                    {'type':'ineq', 'fun':lambda x: 1.05-np.max(np.exp(x[:nb-n0][np.newaxis,...]**2*D[...,np.newaxis]**2*K[...,np.newaxis]/6))}, # constraint that should affect the maximum b-value most
                    {'type':'ineq', 'fun':lambda x: 0.05-np.max(np.exp(-x[:nb-n0][np.newaxis,...]*Dstar[...,np.newaxis]))}, # constraint that should affect the smallest non-zero b-value most
                    {'type':'ineq', 'fun':lambda x: TE_low_constraint(x)}, # the smallest possible TE
                    {'type':'ineq', 'fun':lambda x: np.max(x[-nte:])-np.min(x[-nte:])-usr_input['t_epi']-usr_input['t_180']}) # Lower limit on TE2: need to fit the 180 and EPI readout

    for seed_idx in range(10):
        print('seed ' + str(seed_idx))
        x0 = 1/na * np.ones(nb-n0 + na + nte)
        x0[:nb-n0] = [np.random.uniform(lb[i],ub[i],size=1)[0] for i in range(nb-n0)]
        
        # Compute start minimum TE based on largest b-value in x0:
        r_temp = np.roots([2/3, usr_input['t_180']+usr_input['t_rise'],0,-max(x0[:nb-n0])*1e6/(gamma**2*usr_input['Gmax']**2)])
        delta_temp = r_temp[(r_temp.real>=0)*(r_temp.imag == 0)][0].real
        x0[-nte:] = [usr_input['t_180']+usr_input['t_epi']+usr_input['t_rise']*2+2*delta_temp for _ in range(int(nte/2))]+[2*(usr_input['t_180']+usr_input['t_epi']+usr_input['t_rise']*2+2*delta_temp) for _ in range(int(nte/2))] # The first nb TE will be the minimum and the last nb TE will be 2*minimum

        cost_regime = lambda x: cost(x, n0)
        res = minimize(cost_regime, x0, bounds = bounds, constraints = constraints, method = 'SLSQP')
        if res.fun < mincost:
            b = np.zeros(nb)
            b[n0:] = res.x[:nb-n0]
            a = res.x[nb-n0:-nte]
            TE = res.x[-nte:]
            mincost = res.fun   
                    
    if (a == 1/na * np.ones(na)).all():
            print('It is likely that an optimum was not found')

    idx = np.argsort(b)
    
    return b[idx],a[idx],TE,mincost



# %%
import time, os
from models import SIVIM_REGIME
import numpy as np
import matplotlib.pyplot as plt
from ivim.constants import y as gamma

usr_input = {'Gmax': 61e-3,'t_epi':28e-3,'t_180':8e-3,'t_rise':0.6e-3} # scanner limitations to be used as constraints in the optimization

# Optimze b and TE
data_path = 'simulated_data'
if not os.path.isdir(data_path): os.makedirs(data_path,exist_ok=True)
if not os.path.isdir(data_path+'/sIVIM_Gmax{}'.format(usr_input['Gmax'] *1000)): os.makedirs(data_path+'/sIVIM_Gmax{}'.format(usr_input['Gmax'] *1000),exist_ok=True)
scheme = 'opt_T2'
regime = SIVIM_REGIME
SNR = 70
bmax = 1000 # s/mm2

f = np.stack([np.stack([np.linspace(0.03,0.1,5) for _ in range(5)]).T for _ in range(5)]).ravel()
D = np.ones_like(f)*1e-3 # larger D --> lower bmax
S0 = np.ones_like(f)
K = np.ones_like(f)*1 # larger K --> lower bmax, must be the largest expectable
T2d = np.stack([np.stack([np.linspace(0.05,0.09,5) for _ in range(5)]) for _ in range(5)]).ravel()
T2p = np.stack([np.ones((5,5))*taui for taui in np.linspace(0.055,0.13,5)]).ravel() 
Dstar = np.ones_like(f)*10e-3 # smaller Dstar --> higher lowest b-value, must be the smallest expectable

res={'b':[],'a':[],'te':[],'mincost':[]}
for _ in range(5):
    t0 = time.time()
    b,a,te,mincost = crlb(D=D, f=f, regime=regime, bthr=0, bmax = bmax, fitK = False,  
                        nb_total=14, K=K, SNR=SNR, usr_input=usr_input,Dstar=Dstar,T2d=T2d, T2p=T2p)
    res['a'].append(a)
    res['b'].append(b)
    res['te'].append(te)
    res['mincost'].append(mincost)
    print(res)
    print('time for SIVIM ' + scheme + str(time.time()-t0))

def plot_opt_result(res):
    for i in range(len(res['b'])):
        idx = np.where(res['a'][i]>0.02)
        plt.plot(res['b'][i][idx],res['a'][i][idx],label=res['te'][i])
    plt.legend()
    plt.show()
    
plot_opt_result(res)
with open(data_path+'/sIVIM_Gmax{}'.format(usr_input['Gmax'] *1000)+'/sdiff_'+scheme+'.txt','w') as data: data.write(str(res))

# Print delta and Delta
r = np.roots([2/3, usr_input['t_180']+usr_input['t_rise'],0,-max(res['b'][res['mincost'].index(min(res['mincost']))])*1e6/(gamma**2*usr_input['Gmax']**2)])
delta = r[(r.real>=0)*(r.imag == 0)][0].real
Delta = delta + usr_input['t_180'] + usr_input['t_rise']
print(f'delta: {delta}, Delta: {Delta}')

# Save and print b-values and TE
def save_opt_bval(res,regime,scheme,data_path):
    b = res['b'][res['mincost'].index(min(res['mincost']))]
    a = res['a'][res['mincost'].index(min(res['mincost']))]
    TE = res['te'][res['mincost'].index(min(res['mincost']))]
    # Convert b-values to the correct amount and save
    n = np.round(a*14).astype(int)
    if sum(n) == 15:
        n[np.argmax(n)]-=1
    elif sum(n) == 13:
        n[np.argmin(n)]+=1
    bvals = [b[i] for i in range(len(n)) for _ in range(n[i])]           
    bvals = np.array(bvals)
    bvals = np.tile(bvals,2)  
    bval_file = data_path+'/bval_' +regime+'_'+ scheme+'.bval'
    np.savetxt(bval_file,bvals)
    # Convert TE to be same number as b and save
    TE=np.repeat(TE,14)
    TE_file = data_path+'/TE_' +regime+'_'+ scheme+'.te'
    np.savetxt(TE_file,TE)
    return bval_file, TE_file

bval_file, TE_file = save_opt_bval(res,regime,scheme,data_path+'/sIVIM_Gmax{}'.format(usr_input['Gmax'] *1000))
print(f'b-values: {np.loadtxt(bval_file)}')
print(f'TE: {np.loadtxt(TE_file)}')
# %%
