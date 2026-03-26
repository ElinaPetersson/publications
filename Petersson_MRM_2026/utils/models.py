""" Functions to generate MR signal and corresponding Jacobians based on IVIM parameters. 
    20250401: same as OPT_00_models_new but with T2 relaxation too!
"""

import numpy as np
import numpy.typing as npt
from ivim.constants import Db, y
from ivim.seq.sde import MONOPOLAR, BIPOLAR, G_from_b

SIVIM_REGIME = 'sIVIM'
SBALLISTIC_REGIME = 'sballistic'
DIFFUSIVE_REGIME = 'diffusive'
BALLISTIC_REGIME = 'ballistic'
INTERMEDIATE_REGIME = 'intermediate'


def monoexp(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the monoexponential e^(-b*D).
    
    Arguments:
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]

    Output:
        S: (N+1)D array of signal values
    """
    [b, D] = at_least_1d([b, D])
    if TE is None or T2d is None:
        S = np.exp(-np.outer(D, b))
        return np.reshape(S, list(D.shape) + [b.size]) # reshape as np.outer flattens D if ndim > 1
    else:
        [TE, T2d] = at_least_1d([TE, T2d])
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,T2d] = at_lest_right_dim_tissue([D,T2d])
        return np.exp(-D*b) * np.exp(-TE/T2d) # [N,nb]

def kurtosis(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], K: npt.NDArray[np.float64], TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the kurtosis signal representation.
    
    Arguments: 
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]
        K: ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S: (N+1)D array of signal values
    """        
    [b, D, K] = at_least_1d([b, D, K])
    if TE is None or T2d is None:
        Slin = monoexp(b, D)
        Squad = np.exp(np.reshape(np.outer(D, b)**2, list(D.shape) + [b.size]) * K[..., np.newaxis]/6)
        return Slin * Squad
    else:
        [TE, T2d] = at_least_1d([TE, T2d])
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,K,T2d] = at_lest_right_dim_tissue([D,K,T2d])
        Slin = monoexp(b, D, TE, T2d)
        Squad = np.exp((D*b)**2*K/6)
        return Slin * Squad

def sIVIM(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0, TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None, T2p: npt.NDArray[np.float64] | None = None, Covterm: bool = False, H: npt.NDArray[np.float64] | None = None ) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the simplified IVIM (sIVIM) model.
    
    Arguments: 
        b:  vector of b-values [s/mm2]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S:  (N+1)D array of signal values
    """
    [b, D, f, S0] = at_least_1d([b, D, f, S0])
    if TE is None or T2d is None:
        return S0[..., np.newaxis] * ((1-f[..., np.newaxis]) * kurtosis(b, D, K) + np.reshape(np.outer(f, b==0), list(f.shape) + [b.size]))
    else:
        [TE, T2d] = at_least_1d([TE, T2d])
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,f,K,T2d,T2p,S0] = at_lest_right_dim_tissue([D,f,K,T2d,T2p,S0])
        if not Covterm:
            return S0*((1-f) * kurtosis(b,D,K,TE,T2d) + f * np.exp(-TE/T2p)*(b==0))
        else:
            [H] = at_lest_right_dim_tissue([H])
            return S0*((1-f) * kurtosis(b,D,K,TE,T2d) * np.exp(-TE*b*H) + f * np.exp(-TE/T2p)*(b==0))


def ballistic(b: npt.NDArray[np.float64], c: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], vd: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0, TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None, T2p: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the ballistic IVIM model.
    
    Arguments: 
        b:  vector of b-values [s/mm2]
        c:  vector of c-values [s/mm]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        vd: ND array of velocity disperions [mm/s] (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S:  (N+1)D array of signal values
    """
    [b, c, D, f, vd, S0] = at_least_1d([b, c, D, f, vd, S0])
    if TE is None or T2d is None or T2p is None:
        return S0[..., np.newaxis] * ((1-f[..., np.newaxis])*kurtosis(b, D, K) + f[..., np.newaxis]*monoexp(b, Db)*monoexp(c**2, vd**2))
    else:
        [TE, T2d, T2p] = at_least_1d([TE, T2d, T2p])
        b = at_lest_right_dim_b(b)
        c = at_lest_right_dim_b(c)
        TE = at_lest_right_dim_te(TE)
        [D,f,vd,K,S0,T2d,T2p] = at_lest_right_dim_tissue([D,f,vd,K,S0,T2d,T2p])
        return S0 * ((1-f)*kurtosis(b, D, K, TE, T2d) + f*monoexp(b, Db, TE, T2p)*np.exp(-c**2*vd**2))

def sBallistic(b: npt.NDArray[np.float64], c: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0, TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None, T2p: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the simplified ballistic IVIM model.
    
    Arguments: 
        b:  vector of b-values [s/mm2]
        c:  vector of c-values [s/mm]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S:  (N+1)D array of signal values
    """

    [b, c, D, f, S0] = at_least_1d([b, c, D, f, S0])
    if TE is None or T2d is None or T2p is None:
        return S0[..., np.newaxis] * ((1-f[..., np.newaxis])*kurtosis(b, D, K)+ np.reshape(np.outer(f, c==0), list(f.shape) + [b.size])*monoexp(b, Db))
    else:
        [TE, T2d, T2p] = at_least_1d([TE, T2d, T2p])
        b = at_lest_right_dim_b(b)
        c = at_lest_right_dim_b(c)
        TE = at_lest_right_dim_te(TE)
        [D,f,K,S0,T2d,T2p] = at_lest_right_dim_tissue([D,f,K,S0,T2d,T2p])
        return S0 * ((1-f)*kurtosis(b, D, K, TE, T2d) + f*monoexp(b, Db, TE, T2p)*(c==0))


def diffusive(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], Dstar: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0,TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None, T2p: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the diffusive IVIM model.
    
    Arguments: 
        b:     vector of b-values [s/mm2]
        D:     ND array of diffusion coefficients [mm2/s]
        f:     ND array of perfusion fractions (same shape as D or scalar)
        Dstar: ND array of pseudo-diffusion coefficients [mm2/s] (same shape as D or scalar)
        S0:    (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:     (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S:     (N+1)D array of signal values
    """        
    [b, D, f, Dstar, S0] = at_least_1d([b, D, f, Dstar, S0])
    if TE is None or T2d is None or T2p is None:
        return S0[..., np.newaxis] * ((1-f[..., np.newaxis])*kurtosis(b, D, K) + f[..., np.newaxis]*monoexp(b, Dstar))
    else:
        [TE, T2d, T2p] = at_least_1d([TE, T2d, T2p])
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,f,Dstar,K,S0,T2d,T2p] = at_lest_right_dim_tissue([D,f,Dstar,K,S0,T2d,T2p])
        return S0 * ((1-f)*kurtosis(b,D,K,TE,T2d)+f*monoexp(b,Dstar,TE,T2p))

def intermediate(b: npt.NDArray[np.float64], delta: npt.NDArray[np.float64], Delta: npt.NDArray[np.float64], D: npt.NDArray[np.float64], 
                 f: npt.NDArray[np.float64], v: npt.NDArray[np.float64], tau: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, 
                 K: npt.NDArray[np.float64] = 0, seq = MONOPOLAR, T: npt.NDArray[np.float64] | None = None, k: npt.NDArray[np.float64] | None = None, 
                 TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None, T2p: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the intermediate IVIM model.
    
    Arguments: 
        b:     vector of b-values [s/mm2]
        delta: vector of gradient durations [s] (same shape as b)
        Delta: vector of gradient separations [s] (same shape as b)
        D:     ND array of diffusion coefficients [mm2/s]
        f:     ND array of perfusion fractions (same shape as D or scalar)
        v:     ND array of velocities (same shape as D or scalar)
        tau:   ND array of correlation times [s] (same shape as D or scalar)
        S0:    (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:     (optional) ND array of kurtosis coefficients (same shape as D or scalar)
        seq:   pulse sequence used (monopolar or bipolar)
        T:     vector of encoding times [s] (same shape as b)
        k:     vector indicating if bipolar pulse sequence is flow compensated or not [-1/1] (same shape as b)

    Output:
        S:     (N+1)D array of signal values
    """

    [b, delta, Delta, T, k, D, f, v, tau, S0] = at_least_1d([b, delta, Delta, T, k, D, f, v, tau, S0])

    G = G_from_b(b, Delta, delta, seq)

    Deltam = np.reshape(np.outer(np.ones_like(tau), Delta), list(tau.shape) + [Delta.size])
    deltam = np.reshape(np.outer(np.ones_like(tau), delta), list(tau.shape) + [Delta.size])
    Gm     = np.reshape(np.outer(np.ones_like(tau), G), list(tau.shape) + [Delta.size])
    if seq == BIPOLAR:
        Tm     = np.reshape(np.outer(np.ones_like(tau), T), list(tau.shape) + [Delta.size])
        km     = np.reshape(np.outer(np.ones_like(tau), k), list(tau.shape) + [Delta.size])
    taum   = np.reshape(np.outer(tau, np.ones_like(Delta)), list(tau.shape) + [Delta.size])

    t1 = taum * deltam**2 * (Deltam - deltam/3)
    t3 = -2*taum**3 * deltam
    t4 = -taum**4 * (2*np.exp(-Deltam/taum) + 2*np.exp(-deltam/taum) - np.exp(-(Deltam+deltam)/taum) - np.exp(-(Deltam-deltam)/taum) - 2)
    if seq == BIPOLAR:
        t1 *= 2
        t3 *= 2
        t4 *= 2
        t4 += taum**4 * km * np.exp(-Tm/taum)*(np.exp((2*Deltam+2*deltam)/taum) - 2*np.exp((2*Deltam+deltam)/taum) + np.exp(2*Deltam/taum) - 2*np.exp((Deltam+2*deltam)/taum) + 4*np.exp((Deltam+deltam)/taum) - 2*np.exp(Deltam/taum) + np.exp(2*deltam/taum) - 2*np.exp(deltam/taum) + 1)

    Fp = np.exp(-y**2*(v**2/3)[..., np.newaxis]*Gm**2*(t1+t3+t4))
    if TE is None or T2d is None or T2p is None:
        return S0[..., np.newaxis] * ((1-f[..., np.newaxis])*kurtosis(b, D, K) + f[..., np.newaxis]*monoexp(b, Db)*Fp)
    else:
        [TE, T2d, T2p] = at_least_1d([TE, T2d, T2p])
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,f,K,S0,T2d,T2p] = at_lest_right_dim_tissue([D,f,K,S0,T2d,T2p])
        return S0 * ((1-f)*kurtosis(b,D,K,TE,T2d)+f*monoexp(b,Db,TE,T2p)*Fp.squeeze()) 


def monoexp_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """ 
    Return the Jacobian matrix for the monoexponential expression.
    
    S(b) = exp(-b*D)

    Arguments:
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]

    Output: 
        J: Jacobian matrix
    """
    if TE is None or T2d is None:
        # warning! alternative to b[np.newaxis,:] may be needed
        J = (monoexp(b, D) * -b[np.newaxis, :])[...,np.newaxis] # D is the only parameter, but we still want the last dimension
        return J
    else:
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,T2d] = at_lest_right_dim_tissue([D,T2d])
        dSdD = monoexp(b,D,TE,T2d)*-b # [N,nb]*[1,nb]
        dSdT2 = monoexp(b,D,TE,T2d)*TE/T2d**2 # [N,nb]*[1,nte] and nb shoud be the same as nte since b contains repeats
        return [dSdD, dSdT2]

def kurtosis_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], K: npt.NDArray[np.float64], TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """ 
    Return the Jacobian matrix for the monoexponential expression.
    
    S(b) = exp(-b*D + b**2*D**2*K/6)

    Arguments:
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]
        K: ND array of kurtosis coefficients (same shape as D or scalar)

    Output: 
        J: Jacobian matrix
    """

    [b,D,K] = at_least_1d([b,D,K])
    if TE is None or T2d is None:
        J = np.stack([
                    kurtosis(b,D,K)*(-b[np.newaxis, :]+2*np.reshape(np.outer(D*K,b**2)/6,list(D.shape) + [b.size])),
                    kurtosis(b,D,K)*np.reshape(np.outer(D, b)**2/6, list(D.shape) + [b.size])
                    ], axis=-1)
        return J
    else:
        b = at_lest_right_dim_b(b)
        [D,K,T2d] = at_lest_right_dim_tissue([D,K,T2d])
        dSdD = kurtosis(b,D,K,TE,T2d)*(2*D*b**2*K/6-b) 
        dSdK = kurtosis(b,D,K,TE,T2d)*(b*D)**2/6
        dSdT2 = kurtosis(b,D,K,TE,T2d)*TE/T2d**2
        return [dSdD, dSdK, dSdT2]

def sIVIM_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] | None = None, K: npt.NDArray[np.float64] | None = None, TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None, T2p: npt.NDArray[np.float64] | None = None,  Covterm: bool = False, H: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the Jacobian matrix for the simplified IVIM (sIVIM) model.
    
    S(b) = S0((1-f)*exp(-b*D+b^2*D^2*K/6)+fδ(b))

    Arguments: 
        b:  vector of b-values [s/mm2]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        J:  Jacobian matrix
    """

    [b, D, f] = at_least_1d([b, D, f])

    if TE is None or T2d is None or T2p is None:
        if K is None:
            dSdD = (1-f)[..., np.newaxis] * monoexp_jacobian(b,D)[..., 0]
            dSdf = -monoexp(b,D) + (b==0)[np.newaxis, :]
        else:
            [K] = at_least_1d([K])
            dSdD = (1-f)[..., np.newaxis] * kurtosis_jacobian(b,D,K)[..., 0]
            dSdf = -kurtosis(b, D, K) + (b==0)[np.newaxis, :] 
            dSdK = (1-f)[..., np.newaxis] * kurtosis_jacobian(b,D,K)[..., 1]

        if S0 is None:
            if K is None:
                J_list = [dSdD, dSdf]
            else:
                J_list = [dSdD, dSdf, dSdK]
        else:
            [S0] = at_least_1d([S0])
            if K is None:
                dSdS0 = sIVIM(b, D, f)
            else:
                dSdS0 = sIVIM(b, D, f, K=K)
            dSdD *= S0[..., np.newaxis]
            dSdf *= S0[..., np.newaxis]
            if K is None:
                J_list = [dSdD, dSdf, dSdS0]
            else:
                J_list = [dSdD, dSdf, dSdS0, dSdK * S0[..., np.newaxis]]

        J = np.stack(J_list, axis=-1)
        
        return J
    else: # [x*y*z,nb,nte,np]
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,f,T2d,T2p] = at_lest_right_dim_tissue([D,f,T2d,T2p])
        if not Covterm:
            if K is None:
                dSdD = (1-f) * monoexp_jacobian(b,D,TE,T2d)[0]
                dSdf = -monoexp(b,D,TE,T2d) + np.exp(-TE/T2p) * (b==0)
                dSdT2d = (1-f) * monoexp_jacobian(b,D,TE,T2d)[1]
                dSdT2p = f * TE/T2p**2 * np.exp(-TE/T2p) * (b==0)
                if S0 is None:
                    J_list = [dSdD,dSdf,dSdT2d,dSdT2p]
                else:
                    [S0] = at_least_1d([S0])
                    [S0] = at_lest_right_dim_tissue([S0])
                    dSdD *= S0
                    dSdf *= S0
                    dSdT2d *= S0
                    dSdT2p *= S0 
                    dSdS0 = sIVIM(b, D, f,TE=TE,T2d=T2d,T2p=T2p)
                    J_list = [dSdD,dSdf,dSdS0,dSdT2d,dSdT2p]
            else:
                [K] = at_least_1d([K])
                [K] = at_lest_right_dim_tissue([K])
                dSdD = (1-f) * kurtosis_jacobian(b,D,K,TE,T2d)[0]
                dSdf = -kurtosis(b, D, K,TE,T2d) + np.exp(-TE/T2p)*(b==0)
                dSdK = (1-f) * kurtosis_jacobian(b,D,K,TE,T2d)[1]
                dSdT2d = (1-f) * kurtosis_jacobian(b,D,K,TE,T2d)[2]
                dSdT2p = f * TE/T2p**2 * np.exp(-TE/T2p) * (b==0)
                if S0 is None:
                    J_list = [dSdD,dSdf,dSdK,dSdT2d,dSdT2p]
                else:
                    [S0] = at_least_1d([S0])
                    [S0] = at_lest_right_dim_tissue([S0])
                    dSdD *= S0
                    dSdf *= S0
                    dSdK *= S0
                    dSdT2d *= S0
                    dSdT2p *= S0 
                    dSdS0 = sIVIM(b, D, f,K=K,TE=TE,T2d=T2d,T2p=T2p)
                    J_list = [dSdD,dSdf,dSdS0,dSdK,dSdT2d,dSdT2p]
        elif Covterm and H is not None:
            [H] = at_lest_right_dim_tissue([H])
            dSdD = (1-f) * monoexp_jacobian(b,D,TE,T2d)[0]*np.exp(-TE*b*H)
            dSdf = -monoexp(b,D,TE,T2d)*np.exp(-TE*b*H) + np.exp(-TE/T2p) * (b==0)
            dSdT2d = (1-f) * monoexp_jacobian(b,D,TE,T2d)[1]*np.exp(-TE*b*H)
            dSdT2p = f * TE/T2p**2 * np.exp(-TE/T2p) * (b==0)
            dSdH = -(1-f) * TE * b * monoexp(b,D,TE,T2d) * np.exp(-TE*b*H)
            if S0 is None:
                J_list = [dSdD,dSdf,dSdT2d,dSdT2p,dSdH]
            else:
                [S0] = at_least_1d([S0])
                [S0] = at_lest_right_dim_tissue([S0])
                dSdD *= S0
                dSdf *= S0
                dSdT2d *= S0
                dSdT2p *= S0 
                dSdH *= S0
                dSdS0 = sIVIM(b, D, f,TE=TE,T2d=T2d,T2p=T2p, Covterm=Covterm, H=H)
                J_list = [dSdD,dSdf,dSdS0,dSdT2d,dSdT2p,dSdH]
        J = np.stack(J_list, axis=-1)
        return J

def ballistic_jacobian(b:  npt.NDArray[np.float64], c: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], vd: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] | None = None, K: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the Jacobian matrix for the ballistic IVIM model.
    
    S(b) = S0((1-f)*exp(-b*D+b^2*D^2*K/6)+f*exp(-b*Db-vd^2*c*2))

    Arguments: 
        b:  vector of b-values [s/mm2]
        c:  vector of c-values [s/mm]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        vd: ND array of velocity dispersions [mm/s] (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        J:  Jacobian matrix
    """

    [b, D, f, vd] = at_least_1d([b, D, f, vd])
    if S0 is not None:
        [S0] = at_least_1d([S0])
    exp2 = monoexp(b,np.atleast_1d(Db)) * monoexp(c**2,vd**2)

    J_sIVIM = sIVIM_jacobian(b,D,f,S0,K)
    dSdD  = J_sIVIM[..., 0]
    dSdvd = f[..., np.newaxis] * exp2 * (-2*vd[..., np.newaxis]@((c**2)[np.newaxis, :]))
    if S0 is None:
        dSdf  = J_sIVIM[..., 1] - np.ones_like(f)[..., np.newaxis]@(b==0)[np.newaxis, :] + exp2    
    else:
        dSdf  = J_sIVIM[..., 1] - S0[..., np.newaxis]@(b==0)[np.newaxis, :] + S0[..., np.newaxis]*exp2
        dSdvd *= S0[..., np.newaxis]

    if S0 is None:
        if K is None:
            J_list = [dSdD, dSdf, dSdvd]
        else:
            J_list = [dSdD, dSdf, dSdvd, J_sIVIM[..., 2]]
    else:
        if K is None:
            dSdS0 = ballistic(b,c,D,f,vd)
            J_list = [dSdD, dSdf, dSdvd, dSdS0]
        else:
            dSdS0 = ballistic(b,c,D,f,vd,K=K)
            J_list = [dSdD, dSdf, dSdvd, dSdS0, J_sIVIM[..., 3]]

    J = np.stack(J_list, axis=-1)

    return J

def sBallistic_jacobian(b:  npt.NDArray[np.float64], c: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] | None = None, K: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the Jacobian matrix for the simplified ballistic IVIM model.
    
    S(b) = S0((1-f)*exp(-b*D+b^2*D^2*K/6)+f*exp(-b*Db)*delta(c))

    Arguments: 
        b:  vector of b-values [s/mm2]
        c:  vector of c-values [s/mm]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        J:  Jacobian matrix
    """
    # dSdS0 = (1-f)*exp(-b*D+b^2*D^2*K/6)+f*exp(-b*Db)*delta(c)
    # dSdf = -exp(-b*D+b^2*D^2*K/6)+exp(-b*Db)*delta(c)
    # dSdD = (-b+2*b^2*D*K/6)*(1-f)*exp(-b*D+b^2*D^2*K/6)
    
    [b, D, f] = at_least_1d([b, D, f])
    exp2 = monoexp(b,np.atleast_1d(Db))

    if K is None:
        dSdD = (1-f)[..., np.newaxis] * monoexp_jacobian(b,D)[..., 0]
        dSdf = -monoexp(b,D) + exp2*(c==0)[np.newaxis, :]
    else:
        [K] = at_least_1d([K])
        dSdD = (1-f)[..., np.newaxis] * kurtosis_jacobian(b,D,K)[..., 0]
        dSdf = -kurtosis(b,D,K) + exp2*(c==0)[np.newaxis, :]
        dSdK = (1-f)[..., np.newaxis] * kurtosis_jacobian(b,D,K)[..., 1]

    if S0 is None:
        if K is None:
            J_list = [dSdD, dSdf]
        else:
            J_list = [dSdD, dSdf, dSdK]
    else:
        [S0] = at_least_1d([S0])
        if K is None:
            dSdS0 = sBallistic(b, c, D, f)
        else:
            dSdS0 = sBallistic(b, c, D, f, K=K)
        dSdD *= S0[..., np.newaxis]
        dSdf *= S0[..., np.newaxis]
        if K is None:
            J_list = [dSdD, dSdf, dSdS0]
        else:
            J_list = [dSdD, dSdf, dSdS0, dSdK * S0[..., np.newaxis]]


    J = np.stack(J_list, axis=-1)

    return J

def diffusive_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], Dstar: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] | None = None, K: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the Jacobian matrix for the diffusive IVIM model.
    
    S(b) = S0((1-f)*exp(-b*D+b^2*D^2*K/6)+f*exp(-b*D*))

    Arguments: 
        b:     vector of b-values [s/mm2]
        D:     ND array of diffusion coefficients [mm2/s]
        f:     ND array of perfusion fractions (same shape as D or scalar)
        Dstar: ND array of perfusion fractions (same shape as D or scalar)
        S0:    (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:     (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        J:     Jacobian matrix
    """

    [b, D, f, Dstar] = at_least_1d([b, D, f, Dstar])
    if S0 is not None:
        [S0] = at_least_1d([S0])

    J_sIVIM = sIVIM_jacobian(b,D,f,S0,K)
    dSdD  = J_sIVIM[..., 0]
    dSdDstar = f[..., np.newaxis] * monoexp(b,Dstar) * -(np.ones_like(f)[..., np.newaxis]@b[np.newaxis, :])
    if S0 is None:
        dSdf  = J_sIVIM[..., 1] - np.ones_like(f)[..., np.newaxis]@(b==0)[np.newaxis, :] + monoexp(b,Dstar)
    else:
        dSdf  = J_sIVIM[..., 1] - S0[..., np.newaxis]@(b==0)[np.newaxis, :] + S0[..., np.newaxis]*monoexp(b,Dstar)
        dSdDstar *= S0[..., np.newaxis]

    if S0 is None:
        if K is None:
            J_list = [dSdD, dSdf, dSdDstar]
        else:
            J_list = [dSdD, dSdf, dSdDstar, J_sIVIM[..., 2]]
    else:
        [S0] = at_least_1d([S0])
        if K is None:
            dSdS0 = diffusive(b,D,f,Dstar)
            J_list = [dSdD, dSdf, dSdDstar, dSdS0]
        else:
            dSdS0 = diffusive(b,D,f,Dstar,K=K)
            J_list = [dSdD, dSdf, dSdDstar, dSdS0, J_sIVIM[..., 3]]

    J = np.stack(J_list, axis=-1)

    return J

def at_lest_right_dim_tissue(pars: list) -> list:
    """ Makes sure that each tissue parameter has the correct dimension: [N,1] """
    for i, par in enumerate(pars):
        pars[i] = np.array(par).ravel()[...,np.newaxis]
    return pars

def at_lest_right_dim_b(b: npt.NDArray) -> npt.NDArray:
    """ Makes sure that b-vector has the correct dimension: [1,nb*nte] """
    b = b.ravel()[np.newaxis,...]
    return b

def at_lest_right_dim_te(te: npt.NDArray) -> npt.NDArray:
    """ Makes sure that te vector has the correct dimension: [1,nte*nb] """
    te = te.ravel()[np.newaxis,...]
    return te

def at_least_1d(pars: list) -> list:
    """ Check that each parameter is atleast one dimension in shape. """
    for i, par in enumerate(pars):
        pars[i] = np.atleast_1d(par)
    return pars

def check_regime(regime: str) -> None:
    """ Check that the regime is valid. """
    if regime not in [SIVIM_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME,SBALLISTIC_REGIME]:
        raise ValueError(f'Invalid regime "{regime}". Valid regimes are "{SIVIM_REGIME}", "{DIFFUSIVE_REGIME}", "{BALLISTIC_REGIME}" and "{INTERMEDIATE_REGIME}".')


# from OPT_44_models_T2 import sIVIM,sIVIM_jacobian
# import numpy as np
# import matplotlib.pyplot as plt
# from OPT_02_opt_input_pars import usr_input

# b = np.array([0,10,100,500,1000])
# TE = np.array([0.05])
# rtol = 1e-6
# atol_models = 1e-8
# atol_jac = 1e-4

# f_ = np.stack([np.stack([np.linspace(0.01,0.03,5) for _ in range(5)]).T for _ in range(5)]).ravel()
# D_ = np.ones_like(f_)*0.8e-3
# K_ = np.stack([np.stack([np.linspace(0.5,1.5,5) for _ in range(5)]) for _ in range(5)]).ravel()
# T2d_ = np.ones_like(f_)*usr_input['T2d']
# T2p_ = np.ones_like(f_)*usr_input['T2p']
# S0_ = np.stack([np.stack([np.linspace(0.5,1.5,5) for _ in range(5)]) for _ in range(5)]).ravel()

# for D, f, S0, K, T2d, T2p in zip(D_, f_, S0_, K_, T2d_, T2p_):
#     y = sIVIM(b,D,f,S0,K,TE,T2d,T2p)
#     Jlist = [(sIVIM(b,D*(1+rtol),f,S0,K,TE,T2d,T2p) - y) / (np.atleast_1d(D)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f*(1+rtol),S0,K,TE,T2d,T2p) - y) / (np.atleast_1d(f)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0*(1+rtol),K,TE,T2d,T2p) - y) / (np.atleast_1d(S0)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0,K*(1+rtol),TE,T2d,T2p) - y) / (np.atleast_1d(K)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0,K,TE,T2d*(1+rtol),T2p) - y) / (np.atleast_1d(T2d)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0,K,TE,T2d,T2p*(1+rtol)) - y) / (np.atleast_1d(T2p)[..., np.newaxis]*rtol)]

#     Japp = np.stack(Jlist, axis = -1)
#     Jmy = sIVIM_jacobian(b,D,f,S0,K,TE,T2d,T2p)
#     np.testing.assert_allclose(Japp, Jmy, rtol, atol_jac)

# # Med K=0
# for D, f, S0, K, T2d, T2p in zip(D_, f_, S0_, K_*0, T2d_, T2p_):
#     y = sIVIM(b,D,f,S0,K,TE,T2d,T2p)
#     Jlist = [(sIVIM(b,D*(1+rtol),f,S0,K,TE,T2d,T2p) - y) / (np.atleast_1d(D)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f*(1+rtol),S0,K,TE,T2d,T2p) - y) / (np.atleast_1d(f)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0*(1+rtol),K,TE,T2d,T2p) - y) / (np.atleast_1d(S0)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0,K,TE,T2d*(1+rtol),T2p) - y) / (np.atleast_1d(T2d)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0,K,TE,T2d,T2p*(1+rtol)) - y) / (np.atleast_1d(T2p)[..., np.newaxis]*rtol)]
#     Japp = np.stack(Jlist, axis = -1)
#     Jmy = sIVIM_jacobian(b,D,f,S0,None,TE,T2d,T2p)
#     np.testing.assert_allclose(Japp, Jmy, rtol, atol_jac)

# '''
# Får inget error, så jag tror att det funkar ok!
# '''