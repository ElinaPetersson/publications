""" Functions for IVIM parameter estimation. """
#%%
import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit, least_squares
from scipy.linalg import solve
from models import sIVIM, diffusive, ballistic, intermediate, sBallistic, sIVIM_jacobian, diffusive_jacobian, ballistic_jacobian, sBallistic_jacobian, check_regime, SIVIM_REGIME, DIFFUSIVE_REGIME, BALLISTIC_REGIME, INTERMEDIATE_REGIME, SBALLISTIC_REGIME
from models import monoexp as monoexp_model
from models import kurtosis as kurtosis_model
from ivim.constants import Db
from ivim.seq.sde import MONOPOLAR, BIPOLAR
from ivim.misc import halfSampleMode
from ivim.io.base import data_from_file, file_from_data, read_im, read_time, read_k

def seg_T2_sIVIM(im_file: str, bval_file: str, regime: str, roi_file: str | None = None, outbase: str | None = None, verbose: bool = False, fitK: bool = False, cval_file: str | None = None, TE_file: str | None = None, Covterm: bool = False) -> None:
    """
    Seg fitting of the IVIM model different regimes.

    Arguments:
        im_file:   path to nifti image file, assume N x nbnTE
        bval_file: path to .bval file, assume nb*nTE
        regime:    IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        roi_file:  (optional) path to nifti file defining a region-of-interest (ROI) from with data is extracted
        outbase:   (optional) basis for output filenames to which e.g. '_D.nii.gz' is added 
        verbose:   (optional) if True, diagnostics during fitting is printet to terminal
        fitK:      (optional) if True, the kurtosis signal representation is used instead of a monoexponential one in the first step
        cval_file: (optional) path to .cval file
        TE_file:   (optional) path to TE file, assume nb*nTE
        Covterm:   (optional) bool that specifies if covariance term between D and R2d should be included
    """

    check_regime(regime)
    if regime == SIVIM_REGIME and TE_file is not None:
        TE = read_time(TE_file)
        Y, b = data_from_file(im_file, bval_file, roi_file = roi_file)
        mask = valid_signal(Y)
        Y = Y[mask,:]

        # Step 1: S0*(1-f)*exp(-bD-TE/T2d)
        '''
        LLS metod
        y = A1*exp(-b*D-TE*R2d) --> log(y) = log(A1)-b*D-TE*R2d --> log(y) = X*p_est
        log(y).T: [nb,125000] 
        X = [[1,-b0,-TE0], with one row for each b-value: [nb,3] - not square!
             [1,-b1,-TE1],
             [1,-b2,-TE2],
             [1,-b3,-TE3]]
        p_est = [[log(A1), log(A1), ...,log(A1)], 
                 [D, D, ..., D],
                 [R2d, R2d, ..., R2d]]: [3, 125000]
        '''
        # Find b that are not 0
        bi = np.where(b!=0)[0]

        # Form the design matrix
        X = np.concatenate((np.ones_like(b[bi])[:,np.newaxis],-b[bi][:,np.newaxis],-TE[bi][:,np.newaxis]),axis=1) # [nb,3]
        if Covterm:
            X = np.concatenate((X,-TE[bi][:,np.newaxis]*b[bi][:,np.newaxis]),axis=1) # [nb,3]

        
        # Solve the linear system
        p_est_lls, *_ = np.linalg.lstsq(X,np.log(Y[:,bi]).T)
        # A1_sub = np.exp(p_est_lls[0,:]) # only used if we omitt next step for WLLS
        # D_sub = p_est_lls[1,:]
        # R2d_sub = p_est_lls[2,:]


        '''
        WLLS
        Background:
            Weights according to: https://www.sciencedirect.com/science/article/pii/S1053811913005223?ref=pdf_download&fr=RR-2&rr=935725622c1d92a6
            W = diag(log(2*X*p_est)) : [nb, nb, 125000]
            (X.T*W*X)*p_est_wlls = X.T*W*log(y)
        Choices:
            I take sqrt of the weights and multipy them with both X and y since @ can't deal with batch processing and 
            I use einsum for batch processing since @ can't do that
            I use numpy's solve since scipy's can't do batch processing

        #TODO: perhaps the sqrt step could be replaced with an einsum step, but I am not sure if this is more effecient. Then We would need to make W a diagonal matrix.
        '''
        # Compute weights from LLS parameters 
        W = np.exp(2*X @ p_est_lls) # [nb, 125000]
        W_sqrt = np.sqrt(W) # [nb, 125000], do this to not have to deal with broadcasting that @ can't do

        # Weight X and y separately with sqrt(W) 
        Xw = X[...,np.newaxis] * W_sqrt[:,np.newaxis,:] # [nb,3]*[nb,1,125000]=[nb,3,125000]
        yw = np.log(Y[:,bi].T) * W_sqrt # [nb,125000]*[nb,125000]

        # Perform matrix multiplication batch wise
        XTX = np.einsum('ijk,ilk->jlk',Xw,Xw) # [3,nb,125000] @ [nb,3,125000] = [3,3,125000]
        XTy = np.einsum('ijk,ik->jk',Xw,yw) # [3,nb,125000] @ [nb,125000] = [3,125000]

        # Solve linear system
        p_est_wlls = np.linalg.solve(np.transpose(XTX,(2,0,1)), XTy.T[...,np.newaxis]) # [125000,3,3][125000,3,1]= [125000,3,1] --> [125000,3,1], had to be like this because of numpy's solve function
        A1_ls = np.exp(p_est_wlls[:,0,0])
        D_ls = p_est_wlls[:,1,0]
        R2d_ls = p_est_wlls[:,2,0]
        if Covterm:
            H_ls = p_est_wlls[:,3,0]

        # Make sure parameters are in reasonable limits
        D_ls[D_ls > 3e-3] = 3e-3
        D_ls[D_ls < 0] = 0

        R2d_ls[R2d_ls < 1/500e-3] = 1/500e-3
        R2d_ls[R2d_ls > 1/5e-3] = 1/5e-3
        # A1_ls[A1_ls > 2*np.max(Y)] = 2*np.max(Y) # becomes hacon bacon with this on...
        A1_ls[A1_ls < 0] = 0

        # Step 2: S0*f*exp(-TE/T2p)
        '''
        Anlaytic method
        Y_perf1 = Y(b0,TE1)-A1*exp(-TE1/T2d)
        Y_perf2 = Y(b0,TE2)-A1*exp(-TE2/T2d)
        R2p = log(Y_perf1/Y_perf2)/(TE2/TE1)
        '''
        Y_perf1 = np.nanmean(Y[:,np.where((b==0) * (TE==TE[0]))[0]], axis=-1) - A1_ls*np.exp(-TE[0]*R2d_ls) # mean since we can have multiple b-values = 0
        Y_perf2 = np.nanmean(Y[:,np.where((b==0) * (TE==TE[-1]))[0]], axis=-1) - A1_ls*np.exp(-TE[-1]*R2d_ls)
        R2p_ls = np.log(Y_perf1/Y_perf2)/(TE[-1]-TE[0])

        # make sure parameters are reasonable limits
        R2p_ls[R2p_ls < 1/500e-3] = 1/500e-3
        R2p_ls[R2p_ls > 1/5e-3] = 1/5e-3
        # A2_ls = np.nanmean(np.stack((Y_perf1/np.exp(-TE[0]*R2p_ls),Y_perf2/np.exp(-TE[-1]*R2p_ls)),axis=-1),axis=-1) # this makes no difference and i think it is because the noise has already entered the R2p estimate
        A2_ls = Y_perf2/np.exp(-TE[-1]*R2p_ls)
        
        # make sure parameters are reasonable limits
        A2_ls[A2_ls > 2*np.max(Y)] = 2*np.max(Y)
        A2_ls[A2_ls < 0] = 0

        # Compute S0 and f from the constants A1 and A2
        S0_ls = A1_ls+A2_ls
        f_ls = A2_ls/S0_ls
        
        # make sure parameters are reasonable limits
        f_ls[f_ls < 0] = np.nan
        f_ls[f_ls > 1] = np.nan
        
        if not Covterm:
            p_est = np.full((mask.size,7), np.nan)
        else:
            p_est = np.full((mask.size,8), np.nan)
        p_est[mask,0] = D_ls
        p_est[mask,1] = f_ls
        p_est[mask,2] = S0_ls
        p_est[mask,3] = 1/R2d_ls
        p_est[mask,4] = 1/R2p_ls
        p_est[mask,5] = A1_ls
        p_est[mask,6] = A2_ls
        if Covterm:
            p_est[mask,7] = H_ls

        pars = {'D': p_est[:,0], 'f': p_est[:,1], 'S0': p_est[:,2], 'T2d': p_est[:,3], 'T2p': p_est[:,4], 'A1': p_est[:,5], 'A2': p_est[:,6]}
        if Covterm:
            pars['H'] = p_est[:,7]
        save_parmaps(pars, outbase, im_file, roi_file)
    elif regime == SIVIM_REGIME and TE_file is None:
        Y, b = data_from_file(im_file, bval_file, roi_file = roi_file)
        mask = valid_signal(Y)
        Y = Y[mask,:]

        # Step 1: S0*(1-f)*exp(-bD)
        # Find b that are not 0
        bi = np.where(b!=0)[0]
        # Form the design matrix
        X = np.concatenate((np.ones_like(b[bi])[:,np.newaxis],-b[bi][:,np.newaxis]),axis=1) # [nb,2]
        
        # Solve the linear system
        p_est_lls, *_ = np.linalg.lstsq(X,np.log(Y[:,bi]).T)

        # Compute weights from LLS parameters 
        W = np.exp(2*X @ p_est_lls) # [nb, 125000]
        W_sqrt = np.sqrt(W) # [nb, 125000], do this to not have to deal with broadcasting that @ can't do

        # Weight X and y separately with sqrt(W) 
        Xw = X[...,np.newaxis] * W_sqrt[:,np.newaxis,:] # [nb,3]*[nb,1,125000]=[nb,3,125000]
        yw = np.log(Y[:,bi].T) * W_sqrt # [nb,125000]*[nb,125000]

        # Perform matrix multiplication batch wise
        XTX = np.einsum('ijk,ilk->jlk',Xw,Xw) # [3,nb,125000] @ [nb,3,125000] = [3,3,125000]
        XTy = np.einsum('ijk,ik->jk',Xw,yw) # [3,nb,125000] @ [nb,125000] = [3,125000]

        # Solve linear system
        p_est_wlls = np.linalg.solve(np.transpose(XTX,(2,0,1)), XTy.T[...,np.newaxis]) # [125000,3,3][125000,3,1]= [125000,3,1] --> [125000,3,1], had to be like this because of numpy's solve function
        A1_ls = np.exp(p_est_wlls[:,0,0])
        D_ls = p_est_wlls[:,1,0]

        # Make sure parameters are in reasonable limits
        D_ls[D_ls > 3e-3] = 3e-3
        D_ls[D_ls < 0] = 0

        A1_ls[A1_ls < 0] = 0

        # Step 2: S0
        S0_ls = np.nanmean(Y[:,np.where(b==0)[0]], axis=-1) # mean since we can have multiple b-values = 0
            
        # make sure parameters are reasonable limits
        S0_ls[S0_ls > 2*np.max(Y)] = 2*np.max(Y)
        S0_ls[S0_ls < 0] = 0

        # Compute S0 and f from the constants A1 and A2
        f_ls = 1 - A1_ls/S0_ls
        
        # make sure parameters are reasonable limits
        f_ls[f_ls < 0] = np.nan
        f_ls[f_ls > 1] = np.nan
        
        p_est = np.full((mask.size,5), np.nan)
        p_est[mask,0] = D_ls
        p_est[mask,1] = f_ls
        p_est[mask,2] = S0_ls
        p_est[mask,3] = A1_ls

        pars = {'D': p_est[:,0], 'f': p_est[:,1], 'S0': p_est[:,2], 'A1': p_est[:,3]}
        save_parmaps(pars, outbase, im_file, roi_file)
    else:
        print('Not applicable choice yet...')

def save_parmaps(pars: dict, outbase: str | None = None, imref_file: str | None = None, roi_file: str | None = None) -> None:
    """
    Save IVIM parameter data (vector format) as nifti images

    Arguments:
    pars       -- parameter data in format {par_name: par_value}, e.g. {'D': D, 'f': f}
    outbase    -- (optional) basis for output filenames to which e.g. '_D.nii.gz' is added 
    imref_file -- (optional) path to nifti file from which header info is obtained
    roi        -- (optional) region-of-interest from which data is assumed to originate. The number of True elements must match the size of the parameter vector

    Note! A subset of the optional arguments must be given:
    - if outbase is not set, it is derived from imref_file
    - if imref_file is not set, the image size is derived from roi_file
    i.e. valid argument combinations are:
    - outbase + imref_file + roi_file
    - outbase + roi_file
    - outbase + imref_file
    - imref_file + roi_file
    - imref_file
    """
    
    if imref_file == None:
        if outbase == None:
            raise ValueError('Either outbase or imref_file must be set.')
        # Remaining invalid combinations are handled by file_from_data
    else:
        if outbase == None:
            outbase = imref_file.split('.')[0]
    
    for parname, par in pars.items():
        par_trimmed = trim_par(par, parname)
        filename = outbase + '_' + parname + '.nii.gz'
        file_from_data(filename, par_trimmed, roi = read_im(roi_file), imref_file = imref_file)

def trim_par(par: npt.NDArray[np.float64], parname: str) -> npt.NDArray[np.float64]:
    """
    Trim parameter values beyond reasonable limits to avoid numerical error when saving to file.
    
    Arguments:
    par     -- vector with parameter values
    parname -- name of parameter value ('D', 'f', 'Dstar', 'vd' or 'K')
    
    Output:
    par     -- vector with trimmed parameter values
    """

    lims = {'D':10e-3, 'f':1, 'Dstar':1, 'vd':20, 'K':20, 'v':20, 'tau':10}
    if parname in lims.keys():
        par = np.clip(par, -lims[parname], lims[parname])
    return par

def valid_signal(Y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Return a mask representing all rows in Y with valid values (not non-positive, NaN or infinite).

    Arguments:
    Y    -- v x b matrix with data

    Output:
    mask -- vector of size v indicating valid rows in Y
    """

    mask = ~np.any((Y<=0) | np.isnan(Y) | np.isinf(Y), axis=1)
    return mask

def neighbours(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
    """
    Return an array with index of all 4-neighbours for True elements in mask.

    Arguments:
    mask           -- 3D array identifying a mask in an image 

    Output:
    neighbour_mask -- array with index of all 4-neighbours for True elements in mask

    Note! index of neighbours outside the mask is set to the maximum index + 1
    """

    N = np.sum(mask)
    index_map = np.full(np.array(mask.shape)+2, N) # pad by 1 on each side
    index_map[1:-1,1:-1,1:-1][mask] = np.arange(N) 
    neighbour_mask = np.stack((index_map[0:-2,1:-1,1:-1][mask],
                                index_map[2:,1:-1,1:-1][mask],
                                index_map[1:-1,0:-2,1:-1][mask],
                                index_map[1:-1,2:,1:-1][mask],
                                ),axis=1)
    return neighbour_mask

