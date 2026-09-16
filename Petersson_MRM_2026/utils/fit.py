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

def bayes(im_file: str, bval_file: str, regime: str, roi_file: str | None = None, outbase: str | None = None, verbose: bool = False, fitK: bool = False, spatial_prior: bool = False, n: int = 2000, burns: int = 1000, ctm: str = 'mean', cval_file: str | None = None, TE_file: str | None = None, Covterm: bool = False) -> None:
    """
    Bayesian fitting of the IVIM model in different regimes

    Arguments:
        im_file:       path to nifti image file
        bval_file:     path to .bval file
        regime:        IVIM regime to model: no (= sIVIM), diffusive (long encoding time) or ballistic (short encoding time)
        roi_file:      (optional) path to nifti file defining a region-of-interest (ROI) from with data is extracted
        outbase:       (optional) basis for output filenames to which e.g. '_D.nii.gz' is added 
        verbose:       (optional) if True, diagnostics during fitting is printet to terminal
        fitK:          (optional) if True, the kurtosis signal representation is used instead of a monoexponential one in the first step
        spatial_prior: (optional) if True, a spatial prior enforcing similary between 4-neighbours is applied 
        n:             (optional) number of Markov Chain Monte Carlo (MCMC) iterations
        burns:         (optional) number of MCMC iterations before sampling
        ctm:           (optional) central tendency measure (mean or mode) used to summarize the posterior parameter distributions
        cval_file:     (optional) path to .cval file
        TE_file:       (optional) path to .TE file
        Covterm:       (optional) if True, a covariance term is included in the model
    """

    def _estimation(fn, Y, X, P0, lims, n=500, burns=500, ctm='mean', spatial_prior = False, roi=None, verbose=False):
        """
        Perform a Bayesian parameter estimation and return posterior parameter
        values and their standard deviations.
        """

        ##################
        # Error handling #
        ##################
        n = int(n)
        burns = int(burns)

        #########################
        # Parameter preparation #
        #########################
        mask = valid_signal(Y)
        V = np.sum(mask)  # Y.shape[0]
        pars = P0.shape[1]

        if ctm == 'mean':
            meanonly = True
        elif not ((ctm == 'median') or (ctm == 'mode')):
            raise ValueError(f'Unknown central tendency measure "{ctm}".')
        else:
            meanonly = False

        if meanonly:
            thetasum = np.zeros((V, pars))
            theta2sum = np.zeros((V, pars))

        # Burn-in parameters
        burnUpdateInterval = 100
        burnUpdateFraction = 1

        ########################
        # Parameter estimation #
        ########################

        # Initialize parameter vector.
        if meanonly:
            theta = np.zeros((V, pars, 2))
        else:
            theta = np.zeros((V, pars, n))

        theta[..., 0] = P0[mask, :]

        if spatial_prior:
            if roi is None:
                raise ValueError('A mask is required for the spatial prior.')
            else:
                roi = roi.astype(bool)
            roi[roi] &= mask
            neighbour_mask = neighbours(roi)

        # Step length parameter
        w = theta[..., 0]/10
        N = np.zeros_like(w)  # Number of accepted samples

        # Prior from previous iteration
        prior_old = np.ones_like(P0[mask, :])
        
        # Iterate for j = 0, 1, 2,..., n-1.
        for j in range(n + burns):
            # Initialize theta(j).
            if j > 0:
                if meanonly or (j < burns+1):
                    theta[:, :, 1] = theta[:, :, 0]
                    thetanew = theta[:, :, 1]
                    thetaold = theta[:, :, 0]
                else:
                    theta[:, :, j - burns] = theta[:, :, j - 1 - burns]
                    thetanew = theta[:, :, j - burns]
                    thetaold = theta[:, :, j - 1 - burns]
            else:
                # First iteration
                thetanew = theta[:, :, 0]
                thetaold = theta[:, :, 0]

            # Sample each parameter.
            for k in range(pars):
                # Take a step in parameter space.
                if j > 0:
                    thetanew[:, k] = thetanew[:, k] + np.random.randn(V)*w[:, k]

                # Calculate prior probability.
                prior = ((thetanew[:, k] >= lims[0, k]) & (thetanew[:, k] <= lims[1, k])).astype(float)

                if spatial_prior:
                    theta_neighbours = np.concatenate((thetanew[:,k],np.full(1,np.nan)),axis=0)[neighbour_mask]
                    prior *= np.exp(-np.nansum(np.abs((theta_neighbours-thetanew[:,k][:,np.newaxis])),axis=1)/np.abs(P0[mask,k]))

                # Calculate posterior probability ratio.
                post_ratio = np.zeros_like(prior)
                ssq_new = np.sum((Y[mask, :] - fn(X, thetanew))**2, axis=1)
                ssq_old = np.sum((Y[mask, :] - fn(X, thetaold))**2, axis=1)
                nonzero = (ssq_old > 0) #& (prior_old[:,k]>0)
                post_ratio[nonzero] = ((ssq_new[nonzero] / ssq_old[nonzero])**(-X.shape[0]/2) 
                                        * prior[nonzero]/prior_old[nonzero,k])

                # Evaluate parameter step.
                sample_ok = np.random.rand(V) < post_ratio
                thetanew[~sample_ok, k] = thetaold[~sample_ok, k]  # Reject samples.
                N[:, k] = N[:, k] + sample_ok
                prior_old[sample_ok, k] = prior[sample_ok]
            
            # Prepare for next iteration.
            if meanonly or (j < burns):
                theta[:, :, 0] = thetanew
            else:
                theta[:, :, j-burns] = thetanew
            
            # Save parameter value after burn-in phase.
            if meanonly and j > burns:
                thetasum = thetasum + thetanew
                theta2sum = theta2sum + thetanew**2
            
            # Adapt step length.
            if (j <= burns*burnUpdateFraction) and ((j+1)%burnUpdateInterval == 0):
                w = w * (burnUpdateInterval+1) / (2*((burnUpdateInterval+1)-N))
                N[...] = 0

            # Give update.
            if verbose and ((j%100 == 0) or (j == (n+burns-1))):
                print(f'Iteration {j+1}/{n+burns}')
        
        # Saves distribution measures.
        P = np.full(P0.shape, np.nan)
        std = np.full(P0.shape, np.nan)
        if meanonly:
            P[mask, :] = thetasum/n                              # Mean
            std[mask, :] = np.sqrt(theta2sum/n-(thetasum/n)**2)  # Standard deviation
        else:
            for k in range(P.shape[1]):
                if ctm == 'median':
                    P[mask, k] = np.median(theta[:, k, :], axis=1)
                elif ctm == 'mode':
                    P[mask, k] = halfSampleMode(theta[:, k, :])
                std[mask, k] = np.std(theta[:, k, :], axis=1)

        return P,std
    
    check_regime(regime)

    if regime == BALLISTIC_REGIME:
        Y, b, c = data_from_file(im_file, bval_file, cval_file=cval_file, roi_file=roi_file)
    else:
        Y, b = data_from_file(im_file, bval_file, roi_file = roi_file)
    if TE_file is not None:
        TE = read_time(TE_file)


    if regime == DIFFUSIVE_REGIME:
        if fitK:
            def fn(X, P):
                D, f, Dstar, S0, K = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]
                b = X
                return diffusive(b, D, f, Dstar, S0, K)
        else:
            def fn(X, P):
                D, f, Dstar, S0 = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
                b = X
                return diffusive(b, D, f, Dstar, S0)
    elif regime == BALLISTIC_REGIME:
        if fitK:
            def fn(X, P):
                D, f, vd, S0, K = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]
                b, c = X[:, 0], X[:, 1]
                return ballistic(b, c, D, f, vd, S0, K)
        else:
            def fn(X, P):
                D, f, vd, S0 = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
                b, c = X[:, 0], X[:, 1]
                return ballistic(b, c, D, f, vd, S0)
    else:
        if TE_file is None:
            if fitK:
                def fn(X, P):
                    D, f, S0, K = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
                    b = X
                    return sIVIM(b, D, f, S0, K)
            else:
                def fn(X, P):
                    D, f, S0 = P[:, 0], P[:, 1], P[:, 2]
                    b = X
                    return sIVIM(b, D, f, S0)
        else: ## NEW CODE for T2-sIVIM
            if fitK:
                def fn(X, P):
                    D, f, S0, K, T2d, T2p = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4], P[:, 5]
                    if Covterm:
                        H = P[:,6]
                    else:
                        H = None
                    b, TE = X[:,0], X[:,1]
                    return sIVIM(b, D, f, S0, K, TE=TE,T2d=T2d,T2p=T2p,H=H,Covterm=Covterm)
            else:
                def fn(X, P):
                    D, f, S0, T2d, T2p = P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]
                    if Covterm:
                        H = P[:,5]
                    else:
                        H = None
                    b, TE = X[:,0], X[:,1]
                    return sIVIM(b, D, f, S0, TE=TE,T2d=T2d,T2p=T2p,H=H,Covterm=Covterm)
    
    npars = 4 + fitK - (regime == SIVIM_REGIME)+(TE_file is not None)*2+(Covterm)*1
    P0 = np.zeros((Y.shape[0], npars))
    P0[:, 0] = 1e-3 #D
    P0[:, 1] = 0.05 #f
    lims = np.array([[0, 0, 0], [3e-3, 1, 2*np.max(Y)]])
    if regime == DIFFUSIVE_REGIME:
        P0[:, 2] = 10e-3 #Dstar
        lims = np.insert(lims, 2, [0, 1.0], axis = 1)
        idxS0 = 3
        idxK = 4
    elif regime == BALLISTIC_REGIME:
        P0[:, 2] = 2.0 # Vd
        lims = np.insert(lims, 2, [0, 5.0], axis = 1)
        idxS0 = 3
        idxK = 4
    else:
        idxS0 = 2
        idxK = 3
    P0[:, idxS0] = np.mean(Y[:, b==0], axis = 1)
    if fitK:
        P0[:, idxK] = 1.0 #K
        lims = np.hstack((lims, np.array([0, 5])[:, np.newaxis]))
    if TE_file is not None:
        if fitK:
            idxT2d = 4
            idxT2p = 5
        else:
            idxT2d = 3
            idxT2p = 4
        P0[:, idxT2d] = 100e-3 #T2d
        P0[:, idxT2p] = 100e-3 #T2p
        lims = np.hstack((lims, np.array([5e-3, 500e-3])[:, np.newaxis])) #T2d
        lims = np.hstack((lims, np.array([5e-3, 500e-3])[:, np.newaxis])) #T2p
        if Covterm:
            idxH = idxT2p+1
            P0[:,idxH] = 1e-3
            lims = np.hstack((lims, np.array([-10e-3, 10e-3])[:, np.newaxis])) #H
    
    if regime == BALLISTIC_REGIME:
        X = np.stack((b, c), axis=1)
    elif TE_file is not None:
        X = np.stack((b, TE),axis=1)
    else:
        X = b
    P,_ = _estimation(fn, Y, X, P0, lims, n=n, burns=burns, ctm=ctm, spatial_prior=spatial_prior, roi=read_im(roi_file), verbose=verbose)
  
    pars = {'D': P[:, 0], 'f': P[:, 1], 'S0': P[:, idxS0]}
    if regime == DIFFUSIVE_REGIME:
        pars['Dstar'] = P[:, 2]
    if regime == BALLISTIC_REGIME:
        pars['vd'] = P[:, 2]
    if fitK:
        pars['K'] = P[:, idxK]
    if TE_file is not None:
        pars['T2d'] = P[:, idxT2d]
        pars['T2p'] = P[:, idxT2p]
    if Covterm:
        pars['H'] = P[:, idxH]
    save_parmaps(pars, outbase, im_file, roi_file)

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

