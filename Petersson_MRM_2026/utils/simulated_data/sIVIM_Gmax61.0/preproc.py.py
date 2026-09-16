#%% 
import nibabel as nib
import subprocess, os, copy
import numpy as np
from nipype.interfaces import fsl
import itk
from ivim.preproc.base import average, combine, extract
from ivim.preproc import signal_drift
from ivim.io.base import read_bval, write_bval
from ivim.fit import valid_signal, save_parmaps
from ivim.io.base import read_bval, read_im, read_time, write_time, write_im, data_from_file


def denoise_wrap(im_file):
    ''' Denoising '''
    im_file_denoised = im_file.replace('.nii.gz','-pca.nii.gz')
    noise_level = im_file.replace('.nii.gz','-noise_level.nii.gz')
    # if not os.path.isfile(im_file_denoised):
    print('Denoising...')
    subprocess.run(['dwidenoise', '-noise', noise_level, im_file, im_file_denoised,'-force'])
    subprocess.run(['cp',im_file.replace('.nii.gz','.bval'),im_file_denoised.replace('.nii.gz','.bval')])
    subprocess.run(['cp',im_file.replace('.nii.gz','.bvec'),im_file_denoised.replace('.nii.gz','.bvec')])
    return im_file_denoised

def degibbs_wrap(im_file):
    ''' Gibbs ringing artifact removal '''
    im_file_degibbs = im_file.replace('.nii.gz','-gib.nii.gz')
    # if not os.path.isfile(im_file_degibbs):
    print('Degibbsing...')
    subprocess.run(['mrdegibbs', im_file, im_file_degibbs,'-force'])
    subprocess.run(['cp',im_file.replace('.nii.gz','.bval'),im_file_degibbs.replace('.nii.gz','.bval')])
    subprocess.run(['cp',im_file.replace('.nii.gz','.bvec'),im_file_degibbs.replace('.nii.gz','.bvec')])
    return im_file_degibbs

def extrapolate_register(in_file_e1_raw, in_file_e2_raw, preproc_dir, scheme):
    '''
    Extrapolate and register DWI data.
    '''

    # Create reg preproc directory
    reg_preproc_dir = os.path.join(preproc_dir,'reg')
    os.makedirs(reg_preproc_dir,exist_ok=True)

    # Copy input files to reg preproc directory
    in_file_e1 = os.path.join(reg_preproc_dir,os.path.basename(in_file_e1_raw))
    in_file_e2 = os.path.join(reg_preproc_dir,os.path.basename(in_file_e2_raw))
    subprocess.run(['cp',in_file_e1_raw, in_file_e1])
    subprocess.run(['cp',in_file_e2_raw, in_file_e2])
    subprocess.run(['cp',in_file_e1_raw.replace('.nii.gz','.bval'),in_file_e1.replace('.nii.gz','.bval')])
    subprocess.run(['cp',in_file_e2_raw.replace('.nii.gz','.bval'),in_file_e2.replace('.nii.gz','.bval')])

    # Identify index to use as registration reference (last b0)
    b = read_bval(in_file_e1.replace('.nii.gz','.bval'))
    ref_idx = np.argwhere(b==0)[-1][0]

    # Extract reference b0 image from the last b0
    ref_file = in_file_e1.replace('.nii.gz','-b0.nii.gz')
    subprocess.run(['fslroi', in_file_e1, ref_file, str(ref_idx), '1'])

    for in_file in [in_file_e1, in_file_e2]:
        # Iterate over each volume
        registered_vols = []
        n_vols = nib.load(in_file).shape[3]
        for t in range(n_vols):
            print(f"Registering volume {t+1}/{n_vols} to b0...")
            # Skip registration for specified indices
            if (in_file == in_file_e1) and (t == ref_idx): 
                registered_vols.append(ref_file)
            else:
                in_file_t = extract_volume(in_file, t)
                register_3D_to_ref(in_file=in_file_t, ref_file=ref_file, out_file=in_file.replace('.nii.gz',f'-reg{t}.nii.gz'), method='rigid')
                registered_vols.append(in_file.replace('.nii.gz',f'-reg{t}.nii.gz'))
                    
        out_file = in_file.replace('.nii.gz','-rigid.nii.gz')
        # Stack back to 4D
        subprocess.run(['fslmerge','-t',out_file]+registered_vols)
        print(f"Saved registered 4D DWI to {out_file}")
        for vol in registered_vols:
            if vol != ref_file:
                os.remove(vol)  # Clean up temp registered images

    in_file_e1_loop = copy.deepcopy(in_file_e1.replace('.nii.gz','-rigid.nii.gz'))
    in_file_e2_loop = copy.deepcopy(in_file_e2.replace('.nii.gz','-rigid.nii.gz'))

    # Run the extrapolation-registration loop N times
    N = 1 # I tried 2 times and saw no improvement, so I set it to 1 for now. Can be increased if needed.
    for i in range(N):
        # Average for each echo
        average(im_file=in_file_e1_loop,
                bval_file=in_file_e1.replace('.nii.gz','.bval'),
                outbase=in_file_e1.replace('.nii.gz','_avg'))
        average(im_file=in_file_e2_loop,
                bval_file=in_file_e2.replace('.nii.gz','.bval'),
                outbase=in_file_e2.replace('.nii.gz','_avg'))
        combine(dwi_files=[in_file_e1.replace('.nii.gz','_avg.nii.gz'),
                        in_file_e2.replace('.nii.gz','_avg.nii.gz')],
                bval_files=[in_file_e1.replace('.nii.gz','_avg.nii.gz').replace('.nii.gz','.bval'),
                            in_file_e2.replace('.nii.gz','_avg.nii.gz').replace('.nii.gz','.bval')],
                outbase=os.path.join(reg_preproc_dir,'ivim_opt_' + scheme+'_avg_combined'))
        im_file_combined = os.path.join(reg_preproc_dir,'ivim_opt_' + scheme+'_avg_combined.nii.gz')

        # Save TE file
        if '20' in scheme:
            print(f'20 is in {scheme}')
            write_time(im_file_combined.replace('.nii.gz','.te'), np.array([0.0906, 0.0906, 0.0906, 0.168, 0.168, 0.168]))
        elif '60' in scheme:
            write_time(im_file_combined.replace('.nii.gz','.te'), np.array([0.0598, 0.0598, 0.0598, 0.139, 0.139, 0.139]))

        # Fit T2-sIVIM-cov model to registered data
        pars  = fit_ADC_T2(im_file=im_file_combined,
                            bval_file=im_file_combined.replace('.nii.gz','.bval'),
                            TE_file=im_file_combined.replace('.nii.gz','.te'),
                            roi_file=None,
                            outbase=os.path.join(reg_preproc_dir,'est'),
                            Covterm=False)

        # Predict signal at each b-value and TE
        ADC_est = read_im(os.path.join(reg_preproc_dir,'est_ADC.nii.gz'))
        T2app_est = read_im(os.path.join(reg_preproc_dir,'est_T2app.nii.gz'))
        S0_est = read_im(os.path.join(reg_preproc_dir,'est_S0.nii.gz'))

        b = read_bval(im_file_combined.replace('.nii.gz','.bval'))
        TE = read_time(im_file_combined.replace('.nii.gz','.te'))
        for bi in b:
            for TEi in TE:
                write_im(filename=os.path.join(reg_preproc_dir,f'S_pred_b{int(bi)}_TE{int(TEi*1e3)}.nii.gz'),
                        im=S0_est*np.exp(-bi*ADC_est-TEi/T2app_est),
                        imref_file=im_file_combined)

        # Register raw data to predicted data at each b-value and TE
        b = read_bval(in_file_e1.replace('.nii.gz','.bval'))
        for in_file, TEidx in zip([in_file_e1, in_file_e2],[0,-1]):
            registered_vols = []
            for t in range(n_vols):
                in_file_t = extract_volume(in_file, t)
                ref_file_t = os.path.join(reg_preproc_dir,f'S_pred_b{int(b[t])}_TE{int(TE[TEidx]*1e3)}.nii.gz')
                print(f"Registering volume {t+1}/{n_vols} to {ref_file_t}...")
                register_3D_to_ref(in_file=in_file_t, ref_file=ref_file_t, out_file=in_file.replace('.nii.gz',f'-reg{t}.nii.gz'), method='affine')
                registered_vols.append(in_file.replace('.nii.gz',f'-reg{t}.nii.gz'))
            out_file = in_file.replace('.nii.gz',f'-reg4D_{int(i)}.nii.gz')
            # Stack back to 4D
            subprocess.run(['fslmerge','-t',out_file]+registered_vols)
            print(f"Saved registered 4D DWI to {out_file}")
            for vol in registered_vols:
                if vol != ref_file:
                    os.remove(vol)  # Clean up temp registered images
            # Evaluate the performance by looking at the b0-images
            extract(im_file=in_file.replace('.nii.gz',f'-reg4D_{int(i)}.nii.gz'),bval_file=in_file.replace('.nii.gz','.bval'),outbase=in_file.replace('.nii.gz',f'-reg4D_{int(i)}-b0s'),b_ex=0)

        in_file_e1_loop = copy.deepcopy(in_file_e1.replace('.nii.gz',f'-reg4D_{int(i)}.nii.gz'))
        in_file_e2_loop = copy.deepcopy(in_file_e2.replace('.nii.gz',f'-reg4D_{int(i)}.nii.gz'))

    # Evaluate the performance by looking at the b0-images
    extract(im_file=in_file_e1,bval_file=in_file_e1.replace('.nii.gz','.bval'),outbase=in_file_e1.replace('.nii.gz','-b0s'),b_ex=0)
    
    # Copy the final registered files to preproc_dir
    out_file_e1 = os.path.join(preproc_dir,os.path.basename(in_file_e1.replace('.nii.gz',f'-reg4D.nii.gz')))
    out_file_e2 = os.path.join(preproc_dir,os.path.basename(in_file_e2.replace('.nii.gz',f'-reg4D.nii.gz')))
    subprocess.run(['cp',in_file_e1.replace('.nii.gz',f'-reg4D_{int(i)}.nii.gz'),out_file_e1])
    subprocess.run(['cp',in_file_e2.replace('.nii.gz',f'-reg4D_{int(i)}.nii.gz'),out_file_e2])
    subprocess.run(['cp',in_file_e1.replace('.nii.gz','.bval'),out_file_e1.replace('.nii.gz','.bval')])
    subprocess.run(['cp',in_file_e2.replace('.nii.gz','.bval'),out_file_e2.replace('.nii.gz','.bval')])

    return out_file_e1, out_file_e2

def extract_volume(in_file, t_index):
    '''
    Extracts a 3D volume at time index t_index from a 4D image.
    Returns the path to the temporary file.
    '''
    out_file = in_file.replace('.nii.gz',f'-tmp{t_index}.nii.gz')
    subprocess.run(['fslroi',in_file,out_file,str(t_index),'1'])
    return out_file

def register_3D_to_ref(in_file, ref_file, out_file,method='rigid'):
    '''
    Registers a 3D image to a reference image using ITK Elastix.
    '''
    PixelType = itk.F
    Dimension3D = 3
    Image3DType = itk.Image[PixelType, Dimension3D]

    # Built-in default rigid parameter map
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(parameter_object.GetDefaultParameterMap(method))

    ref_itk = itk.imread(ref_file, pixel_type=PixelType)
    moving_itk = itk.imread(in_file, pixel_type=PixelType)

    elastix = itk.ElastixRegistrationMethod[Image3DType, Image3DType].New()
    elastix.SetFixedImage(ref_itk)
    elastix.SetMovingImage(moving_itk)
    elastix.SetParameterObject(parameter_object)
    elastix.Update()
    result = elastix.GetOutput()
    itk.imwrite(result, out_file)
    os.remove(in_file)  # Clean up temp moving image
    return out_file


def fit_ADC_T2(im_file, bval_file, TE_file, roi_file, outbase, Covterm=False):
    '''
    Fit the ADC-T2 model to DWI data using weighted linear least squares (WLLS).
    Model: S = S0 * exp(-b*ADC - TE/T2app)
    '''

    TE = read_time(TE_file)
    Y, b = data_from_file(im_file, bval_file, roi_file = roi_file)
    mask = valid_signal(Y)
    Y = Y[mask,:]

    # Form the design matrix
    X = np.concatenate((np.ones_like(b)[:,np.newaxis],-b[:,np.newaxis],-TE[:,np.newaxis]),axis=1) # [nb,3]
    if Covterm:
        X = np.concatenate((X,-TE[:,np.newaxis]*b[:,np.newaxis]),axis=1) # [nb,3]

    # Solve the linear system
    p_est_lls, *_ = np.linalg.lstsq(X,np.log(Y).T)

    # WLLS: Compute weights from LLS parameters 
    W = np.exp(2*X @ p_est_lls) # [nb, 125000]
    W_sqrt = np.sqrt(W) # [nb, 125000], do this to not have to deal with broadcasting that @ can't do

    # Weight X and y separately with sqrt(W) 
    Xw = X[...,np.newaxis] * W_sqrt[:,np.newaxis,:] # [nb,3]*[nb,1,125000]=[nb,3,125000]
    yw = np.log(Y.T) * W_sqrt # [nb,125000]*[nb,125000]

    # Perform matrix multiplication batch wise
    XTX = np.einsum('ijk,ilk->jlk',Xw,Xw) # [3,nb,125000] @ [nb,3,125000] = [3,3,125000]
    XTy = np.einsum('ijk,ik->jk',Xw,yw) # [3,nb,125000] @ [nb,125000] = [3,125000]

    # Solve linear system
    p_est_wlls = np.linalg.solve(np.transpose(XTX,(2,0,1)), XTy.T[...,np.newaxis]) # [125000,3,3][125000,3,1]= [125000,3,1] --> [125000,3,1], had to be like this because of numpy's solve function
    S0_ls = np.exp(p_est_wlls[:,0,0])
    ADC_ls = p_est_wlls[:,1,0]
    R2app_ls = p_est_wlls[:,2,0]
    if Covterm:
        H_ls = p_est_wlls[:,3,0]

    # Make sure parameters are in reasonable limits
    ADC_ls[ADC_ls > 3e-3] = 3e-3
    ADC_ls[ADC_ls < 0] = 0

    R2app_ls[R2app_ls < 1/1000e-3] = 1/1000e-3
    R2app_ls[R2app_ls > 1/5e-3] = 1/5e-3

    if not Covterm:
        p_est = np.full((mask.size,3), np.nan)
    else:
        p_est = np.full((mask.size,4), np.nan)
    p_est[mask,0] = ADC_ls
    p_est[mask,1] = 1/R2app_ls
    p_est[mask,2] = S0_ls
    if Covterm:
        p_est[mask,3] = H_ls

    pars = {'ADC': p_est[:,0], 'T2app': p_est[:,1], 'S0': p_est[:,2]}
    if Covterm:
        pars['H'] = p_est[:,3]
    save_parmaps(pars, outbase, im_file, roi_file)
    return pars

def topup_wrap(im_file,bval_file, b0rev_file, b0rev_bval_file):
    ''' Susceptibility distortion correction '''
    print('Running topup...')
    im_file_unwarp = im_file.replace('.nii.gz','-unwarp.nii.gz')
    
    # Prepare b0:s for topup
    extract(im_file=im_file, bval_file=bval_file, outbase=im_file.replace('.nii.gz','-b0'),b_ex=0) # we dont perform averaging since it seems preferable to not do it
    # average(im_file=im_file.replace('.nii.gz','-b0.nii.gz'),bval_file=im_file.replace('.nii.gz','-b0.bval'),outbase=im_file.replace('.nii.gz','-b0-mean'))
    if read_bval(b0rev_bval_file).size > 1:
        extract(im_file=b0rev_file, bval_file=b0rev_bval_file, outbase=b0rev_file.replace('.nii.gz','-b0'),b_ex=0) # just use the first b0 since the other became 0.001 and that is bad
    else:
        subprocess.run(['cp',b0rev_file,b0rev_file.replace('.nii.gz','-b0.nii.gz')])
        write_bval(b0rev_file.replace('.nii.gz','-b0.bval'), np.array([0]))
    combine(dwi_files=[im_file.replace('.nii.gz','-b0.nii.gz'),b0rev_file.replace('.nii.gz','-b0.nii.gz')],
            bval_files=[im_file.replace('.nii.gz','-b0.bval'),b0rev_file.replace('.nii.gz','-b0.bval')], 
            outbase=im_file.replace('.nii.gz','-b0-b0rev'))
    b = read_bval(im_file.replace('.nii.gz','-b0.bval'))
    
    temp_folder = os.path.join(os.path.dirname(im_file),'temp')
    if not os.path.isdir(temp_folder): os.mkdir(temp_folder)
    
    acqparams = os.path.join(temp_folder,'acqparams.txt')
    with open(acqparams,'w') as f:
        for _ in range(b.size):
            f.write('0 1 0 0.050\n')
        f.write('0 -1 0 0.050\n')

    # Run topup
    topup = fsl.TOPUP(in_file = im_file.replace('.nii.gz','-b0-b0rev.nii.gz'),
                encoding_file=acqparams,
                out_base = os.path.join(temp_folder,'SDC'),
                out_corrected = os.path.join(temp_folder,'b0_pair_corrected.nii.gz'),
                out_field = os.path.join(temp_folder, 'b0_pair_field.nii.gz'),
                out_logfile = os.path.join(temp_folder, 'b0_pair_log'))
    res = topup.run()
    subprocess.run(['rm', res.outputs.out_jacs[0]])
    subprocess.run(['rm', res.outputs.out_jacs[1]])
    subprocess.run(['rm', res.outputs.out_warps[0]])
    subprocess.run(['rm', res.outputs.out_warps[1]])
    subprocess.run(['rm', res.outputs.out_mats[0]])
    subprocess.run(['rm', res.outputs.out_mats[1]])
    # subprocess.run(['mv', res.outputs.out_enc_file, temp_folder + '/encfile.txt'])
    # Apply the displacements field:
    apptoptup = fsl.ApplyTOPUP(in_files = [im_file],
                        encoding_file = acqparams,
                        in_index = [1],
                        in_topup_fieldcoef = res.outputs.out_fieldcoef,
                        in_topup_movpar = res.outputs.out_movpar,
                        method = 'jac',
                        interp = 'spline', # or 'trilinear'
                        out_corrected = im_file_unwarp)
    apptoptup.run()
    subprocess.run(['cp',bval_file,im_file_unwarp.replace('.nii.gz','.bval')])
    return im_file_unwarp


def preprocess_patient(sub_dir,preproc_dir,sub,ses,scheme):
    '''
    This is a wrapper to all of the above functions to preprocess a single subject's data.

    - sub_dir: str, path to the subject directory containing the raw data
    - preproc_dir: str, path to the directory where preprocessed data will be saved
    - sub: str, subject ID
    - ses: str, session ID
    - scheme: str, acquisition scheme (e.g., 'sIVIM_Gmax61.0')
    '''

    im_file_e1 = os.path.join(sub_dir,sub+'_'+ses+'_ivim_' + scheme+'_e1.nii.gz')
    im_file_e2 = os.path.join(sub_dir,sub+'_'+ses+'_ivim_' + scheme+'_e2.nii.gz')
    b0rev_e1 = os.path.join(sub_dir,sub+'_'+ses+'_b0rev_' + scheme+'_e1.nii.gz')
    b0rev_e2 = os.path.join(sub_dir,sub+'_'+ses+'_b0rev_' + scheme+'_e2.nii.gz')
    
    # Denoising
    if not os.path.isfile(im_file_e1.replace('.nii.gz','-pca.nii.gz')):
        im_file_e1 = denoise_wrap(im_file_e1)
        im_file_e2 = denoise_wrap(im_file_e2)
    else:
        im_file_e1 = im_file_e1.replace('.nii.gz','-pca.nii.gz')
        im_file_e2 = im_file_e2.replace('.nii.gz','-pca.nii.gz')
    
    # Degibbsing
    if not os.path.isfile(im_file_e1.replace('.nii.gz','-gib.nii.gz')):
        im_file_e1 = degibbs_wrap(im_file_e1)
        im_file_e2 = degibbs_wrap(im_file_e2)
    else:
        im_file_e1 = im_file_e1.replace('.nii.gz','-gib.nii.gz')
        im_file_e2 = im_file_e2.replace('.nii.gz','-gib.nii.gz')

    # Extrapolation-registration
    if not os.path.isfile(im_file_e1.replace('.nii.gz','-reg4D.nii.gz')):
        im_file_e1, im_file_e2 = extrapolate_register(im_file_e1, im_file_e2, preproc_dir, scheme)
    else:
        im_file_e1 = im_file_e1.replace('.nii.gz','-reg4D.nii.gz')
        im_file_e2 = im_file_e2.replace('.nii.gz','-reg4D.nii.gz')

    # Susceptibility distortion correction
    if not os.path.isfile(im_file_e1.replace('.nii.gz','-unwarp.nii.gz')):
        im_file_e1 = topup_wrap(im_file_e1, im_file_e1.replace('.nii.gz','.bval'), b0rev_e1, b0rev_e1.replace('.nii.gz','.bval'))
        im_file_e2 = topup_wrap(im_file_e2, im_file_e2.replace('.nii.gz','.bval'), b0rev_e2, b0rev_e2.replace('.nii.gz','.bval'))
    else:
        im_file_e1 = im_file_e1.replace('.nii.gz','-unwarp.nii.gz')
        im_file_e2 = im_file_e2.replace('.nii.gz','-unwarp.nii.gz')
    
    # Brain extraction
    brain_mask_e1 = im_file_e1.replace('.nii.gz','-brain-mask.nii.gz')
    if not os.path.isfile(brain_mask_e1):
        extract(im_file_e1, im_file_e1.replace('.nii.gz','.bval'), outbase=im_file_e1.replace('.nii.gz','-b0'), b_ex=0)
        average(im_file_e1.replace('.nii.gz','-b0.nii.gz'), im_file_e1.replace('.nii.gz','-b0.bval'), im_file_e1.replace('.nii.gz','-b0-avr'))
        subprocess.run(['mri_synthstrip', 
                        '-i', im_file_e1.replace('.nii.gz','-b0-avr.nii.gz'), 
                        '-o', im_file_e1.replace('.nii.gz', '-brain.nii.gz'), 
                        '-m', im_file_e1.replace('.nii.gz', '-brain-mask.nii.gz'),
                        '--border','-3'])
        os.remove(im_file_e1.replace('.nii.gz','-b0.nii.gz'))
        os.remove(im_file_e1.replace('.nii.gz','-b0.bval'))
        os.remove(im_file_e1.replace('.nii.gz','-b0-avr.nii.gz'))
    brain_mask_e2 = im_file_e2.replace('.nii.gz','-brain-mask.nii.gz')
    if not os.path.isfile(brain_mask_e2):
        extract(im_file_e2, im_file_e2.replace('.nii.gz','.bval'), outbase=im_file_e2.replace('.nii.gz','-b0'), b_ex=0)
        average(im_file_e2.replace('.nii.gz','-b0.nii.gz'), im_file_e2.replace('.nii.gz','-b0.bval'), im_file_e2.replace('.nii.gz','-b0-avr'))
        subprocess.run(['mri_synthstrip', 
                        '-i', im_file_e2.replace('.nii.gz','-b0-avr.nii.gz'), 
                        '-o', im_file_e2.replace('.nii.gz', '-brain.nii.gz'), 
                        '-m', im_file_e2.replace('.nii.gz', '-brain-mask.nii.gz'),
                        '--border','-3'])
        os.remove(im_file_e2.replace('.nii.gz','-b0.nii.gz'))
        os.remove(im_file_e2.replace('.nii.gz','-b0.bval'))
        os.remove(im_file_e2.replace('.nii.gz','-b0-avr.nii.gz'))
        os.remove(im_file_e2.replace('.nii.gz','-b0-avr.bval'))

    # Signal drift correction
    if not os.path.isfile(im_file_e1.replace('.nii.gz','-sigdri_corr.nii.gz')):
        signal_drift.spatiotemporal(im_file = im_file_e1, 
                                    bval_file = im_file_e1.replace('.nii.gz','.bval'), 
                                    outbase = im_file_e1.replace('.nii.gz','-sigdri'), 
                                    roi_file = brain_mask_e1)
        subprocess.run(['cp',im_file_e1.replace('.nii.gz','.bval'),im_file_e1.replace('.nii.gz','-sigdri_corr.bval')])
    if not os.path.isfile(im_file_e2.replace('.nii.gz','-sigdri_corr.nii.gz')):
        signal_drift.spatiotemporal(im_file = im_file_e2, 
                                    bval_file = im_file_e2.replace('.nii.gz','.bval'), 
                                    outbase = im_file_e2.replace('.nii.gz','-sigdri'), 
                                    roi_file = brain_mask_e2)
        subprocess.run(['cp',im_file_e2.replace('.nii.gz','.bval'),im_file_e2.replace('.nii.gz','-sigdri_corr.bval')])    
    im_file_e1 = im_file_e1.replace('.nii.gz','-sigdri_corr.nii.gz')    
    im_file_e2 = im_file_e2.replace('.nii.gz','-sigdri_corr.nii.gz')
    
    # Directional averaging
    average(im_file_e1,im_file_e1.replace('.nii.gz','.bval'),im_file_e1.replace('.nii.gz','-diravr'),avg_type='geo')
    average(im_file_e2,im_file_e2.replace('.nii.gz','.bval'),im_file_e2.replace('.nii.gz','-diravr'),avg_type='geo')
    im_file_e1 = im_file_e1.replace('.nii.gz','-diravr.nii.gz')
    im_file_e2 = im_file_e2.replace('.nii.gz','-diravr.nii.gz')
    im_comb = im_file_e1.replace('_e1','').replace('.nii.gz','-comb.nii.gz')
    combine(dwi_files=[im_file_e1, im_file_e2], bval_files=[im_file_e1.replace('.nii.gz','.bval'), im_file_e2.replace('.nii.gz','.bval')], outbase=im_comb.replace('.nii.gz',''))

    print(f'Preprocessing done for {sub} {scheme}')