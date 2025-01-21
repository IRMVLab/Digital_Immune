import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import segment_from_cfa, bounding_box
from scipy.ndimage.morphology import binary_dilation

def eval_mir(dataroot, nii_path):
    b0_size = 10
    data, affine = load_nifti(dataroot)
    raw_max = np.max(data, axis=(0,1,2), keepdims=True)
    bvals, bvecs = read_bvals_bvecs(dataroot.replace('.nii.gz', '.bval'), dataroot.replace('.nii.gz', '.bvec'))
    sel_b = bvals != 0
    gtab = gradient_table(bvals[sel_b], bvecs[sel_b])
    raw_gtab = gradient_table(bvals, bvecs)
    denoised_data, _ = load_nifti(nii_path)
    denoised_data -= np.min(denoised_data, axis=(0,1), keepdims=True)
    denoised_data /= np.max(denoised_data, axis=(0,1), keepdims=True)
    denoised_data = np.concatenate((data[:,:,:,:b0_size], denoised_data), axis=-1)
    b0_mask, mask = median_otsu(data, vol_idx=[0])
    tenmodel = TensorModel(raw_gtab)
    tensorfit = tenmodel.fit(data, mask=mask)
    threshold = (0.6, 1, 0, 0.1, 0, 0.1)
    CC_box = np.zeros_like(data[..., 0])
    mins, maxs = bounding_box(mask)
    mins = np.array(mins)
    maxs = np.array(maxs)
    diff = (maxs - mins) // 4
    bounds_min = mins + diff
    bounds_max = maxs - diff
    CC_box[bounds_min[0]:bounds_max[0], bounds_min[1]:bounds_max[1], bounds_min[2]:bounds_max[2]] = 1
    mask_cc_part, cfa = segment_from_cfa(tensorfit, CC_box, threshold, return_cfa=True)
    mask_noise = binary_dilation(mask, iterations=10)
    mask_noise = ~mask_noise
    data_normalized = data - np.min(data, axis=(0,1), keepdims=True)
    data_normalized = (data_normalized.astype(np.float32) / np.max(data_normalized, axis=(0,1), keepdims=True))
    mean_signal = np.mean(data_normalized[mask_cc_part], axis=0)
    noise_std = np.std(data_normalized[mask_noise, :], axis=0)
    mean_bg = np.mean(data_normalized[mask_noise, :], axis=0)
    mean_signal_denoised = np.mean(denoised_data[mask_cc_part], axis=0)
    denoised_noise_std = np.std(denoised_data[mask_noise, :], axis=0)
    denoised_mean_bg = np.mean(denoised_data[mask_noise, :], axis=0)
    SNRs = {}
    CNRs = {}
    SNR = mean_signal / noise_std
    CNR = (mean_signal - mean_bg) / noise_std
    SNR = SNR[sel_b]
    CNR = CNR[sel_b]
    SNRs['raw'] = SNR
    CNRs['raw'] = CNR
    print('raw', '[SNR] mean: %.4f std: %.4f' % (np.mean(SNR), np.std(SNR)))
    print('raw', '[CNR] mean: %.4f std: %.4f' % (np.mean(CNR), np.std(CNR)))
    SNR = mean_signal_denoised / (denoised_noise_std+1e-7)
    CNR = (mean_signal_denoised - denoised_mean_bg) / (denoised_noise_std+1e-7)
    SNR = SNR[sel_b]
    CNR = CNR[sel_b]
    SNR -= SNRs['raw']
    CNR -= CNRs['raw']
    SNRs['our'] = SNR
    CNRs['our'] = CNR
    print("=================")
    print('our', '[SNR delta] mean: %.4f std: %.4f best: %.4f worst: %.4f' % (np.mean(SNR), np.std(SNR), np.max(SNR), np.min(SNR)))
    print('our', '[CNR delta] mean: %.4f std: %.4f best: %.4f worst: %.4f' % (np.mean(CNR), np.std(CNR), np.max(CNR), np.min(CNR)))
    return SNRs, CNRs