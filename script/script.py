import numpy as np
import nibabel as nib

from dipy.reconst.ivim import IvimModel
from dipy.data.fetcher import read_ivim
import dipy.core.gradients as dpg
from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit

from exponential import ExponentialModel

def nmse(data, prediction):
    """
    Compute the normalized mean squared error.

    Parameters
    ----------
    data: array
        Actual values

    prediction: array
        Predicted values

    Returns
    -------
    mean : array
        The normalised mean squared error for the given data points
    """
    return (((data - prediction)**2).mean())/sum(data**2)

def leave_one_cross(model, data_slice, gtab):
    """
    Performs a leave one out cross validation on the data points specified by
    `data_slice` and returns the prediction and normalised mean sq error for these
    voxels.

    Parameters
    ----------
    model: ReconstModel
        A ReconstModel class instance
    data_slice: array
        The voxels specifying the volume to perform cross validation.

    Returns
    -------
    predictions: array
        Predicted values of given volume at all bvalues.
    """
    # Preallocate an array for the results:
    predictions = np.zeros([data_slice.shape[0], 
                            data_slice.shape[1],
                            data_slice.shape[2],
                            data_slice.shape[3]])

    # Note that we are not predicting S0, because we always need S0 to fit the model
    predictions[..., 0] = data_slice[..., 0]

    for left_out in range(1, data_slice.shape[-1]):
        # These are the b-values/b-vectors with one of them left out:
        left_out_bvals = np.concatenate([gtab.bvals[:left_out], gtab.bvals[left_out+1:]])
        left_out_bvecs = np.concatenate([gtab.bvecs[:left_out], gtab.bvecs[left_out+1:]])
        left_out_gtab = dpg.gradient_table(left_out_bvals, left_out_bvecs)
        # Create a model for this iteration
        current_model = model(left_out_gtab)
        # We fit to the data leaving out the current measurement
        left_out_data = np.concatenate([data_slice[..., :left_out], 
                                        data_slice[..., left_out+1:]], -1)
        fit = current_model.fit(left_out_data)
        # We try to predict only the left out measurement
        predict_gtab = dpg.gradient_table(np.array([gtab.bvals[left_out]]), 
                                          np.array([gtab.bvecs[left_out]]))
        left_predictions = fit.predict(predict_gtab)        
        predictions[..., left_out] = left_predictions[..., -1]
    return (predictions)

img, gtab = read_ivim()
data = img.get_data()

x1, y1, z1 = 90, 90, 30
x2, y2, z2 = 95, 95, 35

predicted_ivim = np.zeros_like(data)
predicted_ivim[x1:x2, y1:y2, z1:z2, ...] = leave_one_cross(IvimModel, data[x1:x2, y1:y2, z1:z2, :], gtab)

predicted_exp = np.zeros_like(data)
predicted_exp[x1:x2, y1:y2, z1:z2, ...] = leave_one_cross(ExponentialModel, data[x1:x2, y1:y2, z1:z2, :], gtab)

exp_img = nib.Nifti1Image(predicted_exp, np.eye(4))
ivim_img = nib.Nifti1Image(predicted_ivim, np.eye(4))

nib.save(exp_img, "outputs/exponential.nii.gz")
nib.save(ivim_img, "outputs/ivim.nii.gz")