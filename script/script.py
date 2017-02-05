import numpy as np
import csv

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
                            data_slice.shape[3] - 1])

    # Normalized mean sq error
    NMSE = []
    # Note that we are not predicting S0, because we always need S0 to fit the model

    for left_out in range(1, data_slice.shape[-1]-1):
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
        predictions[..., left_out] = left_predictions[..., 0]
        err = nmse(left_predictions.squeeze().ravel(), data_slice[..., left_out].squeeze().ravel())
        NMSE += [err]

    return (predictions, np.array(NMSE))

img, gtab = read_ivim()
data = img.get_data()

# max values for each dim
x, y, z, bval = data.shape

# size of volume chunk for each iteration
vol_size = 2

# Max iterations to run
max_iters = 3
count = 0

# volume id of last data point
volids = []

with open("outputs/ivim.csv", "r") as f: 
    reader = csv.reader(f)
    for row in reader:
        volids += [row[0:3]]

# Initial volume ids
x1, y1, z1 = [int(x) for x in volids[-1]]

while count < max_iters:
    x2, y2, z2 = x1 + vol_size, y1 + vol_size, z1+ vol_size

    data_slice = data[x1:x2, y1:y2, z1:z2, :]
    ivim_predictions, ivim_nmse = leave_one_cross(IvimModel, data_slice, gtab)

    # output row. Add the volume index
    out = [x1, y1, z1]
    out.append(ivim_nmse.mean())
    
    # We can chose to record predicted val for each bval
    # nmse_flat = ivim_nmse.ravel()
    # for x in nmse_flat:
    #     out.append(x)

    with open('outputs/ivim'+'.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(out)
    
    # Exponential decay model
    exp_predictions, exp_nmse = leave_one_cross(ExponentialModel, data_slice, gtab)
    out = [x1, y1, z1]
    out.append(exp_nmse.mean())

    with open('outputs/exp'+'.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(out)
    count += 1
    
    x1 += vol_size
    y1 += vol_size
    z1 += vol_size
