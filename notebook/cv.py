# Script for IVIM cross val

import numpy as np
from dipy.reconst.ivim import IvimModel
from dipy.data.fetcher import read_ivim
import dipy.core.gradients as dpg

from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit

def exponential_prediction(params, gtab, S0 = 1.):
    """
    Prediction using exponential decay model.
    """
    b = gtab.bvals
    return params[0] * np.exp(- b * params[1])


class ExponentialModel(ReconstModel):
    """
    An exponential decay model.
    """
    def __init__(self, gtab=None):
        self.gtab = gtab
        
    @multi_voxel_fit
    def fit(self, data, mask=None):
        """ Fit method
        Parameters
        ----------
        data : array
            The measured signal from one voxel. A multi voxel decorator
            will be applied to this fit method to scale it and apply it
            to multiple voxels.
        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

        Returns
        -------
        ExponentialFit object
        """
        # Get S0_prime and D - paramters assuming a single exponential decay
        D, neg_log_S0 = np.polyfit(self.gtab.bvals, -np.log(data), 1)
        S0 = np.exp(-neg_log_S0)
        params_linear = np.array([S0, D])
        
        return ExponentialFit(self, params_linear)

    def predict(self, params, gtab, S0=1.):
        """
        Predict the values of the signal.
        """
        return exponential_prediction(params, gtab)

class ExponentialFit(object):

    def __init__(self, model, model_params):
        """ Initialize a LinearFit class instance.
        """
        self.model = model
        self.model_params = model_params

    def __getitem__(self, index):
        model_params = self.model_params
        N = model_params.ndim
        if type(index) is not tuple:
            index = (index,)
        elif len(index) >= model_params.ndim:
            raise IndexError("IndexError: invalid index")
        index = index + (slice(None),) * (N - len(index))
        return type(self)(self.model, model_params[index])

    @property
    def S0_predicted(self):
        return self.model_params[..., 0]

    @property
    def D(self):
        return self.model_params[..., 1]

    def predict(self, gtab, S0=1.):
        """Given a model fit, predict the signal.

        Parameters
        ----------
        gtab : GradientTable class instance
               Gradient directions and bvalues

        S0 : float
            S0 value here is not necessary and will not be used to predict the
            signal. It has been added to conform to the structure of the
            predict method in multi_voxel which requires a keyword argument S0.

        Returns
        -------
        signal : array
            The signal values predicted for this model using its parameters.
        """
        return exponential_prediction(self.model_params, gtab)
