import numpy as np
from stats_arrays import *
from scipy import stats


class FourParamBetaUncertainty(UncertaintyBase):

    """
4 PARAM BETA DISTRIBUTIONS
The 4 parameters Beta distribution has the probability distribution function:

The :math:`\\alpha` parameter is ``loc``, and :math:`\\beta` is ``shape``. 
By default, the Beta distribution is defined from 0 to 1; the upper bound can be rescaled with the ``maximum`` parameter
and the lower bound with the ``minimum`` parameter.

Wikipedia: `Beta distribution <http://en.wikipedia.org/wiki/Beta_distribution>`_
    """
    id = 14
    description = "Four Parameters Beta uncertainty"

    @classmethod
    def validate(cls, params):
        scale_param=params['maximum']-params['minimum']
        if (params['loc'] > 0).sum() != params.shape[0]:
            raise InvalidParamsError("Real, positive alpha values are" +
                                     " required for Beta uncertainties.")
        if (params['shape'] > 0).sum() != params.shape[0]:
            raise InvalidParamsError("Real, positive beta values are" +
                                     " required for Beta uncertainties.")
        if (scale_param <= 0).sum():
            raise InvalidParamsError("Scale value must be positive or NaN")

    @classmethod
    def random_variables(cls, params, size, seeded_random=None,
                         transform=False):
        scale_param = params['maximum'] - params['minimum']
        if not seeded_random:
            seeded_random = np.random
        scale = scale_param
        scale[np.isnan(scale)] = 1
        return (params['minimum'].reshape((-1, 1)) + scale.reshape((-1, 1)) * 
                seeded_random.beta(params['loc'], params['shape'], size=(size,params.shape[0])).T)

    @classmethod
    def cdf(cls, params, vector):
        vector = cls.check_2d_inputs(params, vector)
        results = zeros(vector.shape)
        scale_param=params['maximum']-params['minimum']
        scale = scale_param
        scale[np.isnan(scale)] = 1
        for row in range(params.shape[0]):
            results[row, :] = stats.beta.cdf(vector[row, :],
                                             params['loc'][row], params['shape'][row],
                                             loc=params['minimum'][row],
                                             scale=scale[row])
        return results

    @classmethod
    def ppf(cls, params, percentages):
        percentages = cls.check_2d_inputs(params, percentages)
        results = zeros(percentages.shape)
        scale_param=params['maximum']-params['minimum']
        scale = scale_param
        scale[np.isnan(scale)] = 1
        for row in range(percentages.shape[0]):
            results[row, :] = stats.beta.ppf(percentages[row, :],
                                             params['loc'][row], params['shape'][row],
                                             loc=params['minimum'][row],
                                             scale=scale[row])
        return results

    @classmethod
    def statistics(cls, params):
        alpha = float(params['loc'])
        beta = float(params['shape'])
        mini = float(params['minimum'])
        maxi = float(params['maximum'])
        # scale = 1 if isnan(params['maximum'])[0] else float(params['maximum'])
        if alpha <= 1 or beta <= 1:
            mode = "Undefined"
        else:
            mode = mini + (maxi-mini) * (alpha - 1) / (alpha + beta - 2)
        return {
            'mean': mini + (maxi-mini) * alpha / (alpha + beta),
            'mode': mode,
            'median': "Not Implemented",
            'lower': mini,
            'upper': maxi
        }

    @classmethod
    def pdf(cls, params, xs=None):
        scale_param=params['maximum']-params['minimum']
        scale = 1 if np.isnan(scale_param)[0] else float(scale_param)
        if xs is None:
            xs = arange(0, scale, scale / cls.default_number_points_in_pdf)
        ys = stats.beta.pdf(xs, params['loc'], params['shape'],
                            loc=params['minimum'],
                            scale=scale)
        return xs, ys.reshape(ys.shape[1])
    
uncertainty_choices.add(FourParamBetaUncertainty)
print (FourParamBetaUncertainty in uncertainty_choices)