from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
# to avoid the warning o generating a RuntimeWarning
np.seterr(divide='ignore', invalid='ignore')


def normalize_difference_indexes_minus_plus(band_1, band_2):
    """ Uses the general formula of all normalize difference indexes
    where the numerator is a substraction and the denominator is and
    adition

    Parameters
    __________
    band_1: numpy array
    band_2: numpy array
    """
    numerator = np.subtract(band_2, band_1, dtype=np.float64)
    denominator = np.add(band_1, band_2, dtype=np.float64)
    index = np.where(band_2 + band_1 == 0.,
                     0.,
                     numerator / denominator)
    return index


class CommonIndex(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, x):
        # TODO x must be a dataframe
        return x




