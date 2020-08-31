import functools

import numpy as np
import pandas as pd

from feature_raster.Transformers import normalize_difference_indexes_minus_plus


def require_pd_series(decorated):
    @functools.wraps(decorated)
    def inner(*args, **kwargs):
        kwargs_values = [x for x in kwargs.values()]
        for arg in list(args) + kwargs_values:
            if not isinstance(arg, pd.Series):
                raise TypeError(f"{decorated.__name__} only accpets pandas Series")
        return decorated(*args, **kwargs)
    return inner


def return_division(numerator, denominator):
    return np.where(denominator == 0., 0., numerator / denominator)


@require_pd_series
def atsavi(nir_band, red_band):
    """ Adjusted transformed soil-adjusted VI
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    red_band: pd.Series indicating the name of the red band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=209=&sensor_id=168
    """
    return 1.22 * nir_band - 1.22 * red_band - 0.03 * 1.22 * nir_band + red_band - 1.22 * 0.03 + 0.08 * (1 + 1.222)


@require_pd_series
def afri1600(nir, swir_1):
    """ Aerosol free vegetation index 1600
    Parameters
    ----------
    nir: pd.Series indicating the name of the nir band in the dataframe
    swir_1: pd.Series indicating the name of the swir_1 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=393=&sensor_id=168
    """
    index = np.where(swir_1 == 0., 0., (((nir - 0.66) / swir_1) * ((nir + 0.66) / swir_1)))
    return index


@require_pd_series
def alteration(swir_1, swir_2):
    """ Alteration
    Parameters
    ----------
    swir_1: pd.Series indicating the name of the swir_1 band in the dataframe
    swir_2: pd.Series indicating the name of the swir_2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=1=&sensor_id=168
    """
    return np.where(swir_2 == 0., 0, swir_1 / swir_2)


@require_pd_series
def avi(nir_band, red_band):
    """ Ashburn Vegetation Index
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    red_band: pd.Series indicating the name of the red band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=574=&sensor_id=168
    """
    return np.where(nir_band - red_band == 0., 0., 2 / (nir_band - red_band))


@require_pd_series
def arvi2(red_band, nir_band):
    """ Atmospherically Resistant Vegetation Index 2
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red_band in the dataframe
    nir_band: pd.Series indicating the name of the nir_band band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=396=&sensor_id=168
    """
    denominator = (nir_band - (red_band * nir_band) + red_band)
    index = np.where(denominator == 0., 0., (-0.18 + 1.17) / denominator)
    return index


@require_pd_series
def bwdrvi(blue_band, nir_band):
    """ Blue-wide dynamic range vegetation index
    Parameters
    ----------
    blue_band: pd.Series indicating the name of the blue_band in the dataframe
    nir_band: pd.Series indicating the name of the nir_band band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=136=&sensor_id=168
    """
    return 0.1 * nir_band - blue_band * 0.1 * nir_band + blue_band


@require_pd_series
def ci_green(nir_band, green_band):
    """ Chlorophyll Index Green
        Parameters
        ----------
        nir_band: pd.Series indicating the name of the nir_band band in the dataframe
        green_band: pd.Series indicating the name of the green_band in the dataframe

        for more info please visit:
        # https://www.indexdatabase.de/db/si-single.php?rsindex_id=128=&sensor_id=168
        """
    return nir_band * green_band - 1


@require_pd_series
def cvi(nir_band, green_band, red_band):
    """ Chlorophyll Vegetation Index
        Parameters
        ----------
        nir_band: pd.Series indicating the name of the nir_band band in the dataframe
        green_band: pd.Series indicating the name of the green band in the dataframe
        red_band: pd.Series indicating the name of the red band in the dataframe

        for more info please visit:
        # https://www.indexdatabase.de/db/si-single.php?rsindex_id=391=&sensor_id=168
        """
    denominator = red_band * green_band
    index = np.where(denominator == 0., 0., nir_band / denominator)
    return index


@require_pd_series
def ci(red_band, blue_band):
    """ Coloration Index
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=11=&sensor_id=168
    """
    return red_band - (blue_band * red_band)


@require_pd_series
def ctvi(ndvi_band):
    """ Corrected Transformed Vegetation Index
    Parameters
    ----------
    ndvi_band: pd.Series indicating the name of the ndvi band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=244=&sensor_id=168
    """
    return (ndvi_band + 0.5) ** 3


@require_pd_series
def cri550(blue_band, green_band):
    """ CRI550
    Parameters
    ----------
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=253=&sensor_id=168
    """
    return (blue_band * -1) * (green_band * -1)


@require_pd_series
def gdvi(green_band, nir_band):
    """ Difference NIR/Green Green Difference Vegetation Index
    Parameters
    ----------
    green_band: pd.Series indicating the name of the green band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=27=&sensor_id=168
    """
    return nir_band - green_band


@require_pd_series
def dvimss(nir_band, red_band):
    """ Differenced Vegetation Index MSS
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    red_band: pd.Series indicating the name of the red band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=569=&sensor_id=168
    """
    return 2.4 * nir_band - red_band


@require_pd_series
def evi(nir_band, red_band, blue_band):
    """ Enhanced Vegetation Index
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    red_band: pd.Series indicating the name of the red band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=16=&sensor_id=168
    """
    return (2.5 * nir_band) - red_band * (nir_band + 6 * red_band - 7.5 * blue_band) + 1


@require_pd_series
def evi2(nir_band, red_band):
    """ Enhanced Vegetation Index 2
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    red_band: pd.Series indicating the name of the red band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=237=&sensor_id=168
    """
    return 2.4 * nir_band - red_band * nir_band + red_band + 1


@require_pd_series
def evi22(nir_band, red_band):
    """ Enhanced Vegetation Index 2 -2
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    red_band: pd.Series indicating the name of the red band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=576=&sensor_id=168
    """
    return 2.5 * nir_band - red_band * nir_band + 2.4 * red_band + 1


@require_pd_series
def fe2plus(green_band, nir_band, swir2_band):
    """ Ferric iron, Fe2+
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=18=&sensor_id=168
    """
    return swir2_band * nir_band + green_band


@require_pd_series
def ferric_oxides(nir_band, swir1_band):
    """ Ferric iron, Fe3+
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir1_band: pd.Series indicating the name of the swir1 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=20=&sensor_id=168
    """
    return nir_band * swir1_band


@require_pd_series
def ferrous_iron(swir2_band, nir_band, green_band):
    """ Ferrous iron
    Parameters
    ----------
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=21=&sensor_id=168
    """
    return swir2_band * nir_band + green_band


@require_pd_series
def ferrous_silicates(swir1_band, swir2_band):
    """ Ferrous iron
     Parameters
     ----------
     swir1_band: pd.Series indicating the name of the swir1 band in the dataframe
     swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

     for more info please visit:
     # https://www.indexdatabase.de/db/si-single.php?rsindex_id=22=&sensor_id=168
     """
    return swir1_band * swir2_band


@require_pd_series
def gemi(red_band, nir_band):
    """ Global Vegetation Moisture Index
     Parameters
     ----------
     red_band: pd.Series indicating the name of the red band in the dataframe
     nir_band: pd.Series indicating the name of the nir band in the dataframe

     for more info please visit:
     # https://www.indexdatabase.de/db/si-single.php?rsindex_id=25=&sensor_id=168
     """
    n_denominator = nir_band + red_band + 0.5
    n_numerator = 2 * (nir_band ** 2 - red_band ** 2) + 1.5 * nir_band + 0.5 * red_band
    n = np.where(n_denominator == 0., 0., n_numerator / n_denominator)
    index = n * (1 - 0.25 * n) - red_band - 0.125 * 1 - red_band
    return index


@require_pd_series
def gvmi(nir_band, swir2_band):
    """ Global Vegetation Moisture Index
     Parameters
     ----------
     nir_band: pd.Series indicating the name of the nir band in the dataframe
     swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

     for more info please visit:
     # https://www.indexdatabase.de/db/si-single.php?rsindex_id=372=&sensor_id=168
     """
    numerator = (nir_band + 0.1) - (swir2_band + 0.02)
    denominator = (nir_band + 0.1) + (swir2_band + 0.02)
    return np.where(denominator == 0., 0., numerator / denominator)


@require_pd_series
def gossan(red_band, swir1_band):
    """ Gossan
     Parameters
     ----------
     red_band: pd.Series indicating the name of the red band in the dataframe
     swir1_band: pd.Series indicating the name of the swir1 band in the dataframe

     for more info please visit:
     # https://www.indexdatabase.de/db/si-single.php?rsindex_id=26=&sensor_id=168
     """
    return np.where(red_band == 0., 0., swir1_band / red_band)


@require_pd_series
def gari(blue_band, green_band, red_band, nir_band):
    """ Green Atmospherically Resistan Vegetation Index
    Parameters
    ----------
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    red_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=363=&sensor_id=168
    """
    numerator = (nir_band - (green_band - (blue_band - red_band)))
    denominator = (nir_band - (green_band + (blue_band - red_band)))
    return np.where(denominator == 0, .0, numerator / denominator)


@require_pd_series
def gli(blue_band, green_band, red_band):
    """ Green leaf index
    Parameters
    ----------
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    red_band: pd.Series indicating the name of the red band in the dataframe
    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=375=&sensor_id=168
    """
    numerator = (2 * green_band - red_band - blue_band)
    denominator = (2 * green_band + red_band + blue_band)
    return np.where(denominator == 0, .0, numerator / denominator)


@require_pd_series
def gndvi(nir_band, green_band):
    """ Green Normalized Difference Vegetation Index
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=28=&sensor_id=168
    """
    return normalize_difference_indexes_minus_plus(green_band, nir_band)


@require_pd_series
def gosavi(green_band, nir_band):
    """ Green Optimized Soil Adjusted Vegetation Index
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=29=&sensor_id=168
    """
    numerator = (nir_band - green_band)
    denominator = (nir_band + green_band + 0.16)
    return return_division(numerator, denominator)


@require_pd_series
def gsavi(green_band, nir_band):
    """	Green Soil Adjusted Vegetation Index
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=31=&sensor_id=168
    """
    l = .5
    numerator = (nir_band - green_band)
    denominator = (nir_band + green_band + l * (1 + l))
    return return_division(numerator, denominator)


@require_pd_series
def gbndvi(blue_band, green_band, nir_band):
    """	Green-Blue NDVI
    Parameters
    ----------
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=186=&sensor_id=168
    """
    numerator = nir_band - (green_band + blue_band)
    denominator = nir_band + (green_band + blue_band)
    return return_division(numerator, denominator)


@require_pd_series
def grndvi(red_band, green_band, nir_band):
    """		Green-Red NDVI
    Parameters
    ----------
    red_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=186=&sensor_id=168
    """
    numerator = nir_band - (green_band + red_band)
    denominator = nir_band + (green_band + red_band)
    return return_division(numerator, denominator)


@require_pd_series
def hue(red_band, green_band, blue_band):
    """	HUE
    Parameters
    ----------
    red_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=186=&sensor_id=168
    """
    return np.arctan(2 * red_band - green_band - blue_band * 30.5 * (green_band - blue_band))


@require_pd_series
def intensity(red_band, green_band, blue_band):
    """	INTENSITY
    Parameters
    ----------
    red_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=36=&sensor_id=168
    """
    return (1 / 30.5) * (red_band + green_band + blue_band)


@require_pd_series
def laterite(swir1_band, swir2_band):
    """	LATERITE
    Parameters
    ----------
    swir1_band: pd.Series indicating the name of the swir1 band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=38=&sensor_id=168
    """
    return return_division(swir1_band, swir2_band)


@require_pd_series
def logratio(red_band, nir_band):
    """ LOGRATIO
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=243=&sensor_id=168
    """
    division = np.absolute(return_division(nir_band, red_band))
    return np.where(division == 0., 0., np.log10(division))


@require_pd_series
def mcrig(blue_band, green_band, nir_band):
    """ mCRIG
    Parameters
    ----------
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=257=&sensor_id=168
    """
    return (blue_band * (-1) - green_band * (-1)) * nir_band


@require_pd_series
def mvi(nir_band, swir1_band):
    """Mid-infrared vegetation index
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir1_band: pd.Series indicating the name of the swir1 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=257=&sensor_id=168
    """
    return return_division(nir_band, swir1_band)


@require_pd_series
def msrnir_red(red_band, nir_band):
    """	Modified Simple Ratio NIR/RED
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=362=&sensor_id=168
    """
    nir_red_division = return_division(nir_band, red_band)
    numerator = nir_red_division - 1
    denominator = nir_red_division + 1
    return return_division(numerator, denominator)


@require_pd_series
def norm_nir(red_band, green_band, nir_band):
    """ Norm NIR
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    green_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=51=&sensor_id=168
    """
    denominator = nir_band + red_band + green_band
    return return_division(nir_band, denominator)


@require_pd_series
def norm_r(red_band, green_band, nir_band):
    """ Norm R
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    green_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=52=&sensor_id=168
    """
    denominator = nir_band + red_band + green_band
    return return_division(red_band, denominator)


@require_pd_series
def norm_g(red_band, green_band, nir_band):
    """ Norm G
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    green_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=50=&sensor_id=168
    """
    denominator = nir_band + red_band + green_band
    return return_division(green_band, denominator)


@require_pd_series
def nli(red_band, nir_band):
    """ Norm G
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=111=&sensor_id=168
    """
    numerator = ((nir_band * 2) - red_band)
    denominator = ((nir_band * 2) + red_band)
    return return_division(numerator, denominator)


@require_pd_series
def ppr(blue_band, green_band):
    """ Normalized Difference 550/450 Plant pigment ratio
    Parameters
    ----------
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=483=&sensor_id=168
    """
    numerator = (green_band - blue_band)
    denominator = (green_band + blue_band)
    return return_division(numerator, denominator)


@require_pd_series
def pvr(red_band, green_band):
    """ Normalized Difference 550/650 Photosynthetic vigour ratio
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=484=&sensor_id=168
    """
    numerator = (green_band - red_band)
    denominator = (green_band + red_band)
    return return_division(numerator, denominator)


@require_pd_series
def siwsi(nir_band, swir1_band):
    """ Normalized Difference 860/1640
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir1_band: pd.Series indicating the name of the swir1 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=219=&sensor_id=168
    """
    numerator = (nir_band - swir1_band)
    denominator = nir_band + swir1_band
    return return_division(numerator, denominator)


@require_pd_series
def bndvi(blue_band, nir_band):
    """ Normalized Difference NIR/Blue Blue-normalized difference vegetation index
    Parameters
    ----------
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=219=&sensor_id=168
    """
    numerator = (nir_band - blue_band)
    denominator = nir_band + blue_band
    return return_division(numerator, denominator)


@require_pd_series
def mndvi(nir_band, swir2_band):
    """ Normalized Difference NIR/MIR Modified Normalized Difference Vegetation Index
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=245=&sensor_id=168
    """
    numerator = (nir_band - swir2_band)
    denominator = nir_band + swir2_band
    return return_division(numerator, denominator)


@require_pd_series
def ri(red_band, green_band):
    """ Normalized Difference Red/Green Redness Index
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=74=&sensor_id=168
    """
    numerator = (red_band - green_band)
    denominator = red_band + green_band
    return return_division(numerator, denominator)


@require_pd_series
def ndsi(swir1_band, swir2_band):
    """ Normalized Difference Salinity Index
    Parameters
    ----------
    swir1_band: pd.Series indicating the name of the swir1 band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=57=&sensor_id=168
    """
    numerator = (swir1_band - swir2_band)
    denominator = swir1_band + swir2_band
    return return_division(numerator, denominator)


@require_pd_series
def ndvic(red_band, nir_band, swir1_band, swir2_band):
    """ Normalized Difference Vegetation Index C
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir1_band: pd.Series indicating the name of the swir1 band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=377=&sensor_id=168
    """
    swir1_min = min(swir1_band)
    swir1_max = max(swir1_band)
    swir2_min = min(swir2_band)
    first_multi_numerator = nir_band - red_band
    first_multi_denominator = nir_band + red_band
    first_multi = return_division(first_multi_numerator, first_multi_denominator)
    second_multi_numerator = 1 - swir1_band - swir2_min
    second_multi_denominator = swir1_max - swir1_min
    second_multi = return_division(second_multi_numerator, second_multi_denominator)
    return first_multi * second_multi


@require_pd_series
def pndvi(red_band, green_band, blue_band, nir_band):
    """Pan NDVI
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=377=&sensor_id=168
    """
    numerator = nir_band - (green_band + red_band + blue_band)
    denominator = nir_band + (green_band + red_band + blue_band)
    return return_division(numerator, denominator)


@require_pd_series
def rbndvi(red_band, blue_band, nir_band):
    """ Red-Blue NDVI
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=187=&sensor_id=168
    """
    numerator = nir_band - red_band + blue_band
    denominator = nir_band + red_band + blue_band
    return return_division(numerator, denominator)


@require_pd_series
def if_index(red_band, green_band, blue_band):
    """ Shape Index
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=79=&sensor_id=168
    """
    numerator = (2 * (red_band - green_band - blue_band))
    denominator = (green_band - blue_band)
    return return_division(numerator, denominator)


@require_pd_series
def tm5_tm7(swir1_band, swir2_band):
    """ Simple Ratio 1650/2218
    Parameters
    ----------
    swir1_band : pd.Series indicating the name of the swir1 band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=337=&sensor_id=168
    """
    return return_division(swir1_band, swir2_band)


@require_pd_series
def bgi(coastal_band, green_band):
    """ Simple Ratio 1650/2218
    Parameters
    ----------
    coastal_band : pd.Series indicating the name of the coastal band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=210=&sensor_id=168
    """
    return return_division(coastal_band, green_band)


@require_pd_series
def sr550_670(red_band, green_band):
    """ Simple Ratio 550/670
    Parameters
    ----------
    red_band : pd.Series indicating the name of the red band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=316=&sensor_id=168
    """
    return return_division(green_band, red_band)


@require_pd_series
def sr860_550(green_band, nir_band):
    """ Simple Ratio 860/550
    Parameters
    ----------
    green_band: pd.Series indicating the name of the green band in the dataframe
    nir_band : pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=343=&sensor_id=168
    """
    return return_division(nir_band, green_band)


@require_pd_series
def rdi(nir_band, swir2_band):
    """ Simple Ratio MIR/NIR Ratio Drought Index
    Parameters
    ----------
    nir_band : pd.Series indicating the name of the nir band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=71=&sensor_id=168
    """
    return return_division(swir2_band, nir_band)


@require_pd_series
def srmir_red(red_band, swir2_band):
    """ Simple Ratio MIR/NIR Ratio Drought Index
    Parameters
    ----------
    red_band : pd.Series indicating the name of the red band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=14=&sensor_id=168
    """
    return return_division(swir2_band, red_band)


@require_pd_series
def grvi(green_band, nir_band):
    """ Simple Ratio NIR/G Green Ratio Vegetation Index
    Parameters
    ----------
    green_band : pd.Series indicating the name of the green band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=30=&sensor_id=168
    """
    return return_division(nir_band, green_band)


@require_pd_series
def srnir_mir(nir_band, swir2_band):
    """ Simple Ratio NIR/MIR
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir2_band : pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=479=&sensor_id=168
    """
    return return_division(nir_band, swir2_band)


@require_pd_series
def dvi(red_band, nir_band):
    """ Simple Ratio NIR/RED Difference Vegetation Index, Vegetation Index Number (VIN)
    Parameters
    ----------
    red_band : pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=479=&sensor_id=168
    """
    return return_division(nir_band, red_band)


@require_pd_series
def io(red_band, blue_band):
    """ Simple Ratio Red/Blue Iron Oxide
    Parameters
    ----------
    red_band : pd.Series indicating the name of the red band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=203=&sensor_id=168
    """
    return return_division(red_band, blue_band)


@require_pd_series
def rgr(red_band, green_band):
    """ 	Simple Ratio Red/Green Red-Green Ratio
    Parameters
    ----------
    red_band : pd.Series indicating the name of the red band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=213=&sensor_id=168
    """
    return return_division(red_band, green_band)


@require_pd_series
def ssred_nir(red_band, nir_band):
    """ Simple Ratio Red/NIR Ratio Vegetation-Index
    Parameters
    ----------
    red_band : pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=568=&sensor_id=168
    """
    return return_division(red_band, nir_band)


@require_pd_series
def swir_1_nir(nir_band, swir1_band):
    """ Simple Ratio SWIRI/NIR Ferrous Minerals
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir1_band : pd.Series indicating the name of the swir1 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=205=&sensor_id=168
    """
    return return_division(swir1_band, nir_band)


@require_pd_series
def sarvi2(red_band, blue_band, nir_band):
    """ Soil and Atmospherically Resistant Vegetation Index 2
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=387=&sensor_id=168
    """
    return 2.5 * (nir_band - red_band) + (nir_band + (6 * red_band) - (7.5 * blue_band))


@require_pd_series
def sbl(red_band, nir_band):
    """ 	Soil Background Line
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=353=&sensor_id=168
    """
    return nir_band - 2.4 * red_band


@require_pd_series
def sci(nir_band, swir1_band):
    """ Soil Composition Index
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir1_band: pd.Series indicating the name of the swir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=88=&sensor_id=168
    """
    numerator = swir1_band - nir_band
    denominator = swir1_band + nir_band
    return return_division(numerator, denominator)


@require_pd_series
def slavi(red_band, nir_band, swir2_band):
    """ Specific Leaf Area Vegetation Index
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=89=&sensor_id=168
    """
    denominator = red_band + swir2_band
    return return_division(nir_band, denominator)


@require_pd_series
def sqrt_nir_ir(red_band, nir_band):
    """ 	SQRT(IR/R)
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=200=&sensor_id=168
    """
    return np.sqrt(np.absolute(return_division(nir_band, red_band)))


@require_pd_series
def tass_brig(red_band, blue_band, green_band, nir_band, swir2_band):
    """ Tasselled Cap - brightness
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=91=&sensor_id=168
    """
    return 0.3037 * blue_band + 0.2793 * green_band + 0.4773 * red_band + 0.5585 * nir_band + 0.1863 * swir2_band


@require_pd_series
def tass_veg(red_band, blue_band, green_band, nir_band, swir1_band, swir2_band):
    """ Tasselled Cap - vegetation
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir1_band: pd.Series indicating the name of the swir1 band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=92=&sensor_id=168
    """
    return -.2848 * blue_band - .2435 * green_band - .5436 * red_band + \
           .7243 * nir_band + .084 * swir1_band - 0.18 * swir2_band


@require_pd_series
def tass_wet(red_band, blue_band, green_band, nir_band, swir1_band, swir2_band):
    """ Tasselled Cap - wetness
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir1_band: pd.Series indicating the name of the swir1 band in the dataframe
    swir2_band: pd.Series indicating the name of the swir2 band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=93=&sensor_id=168
    """
    return .1509 * blue_band + .1973 * green_band + .3279 * red_band + \
           .3406 * nir_band - .7112 * swir1_band - .4272 * swir2_band


@require_pd_series
def t_ndvi(red_band, nir_band):
    """ Transformed NDVI
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=202=&sensor_id=168
    """
    numerator = nir_band - red_band
    denominator = nir_band + red_band + 0.5
    return return_division(numerator, denominator)


@require_pd_series
def tvi(red_band, green_band):
    """ Transformed Vegetation Index
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=98=&sensor_id=168
    """
    numerator = red_band - green_band
    denominator = red_band + green_band
    return return_division(numerator, denominator) + 0.5


@require_pd_series
def varigreen(red_band, green_band, blue_band):
    """ Visible Atmospherically Resistant Index Green
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    green_band: pd.Series indicating the name of the green band in the dataframe
    blue_band: pd.Series indicating the name of the blue band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=356=&sensor_id=168
    """
    numerator = green_band - red_band
    denominator = green_band + red_band - blue_band
    return return_division(numerator, denominator)


@require_pd_series
def wdrvi(red_band, nir_band):
    """ Wide Dynamic Range Vegetation Index
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe

    for more info please visit:
    # https://www.indexdatabase.de/db/si-single.php?rsindex_id=125=&sensor_id=168
    """
    numerator = .1 * (nir_band - red_band)
    denominator = .1 * (nir_band + red_band)
    return return_division(numerator, denominator)


@require_pd_series
def ndbi(nir_band, swir1_band):
    """ Normalized Difference Built-up Index
    Parameters
    ----------
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir1_band: pd.Series indicating the name of the swir1 band in the dataframe

    for more info please visit:
    # https://www.linkedin.com/pulse/ndvi-ndbi-ndwi-calculation-using-landsat-7-8-tek-bahadur-kshetri
    """
    numerator = swir1_band - nir_band
    denominator = swir1_band + nir_band
    return return_division(numerator, denominator)


@require_pd_series
def bu(red_band, nir_band, swir1_band):
    """ Normalized Difference Built-up Index
    Parameters
    ----------
    red_band: pd.Series indicating the name of the red band in the dataframe
    nir_band: pd.Series indicating the name of the nir band in the dataframe
    swir1_band: pd.Series indicating the name of the swir1 band in the dataframe

    for more info please visit:
    # https://www.linkedin.com/pulse/ndvi-ndbi-ndwi-calculation-using-landsat-7-8-tek-bahadur-kshetri
    """
    ndvi_num = nir_band - red_band
    ndvi_dem = nir_band + red_band
    ndbi_num = swir1_band - nir_band
    ndbi_dem = swir1_band + nir_band
    ndvi = return_division(ndvi_num, ndvi_dem)
    nbdi = return_division(ndbi_num, ndbi_dem)
    return nbdi - ndvi


@require_pd_series
def mndwi(green_band, swir1_band):
    """ Normalized Difference Built-up Index
    Parameters
    ----------
    green_band: pd.Series indicating the name of the green band in the dataframe
    swir1_band: pd.Series indicating the name of the swir1 band in the dataframe

    for more info please visit:
    # https://www.linkedin.com/pulse/ndvi-ndbi-ndwi-calculation-using-landsat-7-8-tek-bahadur-kshetri
    """
    numerator = green_band - swir1_band
    denominator = green_band + swir1_band
    return return_division(numerator, denominator)