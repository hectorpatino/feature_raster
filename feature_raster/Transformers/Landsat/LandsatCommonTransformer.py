from sklearn.base import BaseEstimator, TransformerMixin
from feature_raster.project_enums import LandsatEnums

from .indexes import *


class LandsatGeneralTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cirrus=False):
        self.cirrus = cirrus

    def fit(self, x, y=None):
        return self

    def transform(self, landsatobject):
        x = landsatobject.dataframe.copy()
        x[LandsatEnums.ndvi.value] = normalize_difference_indexes_minus_plus(x[LandsatEnums.red.value],
                                                                             x[LandsatEnums.nir.value])
        x[LandsatEnums.atsavi.value] = atsavi(x[LandsatEnums.nir.value], x[LandsatEnums.red.value])
        x[LandsatEnums.afri1600.value] = afri1600(swir_1=x[LandsatEnums.swir1.value],
                                                  nir=x[LandsatEnums.nir.value])
        x[LandsatEnums.alteration.value] = alteration(x[LandsatEnums.swir1.value],
                                                      x[LandsatEnums.swir2.value])
        x[LandsatEnums.avi.value] = avi(x[LandsatEnums.nir.value],
                                        x[LandsatEnums.red.value])
        x[LandsatEnums.arvi2.value] = arvi2(x[LandsatEnums.red.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.bwdrvi.value] = bwdrvi(x[LandsatEnums.blue.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.ci_green.value] = ci_green(x[LandsatEnums.nir.value], x[LandsatEnums.green.value])
        x[LandsatEnums.cvi.value] = cvi(x[LandsatEnums.nir.value], x[LandsatEnums.green.value],
                                        x[LandsatEnums.red.value])
        x[LandsatEnums.ci.value] = ci(x[LandsatEnums.red.value], x[LandsatEnums.blue.value])
        x[LandsatEnums.ctvi.value] = ctvi(x[LandsatEnums.ndvi.value])
        x[LandsatEnums.cri550.value] = cri550(x[LandsatEnums.blue.value], x[LandsatEnums.green.value])
        x[LandsatEnums.gdvi.value] = gdvi(x[LandsatEnums.green.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.dvimss.value] = dvimss(x[LandsatEnums.nir.value], x[LandsatEnums.red.value])
        x[LandsatEnums.evi.value] = evi(x[LandsatEnums.nir.value],
                                        x[LandsatEnums.red.value],
                                        x[LandsatEnums.blue.value])
        x[LandsatEnums.evi2.value] = evi2(x[LandsatEnums.nir.value], x[LandsatEnums.red.value])
        x[LandsatEnums.evi22.value] = evi22(x[LandsatEnums.nir.value], x[LandsatEnums.red.value])
        x[LandsatEnums.fe2plus.value] = fe2plus(x[LandsatEnums.green.value], x[LandsatEnums.nir.value],
                                                x[LandsatEnums.swir2.value])
        x[LandsatEnums.ferricoxides.value] = ferric_oxides(x[LandsatEnums.nir.value],
                                                           x[LandsatEnums.swir1.value])
        x[LandsatEnums.ferrous_iron.value] = ferrous_iron(x[LandsatEnums.swir2.value],
                                                          x[LandsatEnums.nir.value],
                                                          x[LandsatEnums.green.value])
        x[LandsatEnums.ferrous_silicates.value] = ferrous_silicates(x[LandsatEnums.swir1.value],
                                                                    x[LandsatEnums.swir2.value])
        x[LandsatEnums.gemi.value] = gemi(x[LandsatEnums.red.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.gvmi.value] = gvmi(x[LandsatEnums.nir.value], x[LandsatEnums.swir2.value])
        x[LandsatEnums.gossan.value] = gossan(x[LandsatEnums.red.value], x[LandsatEnums.swir1.value])
        x[LandsatEnums.gari.value] = gari(x[LandsatEnums.blue.value], x[LandsatEnums.green.value],
                                          x[LandsatEnums.red.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.gli.value] = gli(x[LandsatEnums.blue.value], x[LandsatEnums.green.value],
                                        x[LandsatEnums.red.value])
        x[LandsatEnums.gndvi.value] = gndvi(x[LandsatEnums.nir.value], x[LandsatEnums.green.value])
        x[LandsatEnums.gosavi.value] = gosavi(x[LandsatEnums.green.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.gsavi.value] = gsavi(x[LandsatEnums.green.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.gbndvi.value] = gbndvi(x[LandsatEnums.blue.value], x[LandsatEnums.green.value],
                                              x[LandsatEnums.nir.value])
        x[LandsatEnums.grndvi.value] = grndvi(x[LandsatEnums.red.value], x[LandsatEnums.green.value],
                                              x[LandsatEnums.nir.value])
        x[LandsatEnums.hue.value] = hue(x[LandsatEnums.red.value], x[LandsatEnums.green.value],
                                        x[LandsatEnums.blue.value])
        x[LandsatEnums.intensity.value] = intensity(x[LandsatEnums.red.value], x[LandsatEnums.green.value],
                                                    x[LandsatEnums.blue.value])
        x[LandsatEnums.laterite.value] = laterite(x[LandsatEnums.swir1.value], x[LandsatEnums.swir2.value])
        x[LandsatEnums.logratio.value] = logratio(x[LandsatEnums.red.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.mcrig.value] = mcrig(x[LandsatEnums.blue.value], x[LandsatEnums.green.value],
                                            x[LandsatEnums.nir.value])
        x[LandsatEnums.mvi.value] = mvi(x[LandsatEnums.nir.value], x[LandsatEnums.swir1.value])
        x[LandsatEnums.msrnir_red.value] = msrnir_red(x[LandsatEnums.red.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.norm_r.value] = norm_r(x[LandsatEnums.red.value], x[LandsatEnums.green.value],
                                              x[LandsatEnums.nir.value])
        x[LandsatEnums.norm_nir.value] = norm_nir(x[LandsatEnums.red.value], x[LandsatEnums.green.value],
                                                  x[LandsatEnums.nir.value])
        x[LandsatEnums.norm_g.value] = norm_g(x[LandsatEnums.red.value], x[LandsatEnums.green.value],
                                              x[LandsatEnums.nir.value])
        x[LandsatEnums.nli.value] = nli(x[LandsatEnums.red.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.ppr.value] = ppr(x[LandsatEnums.blue.value], x[LandsatEnums.green.value])
        x[LandsatEnums.pvr.value] = pvr(x[LandsatEnums.red.value], x[LandsatEnums.green.value])
        x[LandsatEnums.siwsi.value] = siwsi(x[LandsatEnums.nir.value], x[LandsatEnums.swir1.value])
        x[LandsatEnums.bndvi.value] = bndvi(x[LandsatEnums.blue.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.mndvi.value] = mndvi(x[LandsatEnums.nir.value], x[LandsatEnums.swir2.value])
        x[LandsatEnums.ri.value] = ri(x[LandsatEnums.red.value], x[LandsatEnums.green.value])
        x[LandsatEnums.ndsi.value] = ndsi(x[LandsatEnums.swir1.value], x[LandsatEnums.swir2.value])
        x[LandsatEnums.ndvic.value] = ndvic(x[LandsatEnums.red.value], x[LandsatEnums.nir.value],
                                            x[LandsatEnums.swir1.value], x[LandsatEnums.swir2.value])
        x[LandsatEnums.pndvi.value] = pndvi(x[LandsatEnums.red.value], x[LandsatEnums.green.value],
                                            x[LandsatEnums.blue.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.rbndvi.value] = rbndvi(x[LandsatEnums.red.value], x[LandsatEnums.blue.value],
                                              x[LandsatEnums.nir.value])
        x[LandsatEnums.if_index.value] = if_index(x[LandsatEnums.red.value], x[LandsatEnums.green.value],
                                                  x[LandsatEnums.blue.value])
        x[LandsatEnums.tm5_tm7.value] = tm5_tm7(x[LandsatEnums.swir1.value], x[LandsatEnums.swir2.value])

        x[LandsatEnums.sr550_670.value] = sr550_670(x[LandsatEnums.red.value], x[LandsatEnums.green.value])
        x[LandsatEnums.sr860_550.value] = sr860_550(x[LandsatEnums.green.value],
                                                    x[LandsatEnums.nir.value])
        x[LandsatEnums.rdi.value] = rdi(x[LandsatEnums.nir.value], x[LandsatEnums.swir2.value])
        x[LandsatEnums.srmir_red.value] = srmir_red(x[LandsatEnums.red.value], x[LandsatEnums.swir2.value])
        x[LandsatEnums.grvi.value] = grvi(x[LandsatEnums.green.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.srnir_mir.value] = srnir_mir(x[LandsatEnums.nir.value], x[LandsatEnums.swir2.value])
        x[LandsatEnums.dvi.value] = dvi(x[LandsatEnums.red.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.io.value] = io(x[LandsatEnums.red.value], x[LandsatEnums.blue.value])
        x[LandsatEnums.rgr.value] = rgr(x[LandsatEnums.red.value], x[LandsatEnums.green.value])
        x[LandsatEnums.ssred_nir.value] = ssred_nir(x[LandsatEnums.red.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.swir1_nir.value] = swir_1_nir(x[LandsatEnums.nir.value], x[LandsatEnums.swir1.value])
        x[LandsatEnums.sarvi2.value] = sarvi2(x[LandsatEnums.red.value], x[LandsatEnums.blue.value],
                                              x[LandsatEnums.nir.value])
        x[LandsatEnums.sbl.value] = sbl(x[LandsatEnums.red.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.sci.value] = sci(x[LandsatEnums.nir.value], x[LandsatEnums.swir1.value])
        x[LandsatEnums.slavi.value] = slavi(x[LandsatEnums.red.value], x[LandsatEnums.nir.value],
                                            x[LandsatEnums.swir2.value])
        x[LandsatEnums.sqrt_nir_ir.value] = sqrt_nir_ir(x[LandsatEnums.red.value],
                                                        x[LandsatEnums.nir.value])
        x[LandsatEnums.tas_bri.value] = tass_brig(x[LandsatEnums.red.value],
                                                  x[LandsatEnums.blue.value],
                                                  x[LandsatEnums.green.value],
                                                  x[LandsatEnums.nir.value],
                                                  x[LandsatEnums.swir2.value])
        x[LandsatEnums.tas_veg.value] = tass_veg(x[LandsatEnums.red.value], x[LandsatEnums.blue.value],
                                                 x[LandsatEnums.green.value], x[LandsatEnums.nir.value],
                                                 x[LandsatEnums.swir1.value], x[LandsatEnums.swir2.value])
        x[LandsatEnums.tas_wet.value] = tass_wet(x[LandsatEnums.red.value], x[LandsatEnums.blue.value],
                                                 x[LandsatEnums.green.value], x[LandsatEnums.nir.value],
                                                 x[LandsatEnums.swir1.value], x[LandsatEnums.swir2.value])
        x[LandsatEnums.t_ndvi.value] = t_ndvi(x[LandsatEnums.red.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.tvi.value] = tvi(x[LandsatEnums.red.value], x[LandsatEnums.green.value])
        x[LandsatEnums.varigreen.value] = varigreen(x[LandsatEnums.red.value], x[LandsatEnums.green.value],
                                                    x[LandsatEnums.blue.value])
        x[LandsatEnums.wdrvi.value] = wdrvi(x[LandsatEnums.red.value], x[LandsatEnums.nir.value])
        x[LandsatEnums.ndbi.value] = ndbi(x[LandsatEnums.nir.value], x[LandsatEnums.swir1.value])
        x[LandsatEnums.bu.value] = bu(x[LandsatEnums.red.value], x[LandsatEnums.nir.value],
                                      x[LandsatEnums.swir1.value])
        x[LandsatEnums.mndwi.value] = mndwi(x[LandsatEnums.green.value], x[LandsatEnums.swir1.value])
        return x

