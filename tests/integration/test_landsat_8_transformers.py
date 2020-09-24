import unittest
import numpy as np

from feature_raster.exceptions.some_exceptions import InvalidImage
from feature_raster.project_enums.LandsatEnums import LandsatEnums
from feature_raster.Sensors.Landsat import Landsat8
from feature_raster.Transformers.Landsat import Landsat8Transformer
from tests.paths import small_2018_dataset

# TODO try to create a mock of the dataframe of a Landsat8 and make some of the indexes of that dataframe
#  zeros, with that then create the tests necesary no detect where some denominators may be 0 an return and
#  np.inf when a new index is created


class Landsat8ExceptionsTest(unittest.TestCase):
    def test_check_that_x_is_a_landsat8_image(self):
        landsat8 = []
        with self.assertRaises(InvalidImage):
            transformer8 = Landsat8Transformer()
            transformer8.transform(landsat8)


class Landsat8CorrectTransformsTest(unittest.TestCase):
    def setUp(self):
        self.landsat8 = Landsat8(small_2018_dataset)
        self.transformer = Landsat8Transformer()
        self.transformation = self.transformer.transform(self.landsat8)

    def check_series_equal_expected(self, series_name, array):
        index = self.transformation[series_name].head().to_numpy()
        array = np.array(array)
        return np.allclose(index, array)

    def test_ndvi_landsat8(self):
        array = [0.220217424, 0.23786527, 0.241382438, 0.095803022, -0.029366701]
        equals = self.check_series_equal_expected(LandsatEnums.ndvi.value, array)
        self.assertTrue(equals)

    def test_atsavi_lansat8(self):
        array = [16855.05536, 15848.27936, 14355.11016, 9809.43056, 7222.70796]
        equals = self.check_series_equal_expected(LandsatEnums.atsavi.value, array)
        self.assertTrue(equals)

    def test_afri1600_landsat8(self):
        array = [1.525562881, 1.673295266, 1.847946804, 1.564416779, 1.318282696]
        equals = self.check_series_equal_expected(LandsatEnums.afri1600.value, array)
        self.assertTrue(equals)

    def test_alteration_lansat8(self):
        array = [1.215832017, 1.180432149, 1.183795861, 1.112880296, 1.060208167]
        equals = self.check_series_equal_expected(LandsatEnums.alteration.value, array)
        self.assertTrue(equals)

    def test_avi_landsat8(self):
        array = [0.000342818, 0.000344116, 0.000375799, 0.001168224, -0.004347826]
        equals = self.check_series_equal_expected(LandsatEnums.avi.value, array)
        self.assertTrue(equals)

    def test_arvi2_landsat8(self):
        array = [-5.93094e-09, -7.03196e-09, -8.65191e-09, -1.25184e-08, -1.61576e-08]
        equals = self.check_series_equal_expected(LandsatEnums.arvi2.value, array)
        self.assertTrue(equals)

    def test_bwdrvi_landsat8(self):
        array = [-16768428.3, -14473719.1, -12284251.5, -8713891.9, -6789524.6]
        equals = self.check_series_equal_expected(LandsatEnums.bwdrvi.value, array)
        self.assertTrue(equals)
    
    def test_ci_green_landsat8(self):
        array = [165299000., 144409526., 122330214., 87629449., 69527891.]
        equals = self.check_series_equal_expected(LandsatEnums.ci_green.value, array)
        self.assertTrue(equals)

    def test_cvi_landsat8(self):
        array = [0.000153008, 0.000170092, 0.00018306, 0.000135409, 0.000103099]
        equals = self.check_series_equal_expected(LandsatEnums.cvi.value, array)
        self.assertTrue(equals)

    def test_ci_landsat8(self):
        array = [-107225349., -89171447., -75124829., -71975811., -72098466.]
        equals = self.check_series_equal_expected(LandsatEnums.ci.value, array)
        self.assertTrue(equals)

    def test_ctvi_landsat8(self):
        array = [0.37358624, 0.401727172, 0.407499315, 0.211498896, 0.104243253]
        equals = self.check_series_equal_expected(LandsatEnums.ctvi.value, array)
        self.assertTrue(equals)

    def test_cri550_landsat8(self):
        array = [106176714., 91460322., 80307976., 79744500., 81801824.]
        equals = self.check_series_equal_expected(LandsatEnums.cri550.value, array)
        self.assertTrue(equals)

    def test_gdvi_landsat8(self):
        array = [5936., 5574., 4746., 841., -1544.]
        equals = self.check_series_equal_expected(LandsatEnums.gdvi.value, array)
        self.assertTrue(equals)

    def test_dvimss_landsat8(self):
        array = [28462.2, 26984.2, 24481., 15419.4, 10182.8]
        equals = self.check_series_equal_expected(LandsatEnums.dvimss.value, array)
        self.assertTrue(equals)

    def test_evi_landsat8(self):
        array = [-2769079.5, 7914914.5, 29446884.5, 69180718.5, 89555578.]
        equals = self.check_series_equal_expected(LandsatEnums.evi.value, array)
        self.assertTrue(equals)

    def test_evi2_landsat8(self):
        array = [-166898505.8, -140764645.8, -114406447., -79069910.6, -61261016.2]
        equals = self.check_series_equal_expected(LandsatEnums.evi2.value, array)
        self.assertTrue(equals)

    def test_evi22_landsat8(self):
        array = [-166882428.9, -140750098.1, -114393370.3, -79057620.9, -61248969.2]
        equals = self.check_series_equal_expected(LandsatEnums.evi22.value, array)
        self.assertTrue(equals)

    def test_fe2plus_landsat8(self):
        array = [173972596., 149787741., 116386179., 68878844., 47483636.]
        equals = self.check_series_equal_expected(LandsatEnums.fe2plus.value, array)
        self.assertTrue(equals)

    def test_ferricoxides_landsat8(self):
        array = [211509018., 176802993., 137766895., 76643948., 50332842.]
        equals = self.check_series_equal_expected(LandsatEnums.ferricoxides.value, array)
        self.assertTrue(equals)

    def test_ferrous_iron_landsat8(self):
        array = [173972596., 149787741., 116386179., 68878844., 47483636.]
        equals = self.check_series_equal_expected(LandsatEnums.ferrous_iron.value, array)
        self.assertTrue(equals)

    def test_ferrous_silicates_landsat8(self):
        array = [140844618., 115787664., 85609768., 55062152., 41348145.]
        equals = self.check_series_equal_expected(LandsatEnums.ferrous_silicates.value, array)
        self.assertTrue(equals)

    def test_gemi_landsat8(self):
        array = [-34049736.96, -33791462.1, -28334445.01, -2945307.347, -228176.6283]
        equals = self.check_series_equal_expected(LandsatEnums.gemi.value, array)
        self.assertTrue(equals)

    def test_gvmi_landsat8(self):
        array = [0.200551732, 0.208536979, 0.233496416, 0.163866885, 0.098004495]
        equals = self.check_series_equal_expected(LandsatEnums.gvmi.value, array)
        self.assertTrue(equals)

    def test_gossan_landsat8(self):
        array = [1.266918385, 1.255611642, 1.203754634, 0.968931798, 0.821260233]
        equals = self.check_series_equal_expected(LandsatEnums.gossan.value, array)
        self.assertTrue(equals)
    
    def test_gari_landsat8(self):
        array = [1.018018018, 1.10062182, 1.301090909, 167.2, 0.272877164]
        equals = self.check_series_equal_expected(LandsatEnums.gari.value, array)
        self.assertTrue(equals)

    def test_gli_landsat8(self):
        array = [-0.006243168, 0.005501882, 0.015074521, 0.026111382, 0.036432659]
        equals = self.check_series_equal_expected(LandsatEnums.gli.value, array)
        self.assertTrue(equals)

    def test_gndvi_landsat8(self):
        array = [0.224933687, 0.225924125, 0.209777228, 0.044874873, -0.092190112]
        equals = self.check_series_equal_expected(LandsatEnums.gndvi.value, array)
        self.assertTrue(equals)

    def test_gosavi_landsat8(self):
        array = [0.224932323, 0.225922659, 0.209775744, 0.04487449, -0.092189232]
        equals = self.check_series_equal_expected(LandsatEnums.gosavi.value, array)
        self.assertTrue(equals)
    
    def test_gsavi_landsat8(self):
        array = [0.224927295, 0.225917257, 0.209770274, 0.044873077, -0.092185984]
        equals = self.check_series_equal_expected(LandsatEnums.gsavi.value, array)
        self.assertTrue(equals)

    def test_gbndvi_landsat8(self):
        array = [-0.120907212, -0.116905109, -0.13407998, -0.291815848, -0.408220458]
        equals = self.check_series_equal_expected(LandsatEnums.gbndvi.value, array)
        self.assertTrue(equals)
        
    def test_grndvi_landsat8(self):
        array = [-0.119638334, -0.109966748, -0.116726369, -0.269873229, -0.387182588]
        equals = self.check_series_equal_expected(LandsatEnums.grndvi.value, array)
        self.assertTrue(equals)

    @unittest.skip("searching function in EXCEL to test")
    def test_hue_landsat8(self):
        # TODO im not sure of the values i used to test this.
        array = [2.03702E-08, 1.17913E-07, 8.10483E-08, 3.141592562, 3.141592635]
        equals = self.check_series_equal_expected(LandsatEnums.hue.value, array)
        self.assertTrue(equals)

    # TODO search a way of create the variables of IVI Index

    def test_intensity_landsat8(self):
        array = [1014.360656, 932.3934426, 861.8360656, 850.4590164, 857.442623]
        equals = self.check_series_equal_expected(LandsatEnums.intensity.value, array)
        self.assertTrue(equals)

    def test_laterite_landsat8(self):
        array = [1.215832017, 1.180432149, 1.183795861, 1.112880296, 1.060208167]
        equals = self.check_series_equal_expected(LandsatEnums.laterite.value, array)
        self.assertTrue(equals)

    def test_logratio_landsat8(self):
        array = [0.194463696, 0.210641626, 0.213882705, 0.083469442, -0.025514929]
        equals = self.check_series_equal_expected(LandsatEnums.logratio.value, array)
        self.assertTrue(equals)
        
    def test_mcrig_landsat8(self):
        array = [-2505265., -438567., -615825., 391640., 1535604.]
        equals = self.check_series_equal_expected(LandsatEnums.mcrig.value, array)
        self.assertTrue(equals)

    def test_mvi_landsat8(self):
        array = [1.235136787, 1.293559148, 1.359392073, 1.250766479, 1.14816493]
        equals = self.check_series_equal_expected(LandsatEnums.mvi.value, array)
        self.assertTrue(equals)

    def test_msrnir_red_landsat8(self):
        array = [0.220217424, 0.23786527, 0.241382438, 0.095803022, -0.029366701]
        equals = self.check_series_equal_expected(LandsatEnums.msrnir_red.value, array)
        self.assertTrue(equals)

    def test_norm_nir_landsat8(self):
        array = [0.440180833, 0.445016626, 0.441636815, 0.365063386, 0.306408706]
        equals = self.check_series_equal_expected(LandsatEnums.norm_nir.value, array)
        self.assertTrue(equals)        

    def test_norm_r_landsat8(self):
        array = [0.28129851, 0.273989936, 0.269887372, 0.301230425, 0.324949617]
        equals = self.check_series_equal_expected(LandsatEnums.norm_r.value, array)
        self.assertTrue(equals)
        
    def test_norm_g_landsat8(self):
        array = [0.278520657, 0.280993438, 0.288475812, 0.333706189, 0.368641677]
        equals = self.check_series_equal_expected(LandsatEnums.norm_g.value, array)
        self.assertTrue(equals)
    
    def test_nli_landsat8(self):
        array = [0.515695698, 0.529236292, 0.531917275, 0.41585626, 0.306971546]
        equals = self.check_series_equal_expected(LandsatEnums.nli.value, array)
        self.assertTrue(equals)

    def test_ppr_landsat8(self):
        array = [-0.007520986, -0.001516181, -0.00251074, 0.002239642, 0.01116639]
        equals = self.check_series_equal_expected(LandsatEnums.ppr.value, array)
        self.assertTrue(equals)
        
    def test_pvr_landsat8(self):
        array = [-0.004962055, 0.0126193, 0.033290949, 0.051148042, 0.062993956]
        equals = self.check_series_equal_expected(LandsatEnums.pvr.value, array)
        self.assertTrue(equals)
    
    def test_siwsi_landsat8(self):
        array = [0.105200178, 0.12799284, 0.152324015, 0.111413815, 0.068972791]
        equals = self.check_series_equal_expected(LandsatEnums.siwsi.value, array)
        self.assertTrue(equals)

    def test_bndvi_landsat8(self):
        array = [0.217781126, 0.224484839, 0.207375711, 0.04710978, -0.081107216]
        equals = self.check_series_equal_expected(LandsatEnums.bndvi.value, array)
        self.assertTrue(equals)

    def test_mndvi_landsat8(self):
        array = [0.200549655, 0.208534782, 0.233494074, 0.163863299, 0.097999567]
        equals = self.check_series_equal_expected(LandsatEnums.mndvi.value, array)
        self.assertTrue(equals)

    def test_ri_landsat8(self):
        array = [0.004962055, -0.0126193, -0.033290949, -0.051148042, -0.062993956]
        equals = self.check_series_equal_expected(LandsatEnums.ri.value, array)
        self.assertTrue(equals)
    
    def test_ndsi_landsat8(self):
        array = [0.097404503, 0.082750637, 0.084163481, 0.053424842, 0.029224312]
        equals = self.check_series_equal_expected(LandsatEnums.ndsi.value, array)
        self.assertTrue(equals)

    def test_ndvic_landsat8(self):
        array = [-0.437582526, -0.438539639, -0.404727483, -0.138583483, 0.038836616]
        equals = self.check_series_equal_expected(LandsatEnums.ndvic.value, array)
        self.assertTrue(equals)

    def test_pndvi_landsat8(self):
        array = [-0.313687607, -0.305663323, -0.315253559, -0.451945144, -0.549564496]
        equals = self.check_series_equal_expected(LandsatEnums.pndvi.value, array)
        self.assertTrue(equals)

    def test_rbndvi_landsat8(self):
        array = [0.439767858, 0.452487357, 0.461007992, 0.396639283, 0.34476593]
        equals = self.check_series_equal_expected(LandsatEnums.rbndvi.value, array)
        self.assertTrue(equals)

    def test_if_landsat8(self):
        array = [132.6451613, 676.9655172, 424.8888889, -489.05, -99.28712871]
        equals = self.check_series_equal_expected(LandsatEnums.if_index.value, array)
        self.assertTrue(equals)
        
    def test_tm5_tm7_landsat8(self):
        array = [1.215832017, 1.180432149, 1.183795861, 1.112880296, 1.060208167]
        equals = self.check_series_equal_expected(LandsatEnums.tm5_tm7.value, array)
        self.assertTrue(equals)

    def test_bgi_landsat8(self):
        array = [1.067664027, 1.080741439, 1.092515941, 1.073743017, 1.046905751]
        equals = self.check_series_equal_expected(LandsatEnums.bgi.value, array)
        self.assertTrue(equals)
    
    def test_sr550_670_landsat8(self):
        array = [0.990124891, 1.025561164, 1.068874806, 1.107810373, 1.134457951]
        equals = self.check_series_equal_expected(LandsatEnums.sr550_670.value, array)
        self.assertTrue(equals)

    def test_sr860_550_landsat8(self):
        array = [1.580424367, 1.583726045, 1.530931872, 1.09396648, 0.831183031]
        equals = self.check_series_equal_expected(LandsatEnums.sr860_550.value, array)
        self.assertTrue(equals)
        
    def test_rdi_landsat8(self):
        array = [0.665903607, 0.654896515, 0.621410303, 0.718414871, 0.821494344]
        equals = self.check_series_equal_expected(LandsatEnums.rdi.value, array)
        self.assertTrue(equals)

    def test_srmir_red_landsat8(self):
        array = [1.04201762, 1.063688111, 1.016859978, 0.870652308, 0.774621682]
        equals = self.check_series_equal_expected(LandsatEnums.srmir_red.value, array)
        self.assertTrue(equals)

    def test_grvi_landsat8(self):
        array = [1.580424367, 1.583726045, 1.530931872, 1.09396648, 0.831183031]
        equals = self.check_series_equal_expected(LandsatEnums.grvi.value, array)
        self.assertTrue(equals)

    def test_srnir_mir_landsat8(self):
        array = [1.501718852, 1.526958805, 1.609242709, 1.391953369, 1.217293835]
        equals = self.check_series_equal_expected(LandsatEnums.srnir_mir.value, array)
        self.assertTrue(equals)

    def test_dvi_landsat8(self):
        array = [1.564817504, 1.624207926, 1.636374507, 1.211907414, 0.942942198]
        equals = self.check_series_equal_expected(LandsatEnums.dvi.value, array)
        self.assertTrue(equals)

    def test_io_landsat8(self):
        array = [0.994895011, 0.972123617, 0.930877115, 0.906734007, 0.901386404]
        equals = self.check_series_equal_expected(LandsatEnums.io.value, array)
        self.assertTrue(equals)

    def test_rgr_landsat8(self):
        array = [1.009973599, 0.975075924, 0.935563262, 0.902681564, 0.881478242]
        equals = self.check_series_equal_expected(LandsatEnums.rgr.value, array)
        self.assertTrue(equals)

    def test_ssred_nir_landsat8(self):
        array = [0.639052156, 0.615684719, 0.611107052, 0.825145542, 1.060510392]
        equals = self.check_series_equal_expected(LandsatEnums.ssred_nir.value, array)
        self.assertTrue(equals)
    
    def test_swir_1_nir_landsat8(self):
        array = [0.809626926, 0.773060901, 0.735622945, 0.799509754, 0.870955012]
        equals = self.check_series_equal_expected(LandsatEnums.swir1_nir.value, array)
        self.assertTrue(equals)

    def test_sarvi2_landsat8(self):
        array = [14857., 13684., 9788., -4280., -12256.]
        equals = self.check_series_equal_expected(LandsatEnums.sarvi2.value, array)
        self.assertTrue(equals)

    def test_sbl_landsat8(self):
        array = [-8626.6, -7223.4, -6386.2, -9598.6, -11746.8]
        equals = self.check_series_equal_expected(LandsatEnums.sbl.value, array)
        self.assertTrue(equals)

    def test_sci_landsat8(self):
        array = [-0.105200178, -0.12799284, -0.152324015, -0.111413815, -0.068972791]
        equals = self.check_series_equal_expected(LandsatEnums.sci.value, array)
        self.assertTrue(equals)

    def test_slavi_landsat8(self):
        array = [0.766309501, 0.787041374, 0.811347602, 0.647852842, 0.531348291]
        equals = self.check_series_equal_expected(LandsatEnums.slavi.value, array)
        self.assertTrue(equals)
        
    def test_sqrt_ir_r_landsat8(self):
        array = [1.250926658, 1.274444164, 1.279208547, 1.100866665, 0.971052109]
        equals = self.check_series_equal_expected(LandsatEnums.sqrt_nir_ir.value, array)
        self.assertTrue(equals)

    # TODO as im an idiot i forgot cirrus band on landsat8 so i have to createdi
    def test_tasseled_brightness_landsat8(self):
        array = [21971.6286, 20311.3253, 18444.1311, 15840.5164, 14527.9237]
        equals = self.check_series_equal_expected(LandsatEnums.tas_bri.value, array)
        self.assertTrue(equals)
        
    def test_tasseled_vegetation_landsat8(self):
        array = [-193.1676, 38.4574, -54.463, -2625.5841, -4218.6128]
        equals = self.check_series_equal_expected(LandsatEnums.tas_veg.value, array)
        self.assertTrue(equals)

    def test_tasseled_wetness_landsat8(self):
        array = [-1428.289, -1012.3194, -269.8702, 522.0743, 1010.2072]
        equals = self.check_series_equal_expected(LandsatEnums.tas_wet.value, array)
        self.assertTrue(equals)

    def test_transformed_ndvi_landsat8(self):
        array = [0.220213268, 0.237860402, 0.241376964, 0.095800341, -0.029365763]
        equals = self.check_series_equal_expected(LandsatEnums.t_ndvi.value, array)
        self.assertTrue(equals)

    def test_tvi_landsat8(self):
        array = [0.504962055, 0.4873807, 0.466709051, 0.448851958, 0.437006044]
        equals = self.check_series_equal_expected(LandsatEnums.tvi.value, array)
        self.assertTrue(equals)

    def test_varigreen_landsat8(self):
        array = [-0.010025555, 0.025641026, 0.069247415, 0.107279222, 0.131171346]
        equals = self.check_series_equal_expected(LandsatEnums.varigreen.value, array)
        self.assertTrue(equals)

    def test_wdrvi_landsat8(self):
        array = [0.220217424, 0.23786527, 0.241382438, 0.095803022, -0.029366701]
        equals = self.check_series_equal_expected(LandsatEnums.wdrvi.value, array)
        self.assertTrue(equals)

    def test_ndbi_landsat8(self):
        array = [-0.105200178, -0.12799284, -0.152324015, -0.111413815, -0.068972791]
        equals = self.check_series_equal_expected(LandsatEnums.ndbi.value, array)
        self.assertTrue(equals)

    def test_bu_landsat8(self):
        array = [-0.325417602, -0.365858109, -0.393706453, -0.207216836, -0.03960609]
        equals = self.check_series_equal_expected(LandsatEnums.bu.value, array)
        self.assertTrue(equals)

    def test_mndwi_landsat8(self):
        array = [-0.122635439, -0.100847458, -0.059349679, 0.066873286, 0.160144606]
        equals = self.check_series_equal_expected(LandsatEnums.mndwi.value, array)
        self.assertTrue(equals)



