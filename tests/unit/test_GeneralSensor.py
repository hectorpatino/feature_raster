import unittest
from feature_raster.Sensors.Landsat import Landsat8
from tests.paths import small_2018_dataset
from feature_raster.exceptions import NoCobertureSeries


class GeneralSensorInstancesTest(unittest.TestCase):
    def setUp(self):
        self.landsat8 = Landsat8(small_2018_dataset)

    def test_attempt_to_execute_select_df_of_cobertures_method_of_general_sensor(self):
        """When trying to execute the method without having a coberture pandas series from Sensor
        """
        with self.assertRaises(NoCobertureSeries):
            self.landsat8.select_df_of_cobertures()

    def test_that_dataframe_is_a_dataframe(self):
        """If dataframe attribute in Landsat8 then raise type error"""
        some_random_variable = []
        another_random_variable = ""
        # TODO dataframe MUST BE of the same dimensions
        with self.assertRaises(TypeError):
            self.landsat8.dataframe = some_random_variable
            self.landsat8.dataframe = another_random_variable
