import rasterio

import numpy as np
import pandas as pd
import geopandas as gpd

from os import path
from rasterio import features

from feature_raster.exceptions import InvalidTypeOfGeom, NoCobertureSeries
from feature_raster.project_enums import GeneralSensorEnums


class GeneralSensor:

    def __init__(self, img_path: str):

        if path.exists(img_path):
            self.img_path = img_path
            self.bounds = dict()
            self.crs = None
            self.dtypes = None
            self.indexes = None
            self.meta = {}  # count, driver, dtype,
            self.res = None
            self.__dataframe = None
            self.__read_metadata_file()
            self._to_pandas()
        else:
            raise ValueError('File does not exists')

    @property
    def dataframe(self):
        return self.__dataframe

    @dataframe.setter
    def dataframe(self, dataframe):
        """Generally you want to use this setter after you """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("the property of dataframe MUST BE a pandas dataframe")
        self.__dataframe = dataframe

    def __read_metadata_file(self):
        """used in the inicialization of some of the parameters of the instance"""
        with rasterio.open(self.img_path) as dataset:
            left, bottom, right, top = dataset.bounds
            self.bounds = {"left": left, "bottom": bottom,
                           "right": right, "top": top}
            self.crs = dataset.crs
            self.dtypes = dataset.dtypes
            self.indexes = dataset.indexes
            self.meta = dataset.meta
            self.res = dataset.res[0]

    def _to_pandas(self):
        """used to determinated dataframe property on inicialization"""
        with rasterio.open(self.img_path) as dataset:
            rasters = [dataset.read(band).astype(np.dtype(getattr(np, dataset.dtypes[band - 1])))
                       for band in self.indexes]
            bands = [pd.Series(raster.flatten().tolist()) for raster in rasters]
            cols = [f"BAND_{x + 1}" for x in range(len(bands))]
            df = pd.concat(bands, axis=1, keys=cols)
            self.__dataframe = df

    def dataframe_to_raster(self, name: str, driver="GTiff"):
        """Allow to convert the dataframe property into a raster with the same
        caracteristicis"""
        # TODO  the next version it should get the directory and the filename
        # TODO generated the format file automaticaly
        meta = self.meta.copy()
        meta.update(count=len(self.__dataframe.columns))
        # OPTIMIZEME search for a better way to optimize de dtype while reading
        #  it seams that the uint dtype can fuck me really hard if i don't fix that
        meta.update(dtype='float64')
        meta.update(driver=driver)
        with rasterio.open(name, 'w+', **meta) as dataset:
            for index, column in enumerate(self.__dataframe.columns.tolist(), 1):
                array_column = self.__dataframe[column].to_numpy().reshape((meta["height"], meta["width"]))\
                    .astype('float64')
                dataset.write(array_column, index)
        self._generate_info_of_bands(name)

    def _generate_info_of_bands(self, name):
        # TODO generate the links of the transfomations
        with open(f"{name}.txt", "w") as f:
            for index, column in enumerate(self.__dataframe.columns.tolist(), 1):
                f.write(f"{index} --> {column} \n")

    def coberture_to_pandas(self, coberture_file, coberture_column="coberture", fillna: int = -9999):
        """Takes a geodataframe of cobertures created by the user usually using a GIS software
        convert it to a multidimensional array (as a raster) of cobertures
        with the same dimensions of the working raster and then converted it to a pandas Series"""
        if not isinstance(coberture_file, gpd.GeoDataFrame):
            raise InvalidTypeOfGeom("coberture_file param MUST BE a geopandas.GeoDataFrame object")

        shapes = ((geom, value) for geom, value in zip(coberture_file.geometry,
                                                       coberture_file[coberture_column]))
        burned = features.rasterize(shapes=shapes, fill=fillna,
                                    out_shape=(self.meta["height"], self.meta["width"]),
                                    transform=self.meta["transform"])
        burned_to_series = pd.Series(burned.flatten().tolist())
        self.__dataframe[GeneralSensorEnums.coberture.value] = burned_to_series

    def select_df_of_cobertures(self, fill_na: int = -9999):
        """This method its the first version to deal with the image and the cobertur file
        THIS its the most important feature here, cause this is how you will play with sklearn."""
        # TODO think if the fill_na on this methon an on coberture_to_raster should be a instance variable
        #  or perphaps a class variable
        if GeneralSensorEnums.coberture.value not in self.__dataframe.columns.tolist():
            raise NoCobertureSeries("There is no coberture series in the dataframe property, "
                                    "did u use coberture_to_raster method?")
        return self.__dataframe[self.__dataframe[GeneralSensorEnums.coberture.value] != fill_na]

