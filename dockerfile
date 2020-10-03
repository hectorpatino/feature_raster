FROM osgeo/gdal

COPY requirements-dev.txt .

RUN apt-get update &&\
    apt install -y python3-pip &&\
    pip3 install -r requirements-dev.txt

COPY data/ data/
COPY feature_raster/ feature_raster/
COPY tests/ tests/
COPY APP.py .

CMD ["python", "APP.py"]
