FROM osgeo/gdal

WORKDIR /code

COPY requirements-dev.txt .

RUN apt-get update &&\
    apt install -y python3-pip &&\
    pip3 install -r requirements-dev.txt

COPY data feature_raster/. tests/. APP.py

CMD ["python", "app.py"]
