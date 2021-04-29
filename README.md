# Object Detection

Object detection with Tensorflow 2 Object Detection Api

## Установка

### Сборка OpenCV из исходников
1) Обновление пакетов:
```bash
sudo apt update
```
2) Установка зависимостей:
```bash
sudo apt install build-essential cmake git \
pkg-config libgtk-3-dev libavcodec-dev libavformat-dev \
libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
libjpeg-dev libpng-dev libtiff-dev gfortran openexr \
libatlas-base-dev python3-dev python3-numpy libtbb2 \
libtbb-dev libdc1394-22-dev libopenexr-dev \
libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
```
3) Создание директории в которой будут сохранены репозитории OpenCV:
```bash
mkdir ~/opencv_build

cd ~/opencv_build
```
4) Скачивание репозиториев opencv и opencv_contrib:
```bash
git clone https://github.com/opencv/opencv.git

git clone https://github.com/opencv/opencv_contrib.git
```
5) Создание временной директории для сборки:
```bash
cd ~/opencv_build/opencv

mkdir build

cd build
```
6) Подготовим сборку OpenCV, используя CMake:
```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
-D BUILD_EXAMPLES=ON ..
```
7) Компиляция OpenCV:

Флаг j станавливается в зависимости от количетсва ядер процессора
```bash
make -j2
```
8) Установка OpenCV:
```bash
sudo make install
```
9) Проверка установки
```bash
pkg-config --modversion opencv4
```
### Установка Object Detection Api
1) Установка Tensorflow:
```bash
sudo apt install python3-pip

pip3 install tensorflow
```
2) Клонирование репозитория TensorFlow Models:
```bash
git clone https://github.com/tensorflow/models.git
```
3) Установка Protoc / Protobuf
```bash
apt install -y protobuf-compiler
```
4) Установка Object Detection Api
```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python3 -m pip install .
```
Проверка установки:
```bash
# From models/research/
python3 object_detection/builders/model_builder_tf2_test.py
```
### Запуск модели детектора объектов:

```bash
python3 image_detection.py

python3 video_detection.py --video ./data/video/terrace1.mp4 --output ./data/otput/output_video.avi
```