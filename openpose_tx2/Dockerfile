FROM w251/cuda:dev-tx2-4.3_b132
RUN apt-get update
RUN apt-get install -y software-properties-common

RUN apt-get update && apt-get install -y \
    build-essential\
    curl \
    git \
    python3.6 \
    python3-pip \
    python3-numpy \
    python3-setuptools \
    vim \
    unzip\
    sudo \
    wget \
    unzip \
    pkg-config 

RUN JETSON_BOARD="TX2" \
	JETSON_JETPACK="4.3" \
	JETSON_CUDA="10.0.326" \
	JETSON_L4T="32.3.1" \
	export JETSON_BOARD \
	export JETSON_CUDA \
	export JETSON_JETPACK \
	export JETSON_L4T

#make sure cmake is >3.12
RUN apt-get install -y qtbase5-dev libssl-dev

#Install cmake
RUN wget https://cmake.org/files/v3.17/cmake-3.17.0.tar.gz  \
    && tar xzf cmake-3.17.0.tar.gz  \
    && rm -rf cmake-3.17.0.tar.gz  \
    && cd cmake-3.17.0  \
    && ./configure --qt-gui \
    && ./bootstrap \
    && make -j`nproc` \
    && make install -j`nproc` 
    

#Install system packages required by TensorFlow:
RUN apt-get install -y python3-dev libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

RUN pip3 install -U pip testresources setuptools wrapt google-pasta termcolor astor requests

RUN pip3 install -U numpy grpcio future mock absl-py py-cpuinfo psutil portpicker six h5py keras_preprocessing keras_applications gast futures protobuf pybind11

#Install tensorflow
#RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow==1.15.2

#Install Efficient net
#RUN pip3 install cython
#RUN pip3 install keras_efficientnets

#Install OpenCV
RUN apt-get update && apt-get install -y \
	gfortran \
        libatlas-base-dev \
        libavcodec-dev \
        libavformat-dev \
        libavresample-dev \
        libcanberra-gtk3-module \
        libdc1394-22-dev \
        libeigen3-dev \
        libglew-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-good1.0-dev \
        libgstreamer1.0-dev \
        libgtk-3-dev \
        libjpeg-dev \
        libjpeg8-dev \
        libjpeg-turbo8-dev \
        liblapack-dev \
        liblapacke-dev \
        libopenblas-dev \
        libpng-dev \
        libpostproc-dev \
        libswscale-dev \
        libtbb-dev \
        libtbb2 \
        libtesseract-dev \
        libtiff-dev \
        libv4l-dev \
        libxine2-dev \
        libxvidcore-dev \
        libx264-dev \
        qv4l2 \
        v4l-utils \
        v4l2ucp \
        zlib1g-dev

#OpenCV 4.3
RUN cd \
    && git clone https://github.com/opencv/opencv.git \
    && git clone https://github.com/opencv/opencv_contrib.git \
    && cd opencv \
    && mkdir build \
    && cd build \
    && cmake -D BUILD_EXAMPLES=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=ON -D CMAKE_BUILD_TYPE=RELEASE -D CUDA_ARCH_BIN=6.2 \
    	 	-D WITH_CUDA=ON -D WITH_CUBLAS=ON \
        	-D WITH_GSTREAMER=ON \
        	-D WITH_LIBV4L=ON \
        	-D CUDA_FAST_MATH=ON \
		-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        	-D WITH_OPENGL=ON .. \
    && make -j8 \
    && make install \
    && cd

#Install Open pose dependencies
RUN cd \   
	&& git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose \
	&& cd openpose \
        && chmod +x scripts/ubuntu/install_deps.sh \ 
	&& ./scripts/ubuntu/install_deps.sh 


#Install Openpose
RUN cd \
        && cd openpose \
        && sed -i "s|CV_LOAD_IMAGE_GRAYSCALE|cv::IMREAD_GRAYSCALE |g" examples/tests/resizeTest.cpp \
        && mkdir build \
        && cd build \
        && cmake -D BUILD_EXAMPLES=ON  -D BUILD_PYTHON=ON  -D USE_OPENCV=ON -D WITH_OPENCV_WITH_OPENGL=ON .. \
        && make -j4 \
        && make install \
#==== python build ==== 
# don't do make install command. because it installs openpose module to python 2.7 directory
        && cd python \
        && make -j4

ARG URL=http://169.44.201.108:7002/jetpacks/4.3

RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 apt-utils bzip2 curl sudo unp && apt-get clean && rm -rf /var/cache/apt
WORKDIR /tmp

# Putting all of the downloads, installs, updates, and cleanup on the same line
# to reduce the image size

RUN curl $URL/cuda-repo-l4t-10-0-local-10.0.326_1.0-1_arm64.deb -so cuda-repo-l4t_arm64.deb && \
    curl $URL/libcudnn7_7.6.3.28-1+cuda10.0_arm64.deb -so libcudnn_arm64.deb && \
    curl $URL/libcudnn7-dev_7.6.3.28-1+cuda10.0_arm64.deb -so libcudnn-dev_arm64.deb && \
    dpkg -i /tmp/cuda-repo-l4t_arm64.deb && \
    apt-key add /var/cuda-repo-10-0-local-10.0.326/7fa2af80.pub && \
    apt-get update && apt-get install -y cuda-toolkit-10.0 && \
    dpkg -i /tmp/libcudnn_arm64.deb && \
    dpkg -i /tmp/libcudnn-dev_arm64.deb && \
    apt-get -y autoremove && apt-get -y autoclean && \
    rm -rf /var/cache/apt && \
    rm -f /tmp/*.deb && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra
ENV PATH=/usr/local/cuda-10.0/bin:$PATH

#Install tensorflow
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow==1.15.2

#Install Efficient net
RUN pip3 install cython
RUN pip3 install keras_efficientnets
#Install Cudnn 7.6.3


#RUN sudo apt-get install --no-install-recommends libcudnn7=7.6.3.30-1+cuda10.0 libcudnn7-dev=7.6.3.30-1+cuda10.0
RUN apt-get autoremove

RUN pip3 install matplotlib
#TensorRT
RUN curl $URL/libnvinfer6_6.0.1-1+cuda10.0_arm64.deb -so /tmp/libnvinfer6_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/libnvinfer-dev_6.0.1-1+cuda10.0_arm64.deb -so /tmp/libnvinfer-dev_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/libnvinfer-samples_6.0.1-1+cuda10.0_all.deb -so /tmp/libnvinfer-samples_6.0.1-1+cuda10.0_all.deb
RUN curl $URL/tensorrt_6.0.1.10-1+cuda10.0_arm64.deb -so /tmp/tensorrt_6.0.1.10-1+cuda10.0_arm64.deb
RUN curl $URL/python3-libnvinfer_6.0.1-1+cuda10.0_arm64.deb -so /tmp/python3-libnvinfer_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/python3-libnvinfer-dev_6.0.1-1+cuda10.0_arm64.deb -so /tmp/python3-libnvinfer-dev_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/libnvinfer-plugin6_6.0.1-1+cuda10.0_arm64.deb -so /tmp/libnvinfer-plugin6_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/libnvinfer-plugin-dev_6.0.1-1+cuda10.0_arm64.deb -so /tmp/libnvinfer-plugin-dev_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/libnvparsers6_6.0.1-1+cuda10.0_arm64.deb -so /tmp/libnvparsers6_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/libnvparsers-dev_6.0.1-1+cuda10.0_arm64.deb -so /tmp/libnvparsers-dev_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/libnvonnxparsers6_6.0.1-1+cuda10.0_arm64.deb -so /tmp/libnvonnxparsers6_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/libnvonnxparsers-dev_6.0.1-1+cuda10.0_arm64.deb -so /tmp/libnvonnxparsers-dev_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/libnvinfer-doc_6.0.1-1+cuda10.0_all.deb -so /tmp/libnvinfer-doc_6.0.1-1+cuda10.0_all.deb
RUN curl $URL/libnvinfer-bin_6.0.1-1+cuda10.0_arm64.deb -so /tmp/libnvinfer-bin_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/graphsurgeon-tf_6.0.1-1+cuda10.0_arm64.deb -so /tmp/graphsurgeon-tf_6.0.1-1+cuda10.0_arm64.deb
RUN curl $URL/uff-converter-tf_6.0.1-1+cuda10.0_arm64.deb -so /tmp/uff-converter-tf_6.0.1-1+cuda10.0_arm64.deb

RUN apt install -y /tmp/*.deb
RUN apt install -y tensorrt python3-numpy

RUN rm /tmp/*.deb

# Clean up
RUN rm -fr /tmp/* /var/cache/apt/* && apt-get clean
RUN pip3 install pycuda
#clean
RUN apt-get clean
