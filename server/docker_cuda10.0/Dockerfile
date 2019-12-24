FROM nvidia/cuda:10.0-base-ubuntu16.04

# RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
# RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#  && chmod +x /miniconda.sh \
#  && /miniconda.sh -b -p /miniconda \
#  && rm /miniconda.sh

# ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.6 environment
# RUN /miniconda/bin/conda install -y conda-build \
#  && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
#  && /miniconda/bin/conda clean -ya

# ENV CONDA_DEFAULT_ENV=py36
# ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
# ENV PATH=$CONDA_PREFIX/bin:$PATH
# ENV CONDA_AUTO_UPDATE_CONDA=false

# RUN conda install -y ipython
# RUN pip install requests ninja yacs cython matplotlib opencv-python tqdm

# Install PyTorch 1.0 Nightly
# ARG CUDA
# RUN conda install pytorch-nightly cudatoolkit=${CUDA} -c pytorch \
#  && conda clean -ya

## Install TorchVision master
# RUN git clone https://github.com/pytorch/vision.git \
#  && cd vision \
#  && python setup.py install
# RUN pip install torchvision==0.2.2

# install pycocotools
# RUN git clone https://github.com/cocodataset/cocoapi.git \
#  && cd cocoapi/PythonAPI \
#  && python setup.py build_ext install

# install apex
# RUN git clone https://github.com/NVIDIA/apex.git \
#  && cd apex \
#  && python setup.py install --cuda_ext --cpp_ext

# install squashfuse and tmux
# RUN apt-get install -y squashfuse tmux

# install vscode debugger (ptvsd)
# RUN pip install ptvsd

# install Jupyter
# RUN conda install jupyter \
#  && conda clean -ya

# Set up the timezone
# RUN apt install -y tzdata
# ENV TZ=America/Los_Angeles
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install fs.sshfs
# RUN pip install fs.sshfs

# Install wget and nano.
# RUN apt install -y wget nano

WORKDIR /