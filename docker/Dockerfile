FROM nvidia/cuda:11.1-devel-ubuntu20.04

ENV CONDA_DIR /anaconda3
COPY third_party/softras /workspace/softras
COPY lasr.yml /workspace/lasr.yml

# system dependencies
RUN apt-get update -q
RUN DEBIAN_FRONTEND="noninteractive" TZ=America/New_York apt-get -y install tzdata
RUN apt-get install -q -y freeglut3-dev wget zip ffmpeg

# install anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh; \
    chmod +x ./Anaconda3-2020.11-Linux-x86_64.sh; \
    ./Anaconda3-2020.11-Linux-x86_64.sh -b -p $CONDA_DIR; \
    rm ./Anaconda3-2020.11-Linux-x86_64.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# setup conda env 
RUN cd /workspace
RUN conda env create -f /workspace/lasr.yml
RUN . /anaconda3/etc/profile.d/conda.sh; conda activate lasr; cd /workspace/softras; python setup.py install
RUN apt-get install -y -q git
RUN apt-get install -y -q cmake
RUN git clone --recursive -j8 git://github.com/hjwdzh/Manifold; cd Manifold; mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release;make
