# ==============================================================================
# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================
ARG dldt=dldt-internal
ARG gst=gst-internal

FROM ubuntu:18.04 AS base
WORKDIR /home

# COMMON BUILD TOOLS
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y -q --no-install-recommends \
        cmake \
        build-essential \
        automake \
        autoconf \
        libtool \
        make \
        git \
        wget \
        pciutils \
        cpio \
        libtool \
        lsb-release \
        ca-certificates \
        pkg-config \
        bison \
        flex \
        libcurl4-gnutls-dev \
        zlib1g-dev \
        nasm \
        yasm \
        libx11-dev \
        xorg-dev \
        libgl1-mesa-dev \
        openbox \
        python3 \
        python3-pip \
        python3-setuptools && \
    rm -rf /var/lib/apt/lists/*

# Build x264
ARG X264_VER=stable
ARG X264_REPO=https://github.com/mirror/x264
RUN  git clone ${X264_REPO} && \
    cd x264 && \
    git checkout ${X264_VER} && \
    ./configure --prefix="/usr" --libdir=/usr/lib/x86_64-linux-gnu --enable-shared && \
    make -j $(nproc) && \
    make install DESTDIR="/home/build" && \
    make install

# Build Intel(R) Media SDK
ARG MSDK_REPO=https://github.com/Intel-Media-SDK/MediaSDK/releases/download/intel-mediasdk-19.3.1/MediaStack.tar.gz
RUN wget -O - ${MSDK_REPO} | tar xz && \
    cd MediaStack && \
    \
    cp -r opt/ /home/build && \
    cp -r etc/ /home/build && \
    \
    cp -a opt/. /opt/ && \
    cp -a etc/. /opt/ && \
    ldconfig

ENV PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/opt/intel/mediasdk/lib64/pkgconfig
ENV LIBRARY_PATH=/opt/intel/mediasdk/lib64:/usr/lib:${LIBRARY_PATH}
ENV LIBVA_DRIVERS_PATH=/opt/intel/mediasdk/lib64
ENV LIBVA_DRIVER_NAME=iHD
ENV GST_VAAPI_ALL_DRIVERS=1

# clinfo needs to be installed after build directory is copied over
RUN mkdir neo && cd neo && \
    wget https://github.com/intel/compute-runtime/releases/download/19.31.13700/intel-gmmlib_19.2.3_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.31.13700/intel-igc-core_1.0.10-2364_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.31.13700/intel-igc-opencl_1.0.10-2364_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.31.13700/intel-opencl_19.31.13700_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.31.13700/intel-ocloc_19.31.13700_amd64.deb && \
    dpkg -i *.deb && \
    dpkg-deb -x intel-gmmlib_19.2.3_amd64.deb /home/build/ && \
    dpkg-deb -x intel-igc-core_1.0.10-2364_amd64.deb /home/build/ && \
    dpkg-deb -x intel-igc-opencl_1.0.10-2364_amd64.deb /home/build/ && \
    dpkg-deb -x intel-opencl_19.31.13700_amd64.deb /home/build/ && \
    dpkg-deb -x intel-ocloc_19.31.13700_amd64.deb /home/build/ && \
    cp -a /home/build/. /


FROM base AS gst-internal
WORKDIR /home

# GStreamer core
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install --no-install-recommends -q -y \
        libglib2.0-dev \
        libgmp-dev \
        libgsl-dev \
        gobject-introspection \
        libcap-dev \
        libcap2-bin \
        gettext \
        libgirepository1.0-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir meson ninja

ARG PACKAGE_ORIGIN="Intel (R) GStreamer Distribution"

ARG PREFIX=/usr
ARG LIBDIR=/usr/lib/x86_64-linux-gnu
ARG LIBEXECDIR=/usr/lib/x86_64-linux-gnu

ARG GST_VERSION=1.16.2
ARG BUILD_TYPE=release

ENV GSTREAMER_LIB_DIR=${LIBDIR}
ENV LIBRARY_PATH=${GSTREAMER_LIB_DIR}:${GSTREAMER_LIB_DIR}/gstreamer-1.0:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${LIBRARY_PATH}
ENV PKG_CONFIG_PATH=${GSTREAMER_LIB_DIR}/pkgconfig

RUN mkdir -p build/src

ARG GST_REPO=https://gstreamer.freedesktop.org/src/gstreamer/gstreamer-${GST_VERSION}.tar.xz
RUN wget ${GST_REPO} -O build/src/gstreamer-${GST_VERSION}.tar.xz
RUN tar xvf build/src/gstreamer-${GST_VERSION}.tar.xz && \
    cd gstreamer-${GST_VERSION} && \
    PKG_CONFIG_PATH=$PWD/build/pkgconfig meson \
        -Dexamples=disabled \
        -Dtests=disabled \
        -Dbenchmarks=disabled \
        -Dgtk_doc=disabled \
        -Dpackage-origin="${PACKAGE_ORIGIN}" \
        --buildtype=${BUILD_TYPE} \
        --prefix=${PREFIX} \
        --libdir=${LIBDIR} \
        --libexecdir=${LIBEXECDIR} \
    build/ && \
    ninja -C build && \
    DESTDIR=/home/build meson install -C build/ && \
    meson install -C build/

# ORC Acceleration
ARG GST_ORC_VERSION=0.4.31
ARG GST_ORC_REPO=https://gstreamer.freedesktop.org/src/orc/orc-${GST_ORC_VERSION}.tar.xz
RUN wget ${GST_ORC_REPO} -O build/src/orc-${GST_ORC_VERSION}.tar.xz
RUN tar xvf build/src/orc-${GST_ORC_VERSION}.tar.xz && \
    cd orc-${GST_ORC_VERSION} && \
    meson \
        -Dexamples=disabled \
        -Dtests=disabled \
        -Dbenchmarks=disabled \
        -Dgtk_doc=disabled \
        -Dorc-test=disabled \
        --prefix=${PREFIX} \
        --libdir=${LIBDIR} \
        --libexecdir=${LIBEXECDIR} \
    build/ && \
    ninja -C build && \
    DESTDIR=/home/build meson install -C build/ && \
    meson install -C build/


# GStreamer Base plugins
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y -q --no-install-recommends \
        libx11-dev \
        iso-codes \
        libegl1-mesa-dev \
        libgles2-mesa-dev \
        libgl-dev \
        gudev-1.0 \
        libtheora-dev \
        libcdparanoia-dev \
        libpango1.0-dev \
        libgbm-dev \
        libasound2-dev \
        libjpeg-dev \
        libvisual-0.4-dev \
        libxv-dev \
        libopus-dev \
        libgraphene-1.0-dev \
        libvorbis-dev && \
    rm -rf /var/lib/apt/lists/*


# Build the gstreamer plugin base
ARG GST_PLUGIN_BASE_REPO=https://gstreamer.freedesktop.org/src/gst-plugins-base/gst-plugins-base-${GST_VERSION}.tar.xz
RUN wget ${GST_PLUGIN_BASE_REPO} -O build/src/gst-plugins-base-${GST_VERSION}.tar.xz
RUN tar xvf build/src/gst-plugins-base-${GST_VERSION}.tar.xz && \
    cd gst-plugins-base-${GST_VERSION} && \
    PKG_CONFIG_PATH=$PWD/build/pkgconfig:${PKG_CONFIG_PATH} meson \
        -Dexamples=disabled \
        -Dtests=disabled \
        -Dgtk_doc=disabled \
        -Dnls=disabled \
        -Dgl=disabled \
        -Dpackage-origin="${PACKAGE_ORIGIN}" \
        --buildtype=${BUILD_TYPE} \
        --prefix=${PREFIX} \
        --libdir=${LIBDIR} \
        --libexecdir=${LIBEXECDIR} \
    build/ && \
    ninja -C build && \
    DESTDIR=/home/build meson install -C build/ && \
    meson install -C build/


# GStreamer Good plugins
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y -q --no-install-recommends \
        libbz2-dev \
        libv4l-dev \
        libaa1-dev \
        libflac-dev \
        libgdk-pixbuf2.0-dev \
        libmp3lame-dev \
        libcaca-dev \
        libdv4-dev \
        libmpg123-dev \
        libraw1394-dev \
        libavc1394-dev \
        libiec61883-dev \
        libpulse-dev \
        libsoup2.4-dev \
        libspeex-dev \
        libtag-extras-dev \
        libtwolame-dev \
        libwavpack-dev && \
    rm -rf /var/lib/apt/lists/*

ARG GST_PLUGIN_GOOD_REPO=https://gstreamer.freedesktop.org/src/gst-plugins-good/gst-plugins-good-${GST_VERSION}.tar.xz

# Lines below extract patch needed for Smart City Sample (OVS use case). Patch is applied before building gst-plugins-good
RUN  mkdir gst-plugins-good-${GST_VERSION} && \
    git clone https://github.com/GStreamer/gst-plugins-good.git && \
    cd gst-plugins-good && \
    git diff 080eba64de68161026f2b451033d6b455cb92a05 37d22186ffb29a830e8aad2e4d6456484e716fe7 > ../gst-plugins-good-${GST_VERSION}/rtpjitterbuffer-fix.patch

RUN wget ${GST_PLUGIN_GOOD_REPO} -O build/src/gst-plugins-good-${GST_VERSION}.tar.xz
RUN tar xvf build/src/gst-plugins-good-${GST_VERSION}.tar.xz && \
    cd gst-plugins-good-${GST_VERSION}  && \
    patch -p1 < rtpjitterbuffer-fix.patch && \
    PKG_CONFIG_PATH=$PWD/build/pkgconfig:${PKG_CONFIG_PATH} meson \
        -Dexamples=disabled \
        -Dtests=disabled \
        -Dgtk_doc=disabled \
        -Dnls=disabled \
        -Dpackage-origin="${PACKAGE_ORIGIN}" \
        --buildtype=${BUILD_TYPE} \
        --prefix=${PREFIX} \
        --libdir=${LIBDIR} \
        --libexecdir=${LIBEXECDIR} \
    build/ && \
    ninja -C build && \
    DESTDIR=/home/build meson install -C build/ && \
    meson install -C build/


# GStreamer Bad plugins
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y -q --no-install-recommends \
        libbluetooth-dev \
        libusb-1.0.0-dev \
        libass-dev \
        libbs2b-dev \
        libchromaprint-dev \
        liblcms2-dev \
        libssh2-1-dev \
        libdc1394-22-dev \
        libdirectfb-dev \
        libssh-dev \
        libdca-dev \
        libfaac-dev \
        libfdk-aac-dev \
        flite1-dev \
        libfluidsynth-dev \
        libgme-dev \
        libgsm1-dev \
        nettle-dev \
        libkate-dev \
        liblrdf0-dev \
        libde265-dev \
        libmjpegtools-dev \
        libmms-dev \
        libmodplug-dev \
        libmpcdec-dev \
        libneon27-dev \
        libopenal-dev \
        libopenexr-dev \
        libopenjp2-7-dev \
        libopenmpt-dev \
        libopenni2-dev \
        libdvdnav-dev \
        librtmp-dev \
        librsvg2-dev \
        libsbc-dev \
        libsndfile1-dev \
        libsoundtouch-dev \
        libspandsp-dev \
        libsrtp2-dev \
        libzvbi-dev \
        libvo-aacenc-dev \
        libvo-amrwbenc-dev \
        libwebrtc-audio-processing-dev \
        libwebp-dev \
        libwildmidi-dev \
        libzbar-dev \
        libnice-dev \
        libxkbcommon-dev && \
    rm -rf /var/lib/apt/lists/*

# Uninstalled dependencies: opencv, opencv4, libmfx(waiting intelMSDK), wayland(low version), vdpau

ARG GST_PLUGIN_BAD_REPO=https://gstreamer.freedesktop.org/src/gst-plugins-bad/gst-plugins-bad-${GST_VERSION}.tar.xz
RUN wget ${GST_PLUGIN_BAD_REPO} -O build/src/gst-plugins-bad-${GST_VERSION}.tar.xz
RUN tar xvf build/src/gst-plugins-bad-${GST_VERSION}.tar.xz && \
    cd gst-plugins-bad-${GST_VERSION} && \
    PKG_CONFIG_PATH=$PWD/build/pkgconfig:${PKG_CONFIG_PATH} meson \
        -Dexamples=disabled \
        -Dtests=disabled \
        -Ddoc=disabled \
        -Dnls=disabled \
        -Dx265=disabled \
        -Dyadif=disabled \
        -Dresindvd=disabled \
        -Dmplex=disabled \
        -Ddts=disabled \
        -Dofa=disabled \
        -Dfaad=disabled \
        -Dmpeg2enc=disabled \
        -Dpackage-origin="${PACKAGE_ORIGIN}" \
        --buildtype=${BUILD_TYPE} \
        --prefix=${PREFIX} \
        --libdir=${LIBDIR} \
        --libexecdir=${LIBEXECDIR} \
    build/ && \
    ninja -C build && \
    DESTDIR=/home/build meson install -C build/ && \
    meson install -C build/

# Build the gstreamer plugin ugly set
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y -q --no-install-recommends \
        libmpeg2-4-dev \
        libopencore-amrnb-dev \
        libopencore-amrwb-dev \
        liba52-0.7.4-dev \
    && rm -rf /var/lib/apt/lists/*

ARG GST_PLUGIN_UGLY_REPO=https://gstreamer.freedesktop.org/src/gst-plugins-ugly/gst-plugins-ugly-${GST_VERSION}.tar.xz

RUN wget ${GST_PLUGIN_UGLY_REPO} -O build/src/gst-plugins-ugly-${GST_VERSION}.tar.xz
RUN tar xvf build/src/gst-plugins-ugly-${GST_VERSION}.tar.xz && \
    cd gst-plugins-ugly-${GST_VERSION}  && \
    PKG_CONFIG_PATH=$PWD/build/pkgconfig:${PKG_CONFIG_PATH} meson \
        -Dexamples=disabled \
        -Dtests=disabled \
        -Dgtk_doc=disabled \
        -Dnls=disabled \
        -Dcdio=disabled \
        -Dsid=disabled \
        -Dmpeg2dec=disabled \
        -Ddvdread=disabled \
        -Da52dec=disabled \
        -Dx264=disabled \
        -Dpackage-origin="${PACKAGE_ORIGIN}" \
        --buildtype=${BUILD_TYPE} \
        --prefix=${PREFIX} \
        --libdir=${LIBDIR} \
        --libexecdir=${LIBEXECDIR} \
    build/ && \
    ninja -C build && \
    DESTDIR=/home/build meson install -C build/ && \
    meson install -C build/


# FFmpeg
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y -q --no-install-recommends \
        bzip2 \
        autoconf

RUN mkdir ffmpeg_sources && cd ffmpeg_sources && \
    wget -O - https://www.nasm.us/pub/nasm/releasebuilds/2.14.02/nasm-2.14.02.tar.bz2 | tar xj && \
    cd nasm-2.14.02 && \
    ./autogen.sh && \
    ./configure --prefix=${PREFIX} --bindir="${PREFIX}/bin" && \
    make && make install

RUN wget https://ffmpeg.org/releases/ffmpeg-4.2.2.tar.gz -O build/src/ffmpeg-4.2.2.tar.gz
RUN cd ffmpeg_sources && \
    tar xvf /home/build/src/ffmpeg-4.2.2.tar.gz && \
    cd ffmpeg-4.2.2 && \
    PATH="${PREFIX}/bin:$PATH" PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig" \
    ./configure \
        --disable-gpl \
        --enable-pic \
        --disable-shared \
        --enable-static \
        --prefix=${PREFIX} \
        --extra-cflags="-I${PREFIX}/include" \
        --extra-ldflags="-L${PREFIX}/lib" \
        --extra-libs=-lpthread \
        --extra-libs=-lm \
        --bindir="${PREFIX}/bin" \
        --disable-vaapi && \
    make -j $(nproc) && \
    make install

# Build gst-libav
ARG GST_PLUGIN_LIBAV_REPO=https://gstreamer.freedesktop.org/src/gst-libav/gst-libav-${GST_VERSION}.tar.xz
RUN wget ${GST_PLUGIN_LIBAV_REPO} -O build/src/gst-libav-${GST_VERSION}.tar.xz
RUN tar xvf build/src/gst-libav-${GST_VERSION}.tar.xz && \
    cd gst-libav-${GST_VERSION} && \
    PKG_CONFIG_PATH=$PWD/build/pkgconfig:${PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH} meson \
        -Dexamples=disabled \
        -Dtests=disabled \
        -Dgtk_doc=disabled \
        -Dnls=disabled \
        -Dpackage-origin="${PACKAGE_ORIGIN}" \
        --buildtype=${BUILD_TYPE} \
        --prefix=${PREFIX} \
        --libdir=${LIBDIR} \
        --libexecdir=${LIBEXECDIR} \
    build/ && \
    ninja -C build && \
    DESTDIR=/home/build meson install -C build/ && \
    meson install -C build/

ENV PKG_CONFIG_PATH=/opt/intel/mediasdk/lib64/pkgconfig:${PKG_CONFIG_PATH}
ENV LIBRARY_PATH=/opt/intel/mediasdk/lib64:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/opt/intel/mediasdk/lib64:${LD_LIBRARY_PATH}
ENV LIBVA_DRIVERS_PATH=/opt/intel/mediasdk/lib64
ENV LIBVA_DRIVER_NAME=iHD

# Build gstreamer plugin vaapi
ARG GST_PLUGIN_VAAPI_REPO=https://gstreamer.freedesktop.org/src/gstreamer-vaapi/gstreamer-vaapi-${GST_VERSION}.tar.xz

ENV GST_VAAPI_ALL_DRIVERS=1

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y -q --no-install-recommends \
        libva-dev \
        libxrandr-dev \
        libudev-dev && \
    rm -rf /var/lib/apt/lists/*

ARG GSTREAMER_VAAPI_PATCH_URL=https://raw.githubusercontent.com/opencv/gst-video-analytics/master/patches/gstreamer-vaapi/vasurface_qdata.patch
RUN wget ${GST_PLUGIN_VAAPI_REPO} -O build/src/gstreamer-vaapi-${GST_VERSION}.tar.xz
RUN tar xvf build/src/gstreamer-vaapi-${GST_VERSION}.tar.xz && \
    cd gstreamer-vaapi-${GST_VERSION} && \
    wget -O - ${GSTREAMER_VAAPI_PATCH_URL} | git apply && \
    PKG_CONFIG_PATH=$PWD/build/pkgconfig:${PKG_CONFIG_PATH} meson \
        -Dexamples=disabled \
        -Dtests=disabled \
        -Dgtk_doc=disabled \
        -Dnls=disabled \
        -Dpackage-origin="${PACKAGE_ORIGIN}" \
        --buildtype=${BUILD_TYPE} \
        --prefix=${PREFIX} \
        --libdir=${LIBDIR} \
        --libexecdir=${LIBEXECDIR} \
    build/ && \
    ninja -C build && \
    DESTDIR=/home/build meson install -C build/ && \
    meson install -C build/

# gst-python
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install --no-install-recommends -y \
        python-gi-dev \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

ARG GST_PYTHON_REPO=https://gstreamer.freedesktop.org/src/gst-python/gst-python-${GST_VERSION}.tar.xz
RUN wget ${GST_PYTHON_REPO} -O build/src/gst-python-${GST_VERSION}.tar.xz
RUN tar xvf build/src/gst-python-${GST_VERSION}.tar.xz && \
    cd gst-python-${GST_VERSION} && \
    PKG_CONFIG_PATH=$PWD/build/pkgconfig:${PKG_CONFIG_PATH} meson \
        -Dpython=python3 \
        --buildtype=${BUILD_TYPE} \
        --prefix=${PREFIX} \
        --libdir=${LIBDIR} \
        --libexecdir=${LIBEXECDIR} \
    build/ && \
    ninja -C build && \
    DESTDIR=/home/build meson install -C build/ && \
    meson install -C build/

ENV GI_TYPELIB_PATH=${LIBDIR}/girepository-1.0

ENV PYTHONPATH=${PREFIX}/lib/python3.6/site-packages:${PYTHONPATH}

ARG ENABLE_PAHO_INSTALLATION=false
ARG PAHO_VER=1.3.0
ARG PAHO_REPO=https://github.com/eclipse/paho.mqtt.c/archive/v${PAHO_VER}.tar.gz
RUN if [ "$ENABLE_PAHO_INSTALLATION" = "true" ] ; then \
    wget -O - ${PAHO_REPO} | tar -xz && \
    cd paho.mqtt.c-${PAHO_VER} && \
    make && \
    make install && \
    cp build/output/libpaho-mqtt3c.so.1.0 /home/build/usr/lib/x86_64-linux-gnu/ && \
    cp build/output/libpaho-mqtt3cs.so.1.0 /home/build/usr/lib/x86_64-linux-gnu/ && \
    cp build/output/libpaho-mqtt3a.so.1.0 /home/build/usr/lib/x86_64-linux-gnu/ && \
    cp build/output/libpaho-mqtt3as.so.1.0 /home/build/usr/lib/x86_64-linux-gnu/ && \
    cp build/output/paho_c_version /home/build/usr/bin/ && \
    cp build/output/samples/paho_c_pub /home/build/usr/bin/ && \
    cp build/output/samples/paho_c_sub /home/build/usr/bin/ && \
    cp build/output/samples/paho_cs_pub /home/build/usr/bin/ && \
    cp build/output/samples/paho_cs_sub /home/build/usr/bin/ && \
    chmod 644 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3c.so.1.0 && \
    chmod 644 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3cs.so.1.0 && \
    chmod 644 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3a.so.1.0 && \
    chmod 644 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3as.so.1.0 && \
    ln /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3c.so.1.0 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3c.so.1 && \
    ln /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3cs.so.1.0 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3cs.so.1 && \
    ln /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3a.so.1.0 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3a.so.1 && \
    ln /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3as.so.1.0 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3as.so.1 && \
    ln /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3c.so.1 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3c.so && \
    ln /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3cs.so.1 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3cs.so && \
    ln /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3a.so.1 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3a.so && \
    ln /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3as.so.1 /home/build/usr/lib/x86_64-linux-gnu/libpaho-mqtt3as.so && \
    cp src/MQTTAsync.h /home/build/usr/include/ && \
    cp src/MQTTClient.h /home/build/usr/include/ && \
    cp src/MQTTClientPersistence.h /home/build/usr/include/ && \
    cp src/MQTTProperties.h /home/build/usr/include/ && \
    cp src/MQTTReasonCodes.h /home/build/usr/include/ && \
    cp src/MQTTSubscribeOpts.h /home/build/usr/include/; \
    else \
    echo "PAHO install disabled"; \
    fi

ARG ENABLE_RDKAFKA_INSTALLATION=false
ARG RDKAFKA_VER=1.0.0
ARG RDKAFKA_REPO=https://github.com/edenhill/librdkafka/archive/v${RDKAFKA_VER}.tar.gz
RUN if [ "$ENABLE_RDKAFKA_INSTALLATION" = "true" ] ; then \
    wget -O - ${RDKAFKA_REPO} | tar -xz && \
    cd librdkafka-${RDKAFKA_VER} && \
    ./configure \
        --prefix=${PREFIX} \
        --libdir=${LIBDIR} && \
    make && \
    make install && \
    make install DESTDIR=/home/build; \
    else \
    echo "KAFKA install disabled"; \
    fi


FROM base AS dldt-internal
WORKDIR /home

ARG CMAKE_VERSION=3.17.1
RUN wget https://github.com/Kitware/CMake/archive/v${CMAKE_VERSION}.tar.gz -O - | tar xzf - && \
    cd CMake-${CMAKE_VERSION} && \
    ./bootstrap --system-curl --parallel=$(nproc) && \
    make -j $(nproc) && \
    make install

ARG DLDT_VER=2020.2
ARG DLDT_REPO=https://github.com/opencv/dldt.git

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y -q --no-install-recommends \
        libusb-1.0-0-dev \
        libboost-all-dev \
        libgtk2.0-dev \
        python-yaml

RUN git clone -b ${DLDT_VER} ${DLDT_REPO} && \
    cd dldt && \
    git submodule init && \
    git submodule update --recursive && \
    mkdir build && \
    cd build && \
    cmake -DENABLE_VALIDATION_SET=OFF -DCMAKE_INSTALL_PREFIX=/opt/intel/dldt -DLIB_INSTALL_PATH=/opt/intel/dldt -DENABLE_MKL_DNN=ON -DENABLE_CLDNN=ON -DENABLE_SAMPLE_CORE=OFF  .. && \
    make -j $(nproc) && \
    rm -rf ../bin/intel64/Release/lib/libgtest* && \
    rm -rf ../bin/intel64/Release/lib/libgmock* && \
    rm -rf ../bin/intel64/Release/lib/libmock* && \
    rm -rf ../bin/intel64/Release/lib/libtest*

# Delete TBB debug libs to prevent CI testing failing with TBB asserts
RUN find /home/dldt/inference-engine/temp/tbb/lib/ -name "*tbb*debug*so*" -exec rm -rf {} \;

ARG CUSTOM_IE_DIR=/opt/intel/dldt/inference-engine
ARG LIBDIR=${CUSTOM_IE_DIR}/lib/intel64

RUN mkdir -p    build/${CUSTOM_IE_DIR}/include && \
    cp -r       dldt/inference-engine/include/*                     build/${CUSTOM_IE_DIR}/include && \
    \
    mkdir -p    build/${LIBDIR} && \
    cp -r       dldt/bin/intel64/Release/lib/*     build${LIBDIR} && \
    \
    mkdir -p    build/${CUSTOM_IE_DIR}/src && \
    cp -r       dldt/inference-engine/src/*                         build/${CUSTOM_IE_DIR}/src/ && \
    \
    mkdir -p    build/${CUSTOM_IE_DIR}/share && \
    cp -r       dldt/build/share/*                 build/${CUSTOM_IE_DIR}/share/ && \
    \
    mkdir -p    build/${CUSTOM_IE_DIR}/external/ && \
    mv          dldt/inference-engine/temp/opencv_*_ubuntu18        dldt/inference-engine/temp/opencv && \
    cp -r       dldt/inference-engine/temp/*                        build/${CUSTOM_IE_DIR}/external

RUN for p in /usr /home/build/usr /${CUSTOM_IE_DIR} /home/build/${CUSTOM_IE_DIR}; do \
    pkgconfiglibdir="$p/lib/x86_64-linux-gnu" && \
    mkdir -p "${pkgconfiglibdir}/pkgconfig" && \
    pc="${pkgconfiglibdir}/pkgconfig/dldt.pc" && \
    echo "prefix=/opt" > "$pc" && \
    echo "libdir=${LIBDIR}" >> "$pc" && \
    echo "includedir=${CUSTOM_IE_DIR}/include" >> "$pc" && \
    echo "" >> "$pc" && \
    echo "Name: DLDT" >> "$pc" && \
    echo "Description: Intel Deep Learning Deployment Toolkit" >> "$pc" && \
    echo "Version: 5.0" >> "$pc" && \
    echo "" >> "$pc" && \
    echo "Libs: -L\${LIBDIR} -linference_engine -linference_engine_c_wrapper" >> "$pc" && \
    echo "Cflags: -I\${includedir}" >> "$pc"; \
    done;


FROM ${dldt} AS dldt-build

FROM ${gst} AS gst-build


FROM ubuntu:18.04
LABEL Description="This is the base image for GSTREAMER & DLDT Ubuntu 18.04 LTS"
LABEL Vendor="Intel Corporation"
WORKDIR /root

# Prerequisites
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
    libusb-1.0-0-dev libboost-all-dev libgtk2.0-dev python-yaml \
    \
    clinfo libboost-all-dev libjson-c-dev \
    build-essential cmake ocl-icd-opencl-dev wget gcovr vim git gdb ca-certificates libssl-dev uuid-dev \
    libgirepository1.0-dev \
    python3-dev python3-wheel python3-pip python3-setuptools python-gi-dev python-yaml \
    \
    libglib2.0-dev libgmp-dev libgsl-dev gobject-introspection libcap-dev libcap2-bin gettext \
    \
    libx11-dev iso-codes libegl1-mesa-dev libgles2-mesa-dev libgl-dev gudev-1.0 libtheora-dev libcdparanoia-dev libpango1.0-dev libgbm-dev libasound2-dev libjpeg-dev \
    libvisual-0.4-dev libxv-dev libopus-dev libgraphene-1.0-dev libvorbis-dev \
    \
    libbz2-dev libv4l-dev libaa1-dev libflac-dev libgdk-pixbuf2.0-dev libmp3lame-dev libcaca-dev libdv4-dev libmpg123-dev libraw1394-dev libavc1394-dev libiec61883-dev \
    libpulse-dev libsoup2.4-dev libspeex-dev libtag-extras-dev libtwolame-dev libwavpack-dev \
    \
    libbluetooth-dev libusb-1.0.0-dev libass-dev libbs2b-dev libchromaprint-dev liblcms2-dev libssh2-1-dev libdc1394-22-dev libdirectfb-dev libssh-dev libdca-dev \
    libfaac-dev libfdk-aac-dev flite1-dev libfluidsynth-dev libgme-dev libgsm1-dev nettle-dev libkate-dev liblrdf0-dev libde265-dev libmjpegtools-dev libmms-dev \
    libmodplug-dev libmpcdec-dev libneon27-dev libopenal-dev libopenexr-dev libopenjp2-7-dev libopenmpt-dev libopenni2-dev libdvdnav-dev librtmp-dev librsvg2-dev \
    libsbc-dev libsndfile1-dev libsoundtouch-dev libspandsp-dev libsrtp2-dev libzvbi-dev libvo-aacenc-dev libvo-amrwbenc-dev libwebrtc-audio-processing-dev libwebp-dev \
    libwildmidi-dev libzbar-dev libnice-dev libxkbcommon-dev \
    \
    libmpeg2-4-dev libopencore-amrnb-dev libopencore-amrwb-dev liba52-0.7.4-dev \
    \
    libva-dev libxrandr-dev libudev-dev \
    \
    && rm -rf /var/lib/apt/lists/* \
    && python3.6 -m pip install --upgrade pip && python3.6 -m pip install numpy opencv-python pytest

# Install
COPY --from=dldt-build /home/build /
COPY --from=gst-build /home/build /

ARG LIBDIR=/usr/lib/x86_64-linux-gnu

RUN echo "\
    ${LIBDIR}/gstreamer-1.0\n\
    /opt/intel/dldt/inference-engine/lib/intel64/\n\
    /opt/intel/dldt/inference-engine/external/tbb/lib\n\
    /opt/intel/dldt/inference-engine/external/mkltiny_lnx/lib\n\
    /opt/intel/dldt/inference-engine/external/vpu/hddl/lib\n\
    /opt/intel/dldt/inference-engine/external/opencv/opencv/lib/" > /etc/ld.so.conf.d/opencv-dldt-gst.conf && ldconfig

ENV GI_TYPELIB_PATH=${LIBDIR}/girepository-1.0

ENV PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/opt/intel/mediasdk/lib64/pkgconfig:/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}
ENV InferenceEngine_DIR=/opt/intel/dldt/inference-engine/share
ENV OpenCV_DIR=/opt/intel/dldt/inference-engine/external/opencv/opencv/cmake
ENV LIBRARY_PATH=/opt/intel/mediasdk/lib64:${LIBRARY_PATH}
ENV PATH=/opt/intel/mediasdk/bin:${PATH}

ENV LIBVA_DRIVERS_PATH=/opt/intel/mediasdk/lib64
ENV LIBVA_DRIVER_NAME=iHD
ENV GST_VAAPI_ALL_DRIVERS=1
ENV DISPLAY=:0.0
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:/opt/intel/dldt/inference-engine/external/hddl/lib
ENV HDDL_INSTALL_DIR=/opt/intel/dldt/inference-engine/external/hddl

ARG GIT_INFO
ARG SOURCE_REV

COPY . gst-video-analytics
ARG ENABLE_PAHO_INSTALLATION=false
ARG ENABLE_RDKAFKA_INSTALLATION=false
ARG BUILD_TYPE=Release
ARG EXTERNAL_GVA_BUILD_FLAGS

RUN mkdir -p gst-video-analytics/build \
    && cd gst-video-analytics/build \
    && cmake \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DVERSION_PATCH=${SOURCE_REV} \
        -DGIT_INFO=${GIT_INFO} \
        -DENABLE_PAHO_INSTALLATION=${ENABLE_PAHO_INSTALLATION} \
        -DENABLE_RDKAFKA_INSTALLATION=${ENABLE_RDKAFKA_INSTALLATION} \
        -DHAVE_VAAPI=ON \
        ${EXTERNAL_GVA_BUILD_FLAGS} \
        .. \
    && make -j $(nproc) \
    && make install

ENV GST_PLUGIN_PATH=/usr/local/lib/gstreamer-1.0/:/root/gst-video-analytics/
ENV PYTHONPATH=/root/gst-video-analytics/python:/usr/lib/python3.6/site-packages:$PYTHONPATH

