# C++ data-generation toolchain: ROOT (RNTuple) + FastJet + fjcontrib LundPlane + PYTHIA 8.
# Pins the generation stage so jets.root is reproducible off-cluster (§5).
FROM rootproject/root:6.36.00-ubuntu24.04

ARG FASTJET_VERSION=3.4.3
ARG FJCONTRIB_VERSION=1.100
ARG PYTHIA_VERSION=8312

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake wget ca-certificates rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# FastJet
RUN wget -q http://fastjet.fr/repo/fastjet-${FASTJET_VERSION}.tar.gz \
    && tar xf fastjet-${FASTJET_VERSION}.tar.gz && cd fastjet-${FASTJET_VERSION} \
    && ./configure --prefix=/usr/local && make -j"$(nproc)" && make install && cd .. \
    && rm -rf fastjet-${FASTJET_VERSION}*

# fjcontrib (LundPlane tool)
RUN wget -q https://fastjet.hepforge.org/contrib/downloads/fjcontrib-${FJCONTRIB_VERSION}.tar.gz \
    && tar xf fjcontrib-${FJCONTRIB_VERSION}.tar.gz && cd fjcontrib-${FJCONTRIB_VERSION} \
    && ./configure --fastjet-config=/usr/local/bin/fastjet-config \
    && make -j"$(nproc)" && make install && make fragile-shared-install && cd .. \
    && rm -rf fjcontrib-${FJCONTRIB_VERSION}*

# PYTHIA 8
RUN wget -q https://pythia.org/download/pythia83/pythia${PYTHIA_VERSION}.tgz \
    && tar xf pythia${PYTHIA_VERSION}.tgz && cd pythia${PYTHIA_VERSION} \
    && ./configure --prefix=/usr/local && make -j"$(nproc)" && make install && cd .. \
    && rm -rf pythia${PYTHIA_VERSION}*

ENV PYTHIA8_ROOT_DIR=/usr/local
WORKDIR /work
COPY cpp/ /work/cpp/
RUN cmake -S cpp -B cpp/build && cmake --build cpp/build -j"$(nproc)" \
    && ctest --test-dir cpp/build --output-on-failure
ENTRYPOINT ["/work/cpp/build/pythia_driver"]
