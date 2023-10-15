ARG BASE_IMAGE=nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
FROM $BASE_IMAGE

ARG GHIDRA=https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.1.5_build/ghidra_10.1.5_PUBLIC_20220726.zip

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=AEDT
# Remove any third-party apt sources to avoid issues with expiring keys.
# Then install some required packages.
RUN rm -f /etc/apt/sources.list.d/*.list \
  && apt-get update && apt-get install -y \
    gcc \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    zip \
    vim \
    openjdk-11-jdk \
    wget \
    python3.8 \
    python3-pip \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean \
  && ln -s /usr/bin/python3.8 /usr/bin/python

# Setup ghidra.jar
# Modify the java file, compile jar
RUN  wget -nv ${GHIDRA} -O ghidra_binary.zip \
 && unzip -n ghidra_binary.zip -d . \
 && cd ghidra_10.1.5_PUBLIC \
 && sed -i 's/\/\/builder.addAllModules();/builder.addAllModules();/g' Ghidra/Features/Base/ghidra_scripts/BuildGhidraJarScript.java \
 && support/analyzeHeadless . tempproj -deleteProject -scriptPath ../Ghidra/Features/Base/ghidra_scripts/ -prescript BuildGhidraJarScript.java \
 && mv ghidra.jar /app/ghidra.jar \
 && rm -rf /app/ghidra_binary.zip \
 && rm -rf /app/ghidra_10.1.5_PUBLIC
ENV GHIDRA_JAR=/app/ghidra.jar

# Install python requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# userdb and packer info for analysis
ADD https://raw.githubusercontent.com/ynadji/peid/master/userdb.txt .
ADD https://raw.githubusercontent.com/Yara-Rules/rules/1de664c39014f826f772a2288dd3f73f852ef574/packers/packer.yar .

ARG USER
ARG UID
ARG GID

# All users can use /home/user as their home directory
ENV HOME=/home/${USER}
RUN mkdir /home/${USER} \
 && chmod 777 /home/${USER} \
 && addgroup --gid ${GID} ${USER} || echo "group gid=${GID} user=${USER} already exist" \
 && adduser --disabled-password --uid ${UID} --gid ${GID} --gecos '' --shell /bin/bash ${USER} || echo "user gid=${GID} user=${USER} already exist" \
 && chown -R ${USER}:${USER} /app \
 && echo "${USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-${USER} \
 && chmod 777 /home/${USER}
USER ${USER}

ENTRYPOINT /bin/bash
