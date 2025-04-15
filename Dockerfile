FROM ubuntu:22.04
WORKDIR /neuron

COPY ./entrypoint .

RUN apt update \
    && apt install wget gnupg -y

RUN . /etc/os-release \
    && echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" >> /etc/apt/sources.list.d/neuron.list \
    && wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

RUN apt update \
    && apt install linux-headers-generic git vim -y \
    && apt install aws-neuronx-dkms=2.* aws-neuronx-collectives=2.* aws-neuronx-runtime-lib=2.* aws-neuronx-tools=2.* -y

RUN PATH=/opt/aws/neuron/bin:$PATH

RUN apt install python3 python3-pip python-is-python3 python3-venv -y \
    && python -m venv venv

RUN . venv/bin/activate \
    && pip install --upgrade pip \
    && pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com \
    && pip install awscli neuronx-cc==2.* numpy

ENTRYPOINT [ "./entrypoint" ]
