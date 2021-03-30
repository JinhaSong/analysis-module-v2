FROM sogangmm/cuda:9.1-cudnn7-devel-ubuntu16.04-py27-mysql

RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install python python-pip python-dev \
    git wget ssh vim \
    apt-utils libgl1 libxrender1 libsm6 ffmpeg\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ADD . .
RUN wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
RUN python get-pip.py --force-reinstall
RUN pip install -r requirements.txt
RUN wget ftp://mldisk.sogang.ac.kr/place/place47/place47.pth.tar -P /workspace/Modules/places

ENV DJANGO_SUPERUSER_USERNAME root
ENV DJANGO_SUPERUSER_EMAIL none@none.com
ENV DJANGO_SUPERUSER_PASSWORD password

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

RUN chmod -R a+w /workspace

EXPOSE 8000