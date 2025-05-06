FROM continuumio/miniconda3:latest

# 환경변수 설정
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    unzip \
    vim \
    git \
    curl \
    cmake \
    clang \
    llvm-dev \
    libclang-dev \
    graphviz \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 최신 Rust 설치 (1.69 이상)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# 환경파일 복사 및 conda 환경 생성
COPY bap3.yml /tmp/bap3.yml
RUN conda update -n base -c defaults conda
RUN conda env create -f /tmp/bap3.yml
RUN conda clean --all -f -y

# conda 환경 활성화
ENV CONDA_DEFAULT_ENV=bap3
ENV PATH=/opt/conda/envs/bap3/bin:$PATH

RUN mkdir -p /usr/share/fonts/truetype/pretendard && \
    curl -L -o /tmp/Pretendard.zip https://github.com/orioncactus/pretendard/releases/download/v1.3.9/Pretendard-1.3.9.zip
RUN unzip /tmp/Pretendard.zip -d /tmp/pretendard
RUN cp /tmp/pretendard/public/static/*.otf /usr/share/fonts/truetype/pretendard/ && \
    fc-cache -fv && \
    rm -rf /tmp/Pretendard.zip /tmp/pretendard

# matplotlibrc 설정
RUN mkdir -p /root/.config/matplotlib && echo "\
font.family         : sans-serif\n\
font.sans-serif     : Pretendard, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif\n\
axes.unicode_minus  : False\n\
" > /root/.config/matplotlib/matplotlibrc

# Jupyter Notebook 사용 시 포트 설정
EXPOSE 8888

# VS Code 호환을 위한 기본 명령
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]