
# CUDA 基础镜像：为了匹配 environment.yml 中的 torch==2.7.0+cu128 / nvidia-*-cu12==12.8.*，这里用 CUDA 12.8
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

# ------------------------------------------------------------
# 1) 系统依赖
#    - OpenGL / X11 / GLVND / mesa：IsaacSim / Omniverse kit 常用
#    - audio / nss / dbus / gtk 等：很多 Omniverse 组件会用到
#    - terminfo/ncurses：修复 “can't find terminfo database”
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential \
    vim tmux ranger \
    # terminfo / ncurses
    ncurses-base ncurses-term ncurses-bin libtinfo6 \
    # OpenGL / X11 / GLVND / mesa
    libgl1-mesa-glx libgl1-mesa-dri mesa-utils \
    libglvnd0 libgl1 libegl1 \
    libx11-6 libxext6 libxrender-dev libxrandr2 libxi6 libxfixes3 libxkbcommon0 \
    libxcursor1 libxinerama1 libxcomposite1 libxdamage1 \
    libglib2.0-0 libsm6 \
    # Omniverse/IsaacSim 常见运行依赖
    libnss3 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libgtk-3-0 libasound2 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 2) pip 额外 index（关键）
#    - PyTorch CUDA wheels: https://download.pytorch.org/whl/cu128
#    - NVIDIA PyPI:         https://pypi.nvidia.com
# ------------------------------------------------------------
RUN printf '%s\n' \
  '[global]' \
  'extra-index-url = https://download.pytorch.org/whl/cu128 https://pypi.nvidia.com' \
  'timeout = 120' \
  > /etc/pip.conf

# ------------------------------------------------------------
# 3) 安装 Miniforge / Conda
# ------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh
ENV PATH=${CONDA_DIR}/bin:$PATH

# ------------------------------------------------------------
# 4) 创建 conda 环境 gentle（来自 environment.yml）
# ------------------------------------------------------------
WORKDIR /workspace
COPY environment.yml /workspace/environment.yml

RUN conda install -y mamba -n base -c conda-forge && \
    mamba env create -f /workspace/environment.yml && \
    conda clean -afy

# ------------------------------------------------------------
# 5) 运行时库路径（两层保险：ENV + activate.d）
# ------------------------------------------------------------
ENV CONDA_ENV_NAME=gentle
ENV LD_LIBRARY_PATH=/opt/conda/envs/${CONDA_ENV_NAME}/lib/:${LD_LIBRARY_PATH}

RUN mkdir -p /opt/conda/envs/${CONDA_ENV_NAME}/etc/conda/activate.d && \
    printf '%s\n' \
      "export LD_LIBRARY_PATH=/opt/conda/envs/${CONDA_ENV_NAME}/lib/:\\$LD_LIBRARY_PATH" \
      > /opt/conda/envs/${CONDA_ENV_NAME}/etc/conda/activate.d/env_vars.sh

# ------------------------------------------------------------
# 6) （可选但推荐）拷贝 .netrc
#    - 如果你用它来访问需要鉴权的源（比如某些 NVIDIA/NGC 资源），就保留
# ------------------------------------------------------------
# 如果构建上下文里没有 .netrc，这两行会报错；不需要就注释掉
COPY .netrc /root/.netrc
RUN chmod 600 /root/.netrc

# ------------------------------------------------------------
# 7) 预装 IsaacLab（源码 + editable）
#    - 你的 env 里已有 isaaclab==0.44.9（pip），这里额外把 repo 放进镜像，方便你后续改代码/打补丁
# ------------------------------------------------------------
ARG ISAACLAB_TAG=v2.2.0
RUN git clone https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab && \
    cd /opt/IsaacLab && \
    git checkout ${ISAACLAB_TAG}

# 后续 RUN 默认在 conda env gentle 下执行
SHELL ["conda", "run", "-n", "gentle", "/bin/bash", "-lc"]

# IsaacLab editable 安装（如果你只想用 pip 的 isaaclab 包，可以把这行注释掉）
RUN bash /opt/IsaacLab/isaaclab.sh --install

# ------------------------------------------------------------
# 8) 进入容器时自动启用 conda 并进入 gentle
# ------------------------------------------------------------
RUN conda init bash && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /root/.bashrc && \
    echo 'conda activate gentle || true' >> /root/.bashrc

WORKDIR /workspace
CMD ["bash"]
