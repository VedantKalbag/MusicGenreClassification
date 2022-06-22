FROM tensorflow/tensorflow:2.9.1-gpu
RUN apt-get -y install git wget apt-utils libsndfile1 ffmpeg screen && \
    sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -p git \
    -p ssh-agent \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -a alias gl="git log --oneline" \
    -a alias build="source ~/.zshrc" \
    -a alias config="code ~/.zshrc" \
    -a alias path="echo $PATH" &&\
    zsh &&\
    source ~/.zshrc &&\
    /usr/bin/python3 -m pip install --upgrade pip
# # Let scripts know we're running in Docker (useful for containerised development)
# ENV RUNNING_IN_DOCKER true

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN zsh

# BUILD THE DOCKER IMAGE USING THIS:
# docker build -t <container name>  -f Dockerfile .

# RUN THIS AFTER BUILDING THE DOCKERFILE 
# docker run -it -v ~/Workspace/nvme_2tb_1:/usr/research/drive --gpus all --name <build_name>  <container_name> 