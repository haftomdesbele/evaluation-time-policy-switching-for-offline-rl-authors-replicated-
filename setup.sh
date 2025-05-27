CONDA_DIR=~/miniconda3
MUJOCO_DIR=~/.mujoco

if [ ! -d "$CONDA_DIR" ]; then
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    conda env create -f environment.yml

fi


if [ ! -d "$MUJOCO_DIR/mujoco210" ]; then
    mkdir -p $MUJOCO_DIR
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O $MUJOCO_DIR/mujoco210.tar.gz
    tar -xzf $MUJOCO_DIR/mujoco210.tar.gz -C $MUJOCO_DIR
    rm $MUJOCO_DIR/mujoco210.tar.gz
fi

sudo apt install libglew-dev
unzip models.zip
