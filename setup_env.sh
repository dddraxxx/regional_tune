source ~/.bashrc

conda create -n metamath python=3.10 -y
conda activate metamath

# pip install torch==2.1.0 torchvision torchaudio
pip install -r requirements.txt
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
