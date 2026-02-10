# Documentation for setting up a new environment:
# https://help.rc.ufl.edu/doc/Managing_Python_environments_and_Jupyter_kernels

echo "Starting environment setup at: $(date)"
echo "Running on node: $(hostname)"

# Load required modules
module load conda
# module load cuda/11.8


conda create -p /blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/unicom_distributed python=3.10 -y

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate /blue/arthur.porto-biocosmos/tdeatherage3.gatech/conda/envs/unicom_distributed

conda install cudatoolkit=11.3 pytorch=1.12.1=gpu_cuda* -c pytorch

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "Installing FAISS-GPU..."
conda install -c conda-forge faiss-gpu -y

echo "Installing OpenCLIP..."
pip install open_clip_torch

echo "Installing other required packages..."
conda install pandas numpy scipy pillow tqdm matplotlib -y

# Verify installation
echo "Verifying installations..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import open_clip; print('OpenCLIP imported successfully')"
python -c "import faiss; print('FAISS version:', faiss.__version__)"

echo "Environment setup completed at: $(date)"
