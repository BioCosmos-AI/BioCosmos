#!/bin/bash
#SBATCH --job-name=download_vlm4bio 
#SBATCH --output=download_vlm4bio.log
#SBATCH --error=download_vlm4bio.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=hpg-dev 

# `scontrol show partition` to show types of partition on hypergator

# load module 
module load git

# get VLM4Bio dataset
mkdir /blue/arthur.porto-biocosmos/rdombrovski3.gatech/VLM4Bio
git clone https://huggingface.co/datasets/imageomics/VLM4Bio /blue/arthur.porto-biocosmos/rdombrovski3.gatech/VLM4Bio/

if [ $? -ne 0 ]; then
    echo "Git clone failed. Exiting."
    exit 1
fi

# check if script exists, then run scripts
first_script_path = "/blue/arthur.porto-biocosmos/rdombrovski3.gatech/VLM4Bio/download_bird_images.sh"
if [ -f "$first_script_path" ]; then
    # Run the shell scripts to download + process images as per documentation on VLM4Bio page
    bash /blue/arthur.porto-biocosmos/rdombrovski3.gatech/VLM4Bio/download_bird_images.sh
    bash /blue/arthur.porto-biocosmos/rdombrovski3.gatech/VLM4Bio/process_fish_butterfly_images.sh
else
    echo "script $first_script_path does not exist"
    exit 1
fi 
