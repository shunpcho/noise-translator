# !/bin/bash

# Download and unzip sample image dataset for denoising
# It is from google drive, link https://drive.google.com/file/d/1k_XVweS4EXwTNNCHmVRSAak-2YD65_RY/view?usp=drive_link

# Usage: bash sample_dataset.sh download unzip

download=$1
unzip=$2
FILE_ID="1k_XVweS4EXwTNNCHmVRSAak-2YD65_RY"
if [ "$download" = "download" ]; then
    echo "Downloading dataset..."
    mkdir -p data
    curl -L -o data/CC15.zip "https://drive.google.com/uc?export=download&id=${FILE_ID}"
fi

if [ "$unzip" = "unzip" ]; then
    echo "Unzip dataset..."
    unzip data/CC15.zip -d data/
fi