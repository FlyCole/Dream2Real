#!/bin/bash
echo "Installing Dream2Real..."
echo "An OpenAI API key is required, because Dream2Real uses GPT-4."
read -s -p "Which OpenAI API key would you like to use? " OPENAI_API_KEY
echo "export OPENAI_API_KEY=$OPENAI_API_KEY" > openai_key.sh
echo "Thanks. Key has been saved to: openai_key.sh"
echo "Continuing with Dream2Real installation..."

# Source conda configuration.
# NOTE: replace this if your conda is installed elsewhere.
source ~/miniconda3/etc/profile.d/conda.sh

conda create -y --name dream2real python=3.7.11
conda activate dream2real

# We need to install torch first separately because some packages use torch in setup.py.
pip install torch==1.13.1
pip install -r requirements.txt

# Build Instant-NGP.
cd reconstruction/instant-ngp
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
cd ../..

echo "Downloading demo dataset"
mkdir dataset
base_url="https://huggingface.co/datasets/FlyCole/Dream2Real/resolve/main/dataset"
files=("pool_X" "pool_triangle" "shopping" "shelf")
base_output_dir="./dataset"
for file in "${files[@]}"; do
    output_dir="${base_output_dir}/${file}"
    mkdir -p "$output_dir"
    wget -O "${output_dir}/${file}.zip" "${base_url}/${file}.zip?download=true"
    unzip -d "${base_output_dir}" "${output_dir}/${file}.zip"
    rm "${output_dir}/${file}.zip"
done

echo "Downloading cached method_out"
mkdir method_out
base_url="https://huggingface.co/datasets/FlyCole/Dream2Real/resolve/main/method_out"
files=("pool_X" "pool_triangle" "shopping" "shelf")
base_output_dir="./method_out"
for file in "${files[@]}"; do
    output_dir="${base_output_dir}/${file}"
    mkdir -p "$output_dir"
    wget -O "${output_dir}/${file}.zip" "${base_url}/${file}.zip?download=true"
    unzip -d "${base_output_dir}" "${output_dir}/${file}.zip"
    rm "${output_dir}/${file}.zip"
done

echo "Downloading XMem model"
wget -P segmentation/XMem/saves/ https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth

echo "Downloading SAM model"
mkdir models
wget -P models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo "Finished installing Dream2Real. Enjoy!"