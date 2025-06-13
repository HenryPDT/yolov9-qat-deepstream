#!/bin/bash

# Function to display usage message
usage() {
    echo "Usage: $0 [--no-trex] [--no-trt]" 1>&2
    echo "Options:" 1>&2
    echo "  --no-trex     Skip TREx installation" 1>&2
    echo "  --no-trt      Skip TensorRT upgrade" 1>&2
    exit 1
}

# Set default flags
install_trex=true  # TREx installation enabled by default
upgrade_trt=true   # TensorRT upgrade enabled by default

# Shared paths
TENSORRT_REPO_PATH="./TensorRT"  # Changed from /opt/nvidia/TensorRT to local directory
DOWNLOADS_PATH="./downloads"

# Parse command line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-trex)
            install_trex=false
            ;;
        --no-trt)
            upgrade_trt=false
            ;;
        *)
            usage
            ;;
    esac
    shift
done

# Function to get Ubuntu version
get_ubuntu_version() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo $VERSION_ID
    else
        echo "Unknown"
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    echo "Installing system dependencies..."
    local ubuntu_version=$(get_ubuntu_version)
    echo "Detected Ubuntu version: $ubuntu_version"
    
    sudo apt-get update || return 1
    
    # Common packages for all versions
    local common_packages="zip htop screen graphviz"
    
    # Version-specific packages
    case $ubuntu_version in
        "24.04")
            echo "Installing packages for Ubuntu 24.04"
            sudo apt-get install -y $common_packages libgl1 libfreetype-dev || return 1
            ;;
        "22.04"|"22.10")
            echo "Installing packages for Ubuntu 22.04/22.10"
            sudo apt-get install -y $common_packages libgl1-mesa-glx libfreetype6-dev || return 1
            ;;
        "20.04"|"20.10"|"21.04"|"21.10")
            echo "Installing packages for Ubuntu 20.04-21.10"
            sudo apt-get install -y $common_packages libgl1-mesa-glx libfreetype6-dev || return 1
            ;;
        *)
            echo "Warning: Unknown Ubuntu version. Attempting to install common packages..."
            sudo apt-get install -y $common_packages || return 1
            ;;
    esac
    
    return 0
}

# Function to upgrade TensorRT
upgrade_tensorrt() {
    echo "Upgrading TensorRT..."
    local ubuntu_version=$(get_ubuntu_version)
    echo "Detected Ubuntu version: $ubuntu_version"
    
    # Set TensorRT version
    local trt_version="10.9.0"
    
    # Set OS and CUDA version based on Ubuntu version
    local os
    local cuda
    case $ubuntu_version in
        "24.04")
            os="ubuntu2404"
            cuda="cuda-12.8"
            ;;
        "22.04"|"22.10")
            os="ubuntu2204"
            cuda="cuda-12.8"
            ;;
        "20.04"|"20.10"|"21.04"|"21.10")
            os="ubuntu2004"
            cuda="cuda-12.8"
            ;;
        *)
            echo "Warning: Unknown Ubuntu version. Using Ubuntu 22.04 as default..."
            os="ubuntu2204"
            cuda="cuda-12.8"
            ;;
    esac
    
    local tensorrt_package="nv-tensorrt-local-repo-${os}-${trt_version}-${cuda}_1.0-1_amd64.deb"
    local download_path="${DOWNLOADS_PATH}/${tensorrt_package}"
    
    # Create downloads directory if it doesn't exist
    mkdir -p "$DOWNLOADS_PATH" || return 1
    
    # Check if the package already exists
    if [ ! -f "$download_path" ]; then
        echo "Downloading TensorRT package for ${os} with ${cuda}..."
        wget "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/${trt_version}/local_repo/${tensorrt_package}" -O "$download_path" || return 1
    else
        echo "TensorRT package already exists at $download_path. Reusing existing file."
    fi
    
    # Install the package
    sudo dpkg -i "$download_path" || return 1
    sudo cp /var/nv-tensorrt-local-repo-${os}-${trt_version}-${cuda}/*keyring.gpg /usr/share/keyrings/ || return 1
    sudo apt-get update || return 1
    sudo apt-get install -y tensorrt || return 1
    sudo apt-get purge "nv-tensorrt-local-repo*" -y || return 1
    
    # Keep the downloaded file for potential reuse
    echo "TensorRT package kept at $download_path for future use"
    return 0
}

# Function to clone TensorRT repository once
clone_tensorrt_repo() {
    echo "Cloning NVIDIA TensorRT repository..."
    
    if [ ! -d "$TENSORRT_REPO_PATH" ]; then
        # Create directory and clone repository
        mkdir -p "$(dirname "$TENSORRT_REPO_PATH")" || return 1
        git clone https://github.com/NVIDIA/TensorRT.git "$TENSORRT_REPO_PATH" || return 1
        cd "$TENSORRT_REPO_PATH" || return 1
        git checkout release/10.9 || return 1
        echo "TensorRT repository cloned successfully to $TENSORRT_REPO_PATH"
    else
        echo "TensorRT repository already exists at $TENSORRT_REPO_PATH"
    fi
    
    return 0
}

# Function to install PyTorch Quantization
install_pytorch_quantization() {
    echo "Installing PyTorch Quantization..."
    
    # Navigate to PyTorch Quantization directory in TensorRT repo
    cd "$TENSORRT_REPO_PATH/tools/pytorch-quantization" || return 1
    
    # Install requirements and setup
    pip install setuptools || return 1
    pip install -r requirements.txt || return 1
    python setup.py install || return 1
    
    echo "PyTorch Quantization installed successfully"
    return 0
}

# Function to install TREx
install_trex_environment() {
    echo "Installing NVIDIA TREx environment..."
    # Check if TREx is not already installed
    if [ ! -d "/opt/nvidia_trex/env_trex" ]; then
        sudo apt-get install -y graphviz || return 1
        pip install virtualenv "widgetsnbextension>=4.0.9" || return 1
        
        sudo mkdir -p /opt/nvidia_trex || return 1
        cd /opt/nvidia_trex/ || return 1
        sudo python3 -m virtualenv env_trex || return 1
        source env_trex/bin/activate || return 1
        pip install "Werkzeug>=2.2.2" "graphviz>=0.20.1" || return 1
        
        # Navigate to TREx directory in TensorRT repo
        cd "$TENSORRT_REPO_PATH/tools/experimental/trt-engine-explorer" || return 1
        
        source /opt/nvidia_trex/env_trex/bin/activate || return 1
        pip install -e . || return 1
        pip install jupyter_nbextensions_configurator notebook==6.4.12 ipywidgets || return 1
        jupyter nbextension enable widgetsnbextension --user --py || return 1
        deactivate || return 1
    else
        echo "NVIDIA TREx virtual environment already exists. Skipping installation."
    fi
    return 0
}

# Function to cleanup
cleanup() {
    echo "Cleaning up..."
    sudo apt-get clean
    sudo rm -rf /var/lib/apt/lists/*
}

# Main installation process
main() {
    # Install system dependencies
    install_system_dependencies || { echo "Failed to install system dependencies"; exit 1; }
    
    # Upgrade TensorRT unless --no-trt flag is provided
    if $upgrade_trt; then
        upgrade_tensorrt || { echo "Failed to upgrade TensorRT"; exit 1; }
    else
        echo "Skipping TensorRT upgrade as requested"
    fi

    # Clone TensorRT repository once
    clone_tensorrt_repo || { echo "Failed to clone TensorRT repository"; exit 1; }

    # Install TREx by default unless --no-trex flag is provided
    if $install_trex; then
        install_trex_environment || { echo "Failed to install TREx environment"; exit 1; }
    fi
    
    # Always install PyTorch Quantization
    install_pytorch_quantization || { echo "Failed to install PyTorch Quantization"; exit 1; }

    # Final cleanup
    cleanup

    echo "Installation completed successfully."
    return 0
}

# Execute main function
main

