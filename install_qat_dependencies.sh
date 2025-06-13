#!/bin/bash

# Simple QAT Dependencies Installation Script
# Installs only system dependencies and PyTorch Quantization (no TREx or TensorRT upgrade)

# Shared paths
TENSORRT_REPO_PATH="./TensorRT"
DOWNLOADS_PATH="./downloads"

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

    # Clone TensorRT repository once
    clone_tensorrt_repo || { echo "Failed to clone TensorRT repository"; exit 1; }
    
    # Install PyTorch Quantization
    install_pytorch_quantization || { echo "Failed to install PyTorch Quantization"; exit 1; }

    # Final cleanup
    cleanup

    echo "QAT dependencies installation completed successfully."
    return 0
}

# Execute main function
main 