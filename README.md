# BeautyBank-Inference: Inference Code for BeautyBank

This repository contains the **inference code** for the following paper:  
[BeautyBank: Encoding Facial Makeup in Latent Space](https://arxiv.org/abs/2411.11231).  

This repository provides the official PyTorch implementation for running inference using pre-trained models. It does **not** include training code.

![Teaser Image](img/teaser.png)

## Installation

### Clone this repo:

```bash
git clone https://github.com/Alululululululu/BeautyBank-Inference.git
cd BeautyBank-Inference
```

### Dependencies:

All dependencies for defining the environment are provided in the `environment` folder. We provide two versions of the Conda environment files:

1. **Full environment** (with specific package versions, recommended for exact reproducibility):
   ```bash
   conda env create -f ./environment/BeautyBank_env_full.yaml
   ```

2. **Minimal environment** (only explicitly installed packages, no strict versioning, for flexibility):
   ```bash
   conda env create -f ./environment/BeautyBank_env_minimal.yaml
   ```

### CUDA and PyTorch Compatibility:

- The system runs on **NVIDIA Tesla T4 GPUs**.
- CUDA version: **11.6**
- Recommended PyTorch version: **PyTorch 1.7.1 with CUDA 10.1 support** (other versions may also work but are not guaranteed)

To install the recommended PyTorch version, use:

```bash
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 --index-url https://download.pytorch.org/whl/cu101
```

## Download Data and Models

To run inference, you need to download the **makeup code** used for makeup transfer or other applications, along with the **pre-trained model checkpoint**.  
Additionally, we provide a **makeup dataset (1289 images)** that can be used to verify and select the appropriate makeup code for inference.

---

### **1. Download Makeup Code & Dataset (Required for Inference)**
Inference requires a **makeup code**, which encodes different makeup styles, and a set of reference images for style selection.  
Both are available on Hugging Face:

📂 **[Download from Hugging Face](https://huggingface.co/datasets/lulululululululululu/BeautyBank-Inference-Dataset)**

After downloading, extract and organize the files as follows:

```bash
# Create necessary directories
mkdir -p data/makeup          # Directory for storing PNG image data
mkdir -p checkpoints/makeup   # Directory for storing refined_makeup_code.npy
```

### **2. Download BeautyBank Pre-trained Models**
We provide two essential models for inference:

- **`generator.pt`**: The main generator model for BeautyBank.
- **`sampler.pt`**: The sampler model, used to refine the generated makeup styles.

You can download them from Hugging Face:

📂 **[Download from Hugging Face](https://huggingface.co/lulululululululululu/BeautyBank-Model/tree/main)**

### **3. Download Supporting Models
BeautyBank relies on an external Pixel2Style2Pixel (pSp) encoder to embed facial images into the latent space (Z+).
You must download this model separately from Google Drive:

📂 **[Download encoder.pt](https://drive.google.com/file/d/1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej/view)**

## Inference

### Makeup Transfer

To run the makeup transfer script, you can use one of the following methods:

#### Method 1: Simple Command

```bash
python run_makeup_transfer.py
```

This will run the script with default settings.

#### Method 2: Detailed Command with Parameters

If you need more control over the process, you can use the following command to specify additional parameters:

```bash
python makeup_transfer.py \
    --style makeup \
    --name makeup \
    --style_id 0 \
    --makeup_name refined_makeup_code.npy \
    --content ./data/test/003767.png \
    --output_path ./output/makeup/ \
    --weight 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 \
    --align_face
```

#### Arguments:
- **`--style`**: Type of transformation (e.g., `makeup`).
- **`--name`**: Name of the transformation (e.g., `makeup`).
- **`--style_id`**: ID for the makeup style to apply (e.g., `0` for the first style).
- **`--makeup_name`**: Path to the refined makeup code file (e.g., `refined_makeup_code.npy`).
- **`--content`**: Path to the bare-face image (e.g., `./data/test/003767.png`).
- **`--output_path`**: Directory where the output image will be saved (e.g., `./output/makeup/`).
- **`--weight`**: Weights for different parts of the transformation (e.g., intensity for makeup features).
- **`--align_face`**: Flag to indicate whether the face should be aligned for better results.

This command will apply the makeup transformation based on the provided makeup style, bare-face image, and other parameters and save the result in the specified output directory.





