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
mkdir -p data/makeup/images/train         # Directory for storing PNG image data
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


The structure of the dataset and models will be as follows:

```
checkpoint
├── makeup               
│   ├── generator.pt
│   ├── refined_makeup_code.npy
│   ├── sampler.pt
├── encoder.pt
```
Ensure that the above files are located in the `checkpoint` directory for the proper execution of the makeup transfer process.



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
- **`--weight`**: Weights for different parts of the transformation.
- **`--align_face`**: Flag to indicate whether the face should be aligned for better results.

This command will apply the makeup transformation based on the provided makeup style, bare-face image, and other parameters and save the result in the specified output directory.

### Facial Images Generation with Makeup Injection

We generate facial images with makeup injection by modifying random Gaussian noise to replace the bare-face code. This technique allows for diverse face generation while retaining the specified makeup. 

#### Process Overview:
- We randomly select several sets of encoded makeup codes.
- For each makeup code, random Gaussian noises are generated to replace the bare-face code.
- The fusion module of **BeautyBank** is then applied to create the facial image, incorporating the makeup while varying other aspects like expressions, poses, genders, and hairstyles.

To generate facial images with makeup injection, you can run the following two versions of the script depending on your preference for simplicity or more control over the parameters.

#### Method 1: Simple Command

If you prefer a quick and straightforward way to generate facial images with makeup, you can use the following command:

```bash
python run_generate_face.py
```

This will run the script with default settings and generate the facial images based on the pre-configured parameters.

#### Method 2: Detailed Command with Parameters

For more flexibility and control over the generation process, you can run the script with specific arguments. Use the following command:

```bash
python generate_face.py \
    --style makeup \
    --name makeup \
    --style_id 0 \
    --content ./data/test/003767.png \
    --makeup_name refined_makeup_code.npy \
    --output_path ./output/makeup/ \
    --weight 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 \
    --align_face
```

#### Arguments:
- **`--style`**: Specifies the type of transformation to apply (e.g., `makeup`).
- **`--name`**: The name for the transformation, such as `makeup`.
- **`--style_id`**: ID for the specific makeup style (e.g., `0` for the first style).
- **`--content`**: Path to the default input content image (e.g., `./data/test/003767.png`).
- **`--makeup_name`**: Path to the refined makeup code file (e.g., `refined_makeup_code.npy`).
- **`--output_path`**: Directory where the output facial images will be saved (e.g., `./output/makeup/`).
- **`--weight`**: Weights for different aspects of the transformation.
- **`--align_face`**: Option to align the face for better consistency in the generated results.


### Makeup Interpolation

Makeup interpolation allows seamless transitions between different makeup styles by interpolating either the bare-face codes or the makeup codes. Since **BeautyBank** includes two style paths, interpolation between different source images and reference makeup styles is achieved by blending these codes, enabling smooth transitions in makeup styles.

#### Method 1: Simple Command

To quickly perform makeup interpolation using default parameters, you can use the following command:

```bash
python run_interpolate_makeup.py
```

This will execute the makeup interpolation with pre-set configurations, interpolating between two makeup codes for facial image generation.

#### Method 2: Detailed Command with Parameters

For more flexibility and control over the interpolation process, you can run the script with specific arguments. The following command allows you to specify the two source images, corresponding makeup styles, and other parameters:

```bash
python interpolate_makeup.py \
    --align_face \
    --style makeup \
    --name makeup \
    --content ./data/test/003767.png \
    --content2 ./data/test/083311.png \
    --makeup_name_1 refined_makeup_code.npy \
    --style_id 0 \
    --makeup_name_2 refined_makeup_code.npy \    # can be different npy files
    --style_id2 0
```

#### Arguments:
- **`--align_face`**: Flag to align the face for better results during interpolation.
- **`--style`**: Name of the style for the first source makeup.
- **`--name`**: The name of the style to apply.
- **`--content`**: Path to the first content image for makeup interpolation.
- **`--content2`**: Path to the second content image for makeup interpolation.
- **`--makeup_name_1`**: Path to the first refined makeup code file.
- **`--style_id`**: The style ID for the first makeup style.
- **`--makeup_name_2`**: Path to the second refined makeup code file.
- **`--style_id2`**: The style ID for the second makeup style.

## Important Notes

- **Non-Makeup Features Disentanglement**: The current `refined_makeup_code.npy` files have not undergone Non-Makeup Features Disentanglement. Performing this step can significantly enhance the performance of generation and transfer tasks.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{lu2024beautybank,
  title={BeautyBank: Encoding Facial Makeup in Latent Space},
  author={Lu, Qianwen and Yang, Xingchao and Taketomi, Takafumi},
  journal={arXiv preprint arXiv:2411.11231},
  year={2024},
  url={https://github.com/Alululululululu/BeautyBank-Inference.git}
}







