# Droplet Evaporation Analysis 💧

An automated computer vision tool for analyzing the evaporation kinetics of sessile droplets. This project compares the evaporation rates of water droplets with and without surfactants (Decyl) on hydrophobic vs. hydrophilic surfaces.

## 📌 Overview

This repository contains the code, analysis tools, and full documentation of the project.
The goal was to overcome the instability of manual machine measurements by developing a robust image processing algorithm.

**Experimental Setup:**
The analysis covers 4 distinct treatments, each with **3 repetitions**:
1. **Tap Water on Glass** 
2. **Tap Water on Parafilm** 
3. **Decyl Surfactant on Glass**
4. **Decyl Surfactant on Parafilm**

## 📂 Project Resources

### 1. Documentation (In this Repository)
You can find the full research details directly in this GitHub repo:
* **📄 Project Report (PDF):** Full explanation of background, methods, and results.
* **📊 Presentation (PPTX):** Summary slides of the project.


### 2. Dataset Access (External Drive)
Due to the large size of the high-resolution image sequences, the raw image dataset is hosted externally.

👉 **[Download Raw Images from Google Drive](https://drive.google.com/drive/folders/1j0XsO7YX_w4uiOnQNOatF7oynLYMRf3V)**

**Note:** The Drive folder contains the raw image sequences organized by treatment. You need to download them (or mount the drive) to run the code.

## 🚀 How to Run the Code

**Option A: Google Colab (Recommended)**
1.  Upload the `.ipynb` notebook to Google Colab.
2.  Mount the Drive containing the images:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3.  Point the script to the folder path in your Drive.

**Option B: Running Locally**
1.  Download the image folders from the Drive link above.
2.  Ensure your local folder structure matches the expected format:
    ```text
    root_folder/
    ├── tap on glass/
    │   ├── rep1/
    │   ├── rep2/
    │   └── rep3/
    ├── decyl on glass/ ...
    ...
    ```
3.  Run the notebook or script.

## 🛠️ Code Description
* **`finding_base.py`**: Main algorithm for segmentation and volume calculation.
* **`evaporation.ipynb`**: Notebook for batch processing and generating graphs.
* **`Droplet Evaporation Analysis.ipynb`**: This application analyzes time-lapse images of sessile droplets to quantify changes in droplet geometry and volume during evaporation. The workflow combines automated image processing with optional user-guided baseline selection to ensure robust and reproducible measurements across different surfaces.
