# NHANES-Image Data Mapper

This project creates a synthetic mapping between NHANES (National Health and Nutrition Examination Survey) data and medical images, providing a framework for multimodal machine learning research.

## Project Overview

The NHANES-Image Data Mapper creates associations between patient health data from NHANES datasets and medical images. This allows researchers to:

- Study relationships between health metrics and visual medical data
- Create multimodal machine learning models
- Generate synthetic paired datasets for research purposes

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv env
   ```
3. Activate the virtual environment:
   - Windows: `env\Scripts\activate`
   - Mac/Linux: `source env/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Structure

The project expects the following directory structure:

```
mini proj/
├── dataset/
│   ├── NHANES/
│   │   ├── diet.csv
│   │   ├── examination.csv
│   │   ├── labs.csv
│   │   ├── demographic.csv
│   │   └── ...
│   └── Image data/
│       └── [image categories]/
├── data_mapper.py
└── eda.ipynb
```

## Usage

### Basic Usage

```python
# Initialize the mapper
mapper = NHANESImageMapper(
    nhanes_dir="dataset/NHANES",
    image_dir="dataset/Image data"
)

# Load and process data
nhanes_data = mapper.load_nhanes_data()

# Scan for available image categories
categories = mapper.scan_image_categories()

# Create synthetic mapping between patients and images
mapping = mapper.create_synthetic_mapping(save_path="patient_image_mapping.json")

# Create a paired dataset for training
paired_dataset = mapper.create_paired_dataset()
```

### Exploratory Data Analysis

The project includes an EDA notebook (`eda.ipynb`) that demonstrates how to:
- Load and explore NHANES datasets
- Visualize key health metrics
- Understand the structure of the image data

## Files

- `data_mapper.py`: Core implementation of the NHANES-Image mapper
- `eda.ipynb`: Exploratory data analysis notebook
- `patient_image_mapping.json`: Generated mapping between patients and images
