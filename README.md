
# Project: Automated Vision AI

## Overview

**Automated Vision AI** is a machine learning solution focused on solving image-based challenges using advanced AI techniques. The project includes two primary components:

1. **Fresh and Rotten Fruit Classification** – A convolutional neural network (CNN) model that classifies fruit images into fresh or rotten categories.
2. **OCR Image Processing** – Optical Character Recognition (OCR) to detect and extract text from images, enabling text-based analysis of images.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [Team](#team)
- [License](#license)

## Project Structure

The project is divided into two Jupyter notebooks for the respective tasks:

1. **fresh-and-rotten-fruit-classification.ipynb**: Contains the code for fruit image classification.
2. **ocrimage.ipynb**: Contains the code for extracting text from images using OCR techniques.

Each notebook consists of the following steps:

1. **Data Collection**: Gathering datasets for training and testing the models.
2. **Data Preprocessing**: Cleaning, normalizing, and preparing the data for model input.
3. **Model Training**: Implementing machine learning models (CNN for classification, OCR algorithms for text detection).
4. **Evaluation and Results**: Testing the model performance and evaluating results.
5. **Deployment (optional)**: Code for deploying the models into a production environment, if applicable.

## Installation

To get started with Automated Vision AI, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/25Anurag/flipkartgrid6.0.git
   ```

2. **Install Dependencies**: Ensure you have Python installed. Then, install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```

   **Libraries include**:
   - TensorFlow / PyTorch (for fruit classification)
   - OpenCV (for image processing)
   - Tesseract or other OCR libraries (for text extraction)
   - NumPy, Pandas, Matplotlib (for data handling and visualization)

3. **Set up Tesseract for OCR**: Follow [this guide](https://github.com/tesseract-ocr/tesseract) to install Tesseract OCR on your system.

## Usage

### Fresh and Rotten Fruit Classification

1. **Load the notebook**: Open `fresh-and-rotten-fruit-classification.ipynb` in Jupyter or any compatible environment.
2. **Prepare Data**: Ensure that the dataset of fruit images (fresh and rotten) is available in the specified directory. You can modify the path in the notebook accordingly.
3. **Run the model**: Execute each cell of the notebook to preprocess data, train the model, and classify the fruit images. The final output will classify images as "Fresh" or "Rotten".

### OCR Image Processing

1. **Load the notebook**: Open `ocrimage.ipynb` in Jupyter or any compatible environment.
2. **Provide Input Image**: Supply the image containing text that you want to process. Modify the image path in the notebook accordingly.
3. **Run the OCR Model**: Execute the cells to preprocess the image, run OCR, and extract text. The output will be the text detected in the image.

## Workflow

### Fruit Classification Workflow:
1. **Data Collection**: Gather fruit images (both fresh and rotten).
2. **Preprocessing**: Resize, normalize, and augment the images.
3. **Model Training**: Use a CNN to classify images into categories.
4. **Evaluation**: Measure accuracy, precision, and recall to assess model performance.
5. **Deployment** (Optional): Deploy the model using a web service or local hosting.

### OCR Workflow:
1. **Image Input**: Provide the image containing text.
2. **Preprocessing**: Convert to grayscale, binarize, and apply image filters to enhance text.
3. **OCR**: Use Tesseract or another OCR library to extract text.
4. **Postprocessing**: Clean up the extracted text (if necessary).

## License

This project is licensed under the MIT License.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.
```

## Team

- **Team Lead**: Anurag Kumar
- **Developers**: Anurag Kumar

## Contact

For questions or collaborations, please contact:
- **Email**: anu25rag1@gmail.com
- **GitHub**: [https://github.com/25Anurag/flipkartgrid6.0](https://github.com/25Anurag/flipkartgrid6.0)
