# Senior-Comps-2023

Replication Instructions

1. Anaconda Installation:

Start by downloading Anaconda from the official website:  [Free Download | Anaconda](https://www.anaconda.com/download). Make sure to select the version compatible with your operating system (Mac, Windows, Linux). It is important to note that Anaconda is a powerful suite of tools that simplifies package management and deployment for Python projects. Also, ensure there is at least 1.1GB of storage available for the installation.
  
2. Anaconda Navigator Installation:
Once Anaconda is installed, open a terminal (for Linux/Mac) or Anaconda Prompt (for Windows). To install Anaconda Navigator, a graphical user interface that helps in managing conda packages, environments, and more, type the command:
     
     conda install -c anaconda anaconda-navigator
     
3. Launching Anaconda Navigator:
   - Launch Anaconda Navigator from your computer's applications or by running the following command in the terminal/Anaconda Prompt:
     
     anaconda-navigator

This tool streamlines access to different applications and tools for your data science projects.
     
4. Jupyter Notebook:
Inside Anaconda Navigator, launch Jupyter Notebook 6.5.4. Jupyter Notebook is chosen for its interactive features that facilitate live code, equations, visualizations, and explanatory text. Create a new Python 3 notebook, which allows for working with multiple code kernels and running different sections in an organized manner.

5. Python Version:
This project is using Python version 3.11.5, which is packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)].

6. Dataset Download:
Download the HAM10000 dataset from[ [ViDIR Dataverse (harvard.edu)](The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions - ViDIR Dataverse (harvard.edu))](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). Download both the images in the folder and the metadata in a .txt file format, which includes crucial information like image names and corresponding labels, vital for the classification system. This dataset is particularly chosen for its diversity and size, making it suitable for training a robust model.

7. Python Packages:
Make sure you have the necessary Python packages installed. You can install them using `conda` or `pip`:

Image Processing and Computer Vision:
OpenCV (`cv2`): OpenCV library for computer vision tasks, used for image loading, color conversion, and Hu moments calculation.
Pillow (`PIL`): Python Imaging Library (PIL) for working with images, used for opening and resizing images.

Data Manipulation and Computation:
NumPy (`numpy`): A fundamental package for numerical computing in Python, used for handling arrays and numerical operations.
TensorFlow (`tensorflow`): An open-source deep learning framework developed by Google, used for building and training neural networks.
Scikit-Learn (`sklearn`): A machine learning library for various tasks like data splitting, model evaluation, and metrics calculation.

Plotting and Visualization:
Matplotlib (`matplotlib`): A widely-used Python library for creating static, animated, and interactive visualizations in Python.

File and Directory Operations:
Os: The `os` module provides functions for interacting with the operating system, used for working with file paths and directories.

Deep Learning Framework (Keras):
keras.layers: Keras layers for building the neural network architecture.
keras.models: Keras models for defining and compiling the deep learning model.
keras.optimizers: Keras optimizers for configuring the model's training optimization algorithm.
keras.callbacks: Keras callbacks for monitoring and controlling the training process.
keras.preprocessing.image: Keras tools for image preprocessing and data augmentation.

Data Splitting and Evaluation Metrics:
sklearn.model_selection.train_test_split: Used for splitting the dataset into training, validation, and test sets.
sklearn.metrics: Various metrics for evaluating machine learning model performance, such as precision, recall, F1-score, ROC curve, AUC, and confusion matrix.

Data Augmentation and Image Generation:
ImageDataGenerator (from keras.preprocessing.image): Used for data augmentation, which helps improve the model's robustness.


8. File and Folder Organization:
Organize your project files and folders according to the provided code architecture. It is crucial to correctly manage folder paths for images and labels and to meticulously handle the metadata, as it contains essential information for every image in the dataset.
