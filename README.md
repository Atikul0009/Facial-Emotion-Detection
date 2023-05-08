

<h2 align="center">Hi 👋, I'm Md Atikul Islam</h2>


- 📫 How to reach me **atikulislam339@gmail.com**


## Project Title

<h3 align="center">Facial Emotion Detection by feature extraction and classification using Deep Neural Networks</h3>


 # System Requirement
1. Ubuntu 22.0
2. Nvidia GPU with cuda 11.8 support and minimum 16 GB VRAM
3. Ram 32 GB

# Libraries and tools

1. Anaconda 22.9.0
2. Tensorflow
3. OpenCV
4. Scikit-learn
5. Numpy
6. Pandas

## Install Dependcies

Install Anaconda

```bash
  Follow the link https://docs.anaconda.com/free/anaconda/install/linux/
```

Create virtual environment using environement.yml

```bash
  conda env create -f environment.yml
```
Activate environment

```bash
  conda activate pytorch-tf
```


## Run The project

Clone the project

```bash
  https://github.com/Atikul0009/Facial-Emotion-Detection.git
```

Go to the project directory

```bash
  cd Facial-Emotion-Detection
```
Download Dataset

```bash
  Download from https://www.kaggle.com/datasets/msambare/fer2013/download?datasetVersionNumber=1
```

Prepare Dataset

```bash
  python3 ferprep.py
  
  This will create a folder 'imagesFer' and 'fer2013_updated.csv' that contain location of each image , it's label and train/test split.
```
Training Model
```bash 
 python3 ferDet.py
```
