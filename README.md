# SoundSense — Music Genre Classification

SoundSense is a user-friendly web application that uses deep learning to classify various music genres from audio files.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation Guide](#installation-guide)
- [How to Run](#how-to-run)
- [Supported Formats](#supported-formats)
- [Troubleshooting](#troubleshooting)

---

## Features
- **Instant Classification**: Upload an audio file and get the genre prediction in seconds.
- **Deep Learning Model**: Uses a trained Convolutional Neural Network (CNN) for high accuracy.
- **Ensemble Prediction**: Analyzes multiple segments of the audio for more reliable results.
- **Prediction History**: Keep track of your previous classification sessions.
- **Downloadable Reports**: Save your results as JSON reports.

## Requirements
To run this project, you need:
- **Python 3.8+**
- **Microsoft Visual C++ Redistributable** (Required for TensorFlow on Windows)

## Installation Guide

Follow these steps to set up the project on your local machine:

### 1. Set up a Virtual Environment
It's recommended to use a virtual environment to keep dependencies isolated. If you already have a `venu` folder, you can use it.
```bash
# To create a new environment
python -m venv venu
```

### 2. Install Project Dependencies
Use `pip` to install all necessary Python packages:
```bash
# Activate the environment (Windows)
.\venu\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Install System Dependencies
If you're on Windows and encounter DLL errors (like `msvcp140.dll`), download and install the **Microsoft Visual C++ Redistributable 2015-2022**:
[Download Here](https://aka.ms/vs/17/release/vc_redist.x64.exe)

## How to Run

To run the program with the simple `python main.py` command, you must first **activate** your virtual environment:

1. **Activate the Environment**:
   ```powershell
   # In PowerShell
   .\venu\Scripts\Activate.ps1

   # OR in Command Prompt (CMD)
   .\venu\Scripts\activate.bat
   ```
   *(You will see `(venu)` appear at the start of your command prompt line.)*

2. **Start the Server**:
   ```bash
   python main.py
   ```
3. **Access the Website**: Open your web browser and go to:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Supported Formats
The application supports the following audio formats:
- `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`

## Troubleshooting
- **Model Loading Slow?**: The first time you run the project, or upon initial loading, it may take a minute to load the model file (~86MB).
- **ImportErrors?**: Make sure your virtual environment is activated and you've installed all requirements correctly.