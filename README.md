# AI-Based Real-Time Sign Language to Voice Translator

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)

## 📌 Abstract
The **AI-Based Real-Time Sign Language to Voice Translator** is designed to bridge the communication gap between the hearing-impaired community and non-signers. This system leverages **Computer Vision** and **Deep Learning** to interpret hand gestures captured via a webcam and translates them into audible speech and text in real-time.

Using **MediaPipe** for hand landmark detection and a **Convolutional Neural Network (CNN)** for gesture classification, the model achieves robust recognition of static sign language alphabets (ASL) even under varying lighting and background conditions. The recognized text is then converted into speech using the `pyttsx3` library.

## 🚀 Key Features
* **Real-Time Detection:** Instantly detects hand gestures using a standard webcam.
* **Robust Recognition:** Uses MediaPipe for accurate hand landmarking and CNN for classification.
* **Text-to-Speech (TTS):** Converts recognized gestures into spoken words using `pyttsx3`.
* **User-Friendly GUI:** Interactive interface built with Python Tkinter.
* **Offline Capability:** Does not require an active internet connection for recognition.

## 🛠️ Tech Stack
* **Language:** Python 3.9
* **Computer Vision:** OpenCV (`cv2`), MediaPipe
* **Deep Learning:** TensorFlow, Keras (CNN Architecture)
* **Text-to-Speech:** pyttsx3
* **Interface:** Tkinter

## ⚙️ Installation & Setup

### Prerequisites
Ensure you have **Python 3.9** or higher installed.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/sign-language-translator.git](https://github.com/your-username/sign-language-translator.git)
cd sign-language-translator
