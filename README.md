\# Real-Time Face Mask Detection 😷



A real-time face mask detection system built using \*\*TensorFlow, MobileNetV2, and OpenCV\*\*.



\## 🚀 Features

\- 3-class classification:

&nbsp; - With Mask

&nbsp; - Without Mask

&nbsp; - Incorrect Mask

\- Real-time webcam detection

\- SSD face detector (Caffe model)

\- Transfer learning using MobileNetV2



\## 🛠 Tech Stack

\- Python

\- TensorFlow / Keras

\- OpenCV

\- NumPy



\## 📂 Project Structure

real-time-face-mask-detection/

│

├── detect\_mask.py                       # Real-time mask detection script

├── real-time-face-mask-detection.ipynb  # Model training notebook

├── mask\_detector\_finetuned\_3class.keras # Trained classification model

├── deploy.prototxt                      # Face detector configuration

├── res10\_300x300\_ssd\_iter\_140000.caffemodel  # Pretrained SSD face model

├── requirements.txt                     # Project dependencies

├── .gitignore

└── README.md



\## ▶️ How to Run



1\. Install dependencies:

\- pip install -r requirements.txt





2\. Run detection:

\- python detect\_mask.py



Press `Q` to quit.



\## 📌 Model Architecture

\- Base Model: MobileNetV2 (ImageNet pretrained)

\- Fine-tuned for 3-class mask detection

\- Input size: 224x224



---



\## 👨‍💻 Author

Siddhartha Khatri

