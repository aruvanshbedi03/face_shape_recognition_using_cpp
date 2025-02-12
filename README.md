# 🚀 Face Detection using OpenCV (Haarcascade & DNN)

## 🔹 Overview  
This project implements **real-time face detection** using OpenCV in C++. It offers two approaches:  

1. **Haarcascade Classifier** (Traditional, Fast, but Less Accurate)  
   - Uses pre-trained XML classifiers to detect faces based on pattern recognition.  
   - Works well in good lighting but has many false detections.
   - ![Screenshot 2025-02-12 102027](https://github.com/user-attachments/assets/86a7bcfb-9ef4-4179-8c69-6b2f4edc64ff)


2. **Deep Learning DNN Model** (More Accurate but Requires a Pre-trained Model)  
   - Uses a TensorFlow-based pre-trained model (`opencv_face_detector_uint8.pb`).  
   - Can detect faces in different lighting conditions and angles with better precision.
     ![Screenshot 2025-02-12 104904](https://github.com/user-attachments/assets/9f864d57-8737-4fe1-b8d2-776e0a320618)


The model **captures frames from a webcam**, processes them, detects faces, and displays the output in real time.  

---

## 📌 How It Works  

![Screenshot 2025-02-12 102135](https://github.com/user-attachments/assets/68b66cd6-8b8e-4cd3-8637-35acae9e3ca6)

1️⃣ **Capture Video**  
   - The program initializes the webcam (`cv::VideoCapture`) to capture real-time video frames.  

2️⃣ **Preprocess Frames**  
   - Converts each frame to **grayscale** using `cv::cvtColor()`.  
   - Resizes the frame for **faster processing**.  
   - Normalizes brightness using `cv::equalizeHist()`.  

3️⃣ **Face Detection**  
   - **Haarcascade**: Uses `CascadeClassifier.detectMultiScale()` for face detection.  
   - **DNN Model**: Loads a TensorFlow model (`opencv_face_detector_uint8.pb`) and performs deep learning-based inference.  

4️⃣ **Draw Bounding Boxes**  
   - Detected faces are highlighted using `cv::rectangle()` or `cv::circle()`.  
   - If using Haarcascade, **eye detection** is also performed.  

5️⃣ **Display Output**  
   - The processed frame is displayed using `cv::imshow()`.  
   - The program runs in a loop until the user presses **'q' or 'Esc'**.  

---

## 📌 Installation Steps  
 

#### 📌 Install OpenCV on Linux  
```bash
sudo apt update
sudo apt install libopencv-dev python3-opencv

## 🔹 Compiling & Running
1️⃣ **Compile the Code**

g++ face_detection.cpp -o face_detect pkg-config --cflags --libs opencv4

2️⃣ Run the Program

./face_detect

🔹 Troubleshooting
🚨 "Can't open model file" Error?

    Ensure the downloaded files are in the correct path.
    Update the file paths in your code:

    string modelPath = "/full/path/to/opencv_face_detector_uint8.pb";
    string configPath = "/full/path/to/opencv_face_detector.pbtxt";

🚨 "Random circles" Issue?

    Haarcascade is not very accurate; switch to the DNN model for better results.

🚨 "Camera not opening"?

    Ensure your webcam is properly connected.
    Try changing capture.open(0); to capture.open(1); if using an external webcam.

🚨 Compilation Error: opencv4 not found

    Run the following to check OpenCV version:

pkg-config --modversion opencv4

If OpenCV is missing, reinstall it using:

    sudo apt install libopencv-dev

🚨 ld: cannot open output file facednn: Is a directory Error

    The output file name conflicts with an existing directory. Delete or rename it:

    rm -r facednn
    g++ face_detection.cpp -o face_detect pkg-config --cflags --libs opencv4

🚨 terminate called after throwing an instance of 'cv::Exception' Error

    This usually happens when OpenCV cannot find the required files.
    Ensure the correct paths are set for .pb and .pbtxt files.
    Try running as administrator if using Windows.


👨‍💻 Created by Aruvansh SIngh Bedi
📌 Thapar University Student
