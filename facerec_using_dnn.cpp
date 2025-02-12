#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace dnn;

// Load Face Detector Model
string modelPath = "/home/aruvv/facednn_backup/opencv_face_detector_uint8.pb";
string configPath = "/home/aruvv/facednn_backup/opencv_face_detector.pbtxt";


dnn::Net net = readNetFromTensorflow(modelPath, configPath);

// Function to Detect and Draw Faces
void detectAndDraw(Mat& img, double scale);

int main() {
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cout << "Could not open camera" << endl;
        return -1;
    }

    cout << "Face Detection Started...." << endl;
    Mat frame;
    while (true) {
        capture >> frame;
        if (frame.empty()) break;

        detectAndDraw(frame, 1.0); 

        char c = (char)waitKey(10);
        if (c == 27 || c == 'q' || c == 'Q') break;
    }
    return 0;
}

// Face Detection Function
void detectAndDraw(Mat& img, double scale) {
    Mat blob;
    Size inpSize(300, 300);
    Scalar meanVal(104.0, 177.0, 123.0); 

    // Convert Image to DNN Format
    blobFromImage(img, blob, 1.0, inpSize, meanVal, true, false);
    net.setInput(blob);
    Mat detection = net.forward();

    // Get Detection Data
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++) {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > 0.6) {  // Set high confidence threshold
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * img.cols);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * img.rows);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * img.cols);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * img.rows);

            // Draw rectangle around the face
            rectangle(img, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
        }
    }
    imshow("Face Detection", img);
}
