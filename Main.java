package com.learn;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
public class Main {
    public static void main(String[] args){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat image = Imgcodecs.imread("images/img.jpg");

        if (image.empty()) {
            System.out.println("Error: Could not read the image.");
            return;
        }
        detectAndSave(image);
    }
    private static void detectAndSave(Mat image) {
        MatOfRect faces = new MatOfRect(); // store more than one face
        Mat grayFrame = new Mat();
        Imgproc.cvtColor(image, grayFrame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(grayFrame, grayFrame);
        int height = grayFrame.rows();
        int absoluteFaceSize = 0;
        if (Math.round(height * 0.2f) > 0) {
            absoluteFaceSize = Math.round(height * 0.2f);
        }
        CascadeClassifier faceCascade = new CascadeClassifier();
        if (!faceCascade.load("data/haarcascade_frontalface_alt2.xml")) {
            System.out.println("Error: Could not load the face cascade classifier.");
            return;
        }
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        Rect[] faceArray = faces.toArray();
        for (int i = 0; i < faceArray.length; i++) {
            Imgproc.rectangle(image, faceArray[i], new Scalar(0, 0, 255), 3);
        }
        Imgcodecs.imwrite("images/output.jpg", image);
    }
}
