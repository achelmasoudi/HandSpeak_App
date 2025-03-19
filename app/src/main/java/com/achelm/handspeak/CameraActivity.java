package com.achelm.handspeak;

import androidx.cardview.widget.CardView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;

import com.example.imagepro.R;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "MainActivity"; // Log tag for debugging
    private Mat mRgba; // Matrix to store RGBA camera frames
    private Mat mGray; // Matrix to store grayscale camera frames
    private CameraBridgeViewBase mOpenCvCameraView; // OpenCV camera view object
    private SignLanDetectorClass signLanDetectorClass; // Custom class for sign language detection
    private CardView addButton, clearButton, textSpeechBtn, spaceBtn; // UI elements
    private TextView changeTxt; // TextView for displaying recognized text

    // Callback for managing OpenCV library loading
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV is loaded successfully");
                    mOpenCvCameraView.enableView(); // Enable camera view on successful OpenCV load
                    break;
                }
                default: {
                    super.onManagerConnected(status); // Handle other statuses
                }
            }
        }
    };

    public CameraActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass()); // Debug log for activity instantiation
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE); // Hide title bar
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); // Keep screen awake
        getWindow().setNavigationBarColor(getResources().getColor(R.color.primary_color)); // Set navigation bar color

        int MY_PERMISSIONS_REQUEST_CAMERA = 0;

        // Request camera permission if not already granted
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(CameraActivity.this,
                    new String[]{Manifest.permission.CAMERA},
                    MY_PERMISSIONS_REQUEST_CAMERA);
        }

        setContentView(R.layout.activity_camera); // Set the activity layout

        // Initialize camera view and set visibility
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this); // Set camera view listener

        // Initialize UI elements
        addButton = findViewById(R.id.add_button);
        clearButton = findViewById(R.id.clear_button);
        changeTxt = findViewById(R.id.change_text);
        textSpeechBtn = findViewById(R.id.textSpeechBtn);
        spaceBtn = findViewById(R.id.spaceBtn);

        try {
            // Initialize the sign language detector with models and parameters
            signLanDetectorClass = new SignLanDetectorClass(CameraActivity.this, spaceBtn, clearButton, addButton, changeTxt,
                    textSpeechBtn, getAssets(), "hand_model.tflite", 300,
                    "sign_lan.tflite", 96);

            Log.d("MainActivity", "Model is successfully loaded");
        } catch (IOException e) {
            Log.d("MainActivity", "Error loading the model");
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Initialize OpenCV in debug mode
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV initialization successful");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d(TAG, "OpenCV initialization failed, trying async initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        // Disable camera view to release resources
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        // Release camera resources when activity is destroyed
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        // Initialize matrices for RGBA and grayscale frames
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release(); // Release the RGBA matrix
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        // Capture RGBA and grayscale frames from camera input
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        // Process the frame through the sign language detection model
        return signLanDetectorClass.recognizeImage(mRgba);
    }
}