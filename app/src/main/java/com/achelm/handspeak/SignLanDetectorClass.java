package com.achelm.handspeak;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.cardview.widget.CardView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

public class SignLanDetectorClass {
    private Interpreter interpreter; // Model for object detection
    private Interpreter interpreter2; // Model for classification
    private int INPUT_SIZE; // Input size for detection model
    private int classificationInputSize; // Input size for classification model
    private GpuDelegate gpuDelegate; // GPU delegate for better performance
    private int height = 0; // Frame height
    private int width = 0; // Frame width
    private String finalText = ""; // Stores the full detected text
    private String currentText = ""; // Stores the current detected character
    private TextToSpeech textToSpeech; // For text-to-speech functionality


    // Constructor to initialize models, TTS, and UI interactions
    SignLanDetectorClass(Context context, CardView spaceBtn , CardView clearButton , CardView addButton , TextView changeText , CardView textSpeech , AssetManager assetManager, String modelPath, int inputSize , String classificationModel , int classificationInputSize) throws IOException {
        INPUT_SIZE = inputSize;
        this.classificationInputSize = classificationInputSize;

        // Load detection model with GPU delegate
        Interpreter.Options options = new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);

        options.setNumThreads(4); // Optimize for device performance
        interpreter = new Interpreter(loadModelFile(assetManager,modelPath),options);

        // Load classification model ( Sign Lan )
        Interpreter.Options options2 = new Interpreter.Options();
        options2.setNumThreads(2);
        interpreter2 = new Interpreter(loadModelFile(assetManager, classificationModel), options2);

        // Clear the last character in finalText
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!finalText.isEmpty()) {
                    // Remove the last character from the finalText
                    finalText = finalText.substring(0, finalText.length() - 1);
                    changeText.setText(finalText);
                } else {
                    Toast.makeText(context, "No letters to clear!", Toast.LENGTH_SHORT).show();
                }
            }
        });

        // Add current detected character to finalText
        addButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (finalText.isEmpty()) {
                    finalText = currentText.toUpperCase(); // Capitalize the first letter if finalText is empty
                } else {
                    finalText += currentText.toLowerCase(); // Append in lowercase
                }
                changeText.setText(finalText);
            }
        });

        // Initialize Text-to-Speech engine
        textToSpeech = new TextToSpeech(context, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                // Check if initialization is successful
                if (status != TextToSpeech.ERROR) {
                    textToSpeech.setLanguage(new Locale("tr")); // Set language to Turkish
                }
            }
        });
        // Speak out the text in finalText
        textSpeech.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // when this button is clicked read the text
                textToSpeech.speak(finalText, TextToSpeech.QUEUE_FLUSH, null, null);
            }
        });

        // Add a space to finalText
        spaceBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                finalText += " ";
                changeText.setText(finalText);
            }
        });
    }

    // Load TensorFlow Lite model file
    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset =fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }

    // Process the input image for detection and classification
    public Mat recognizeImage(Mat mat_image){
        // Rotate the frame for portrait mode
        Mat rotated_mat_image = new Mat();
        Mat a = mat_image.t(); // Transpose the image for rotation
        Core.flip(a, rotated_mat_image, 1); // Flip horizontally
        a.release();

        // Convert Mat to Bitmap
        Bitmap bitmap = Bitmap.createBitmap(rotated_mat_image.cols(), rotated_mat_image.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image, bitmap);

        // Define height and width of the frame
        height = bitmap.getHeight();
        width = bitmap.getWidth();


        // Resize Bitmap to match the model's input size
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        // Convert the scaled Bitmap to ByteBuffer, as required by the model
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);

        // Define input and output for the detection model
        Object[] input = new Object[1];
        input[0] = byteBuffer;


        Map<Integer, Object> output_map = new TreeMap<>();
        // Allocate arrays to hold output results: boxes, scores, and classes
        float[][][] boxes = new float[1][10][4]; // 10: top 10 detections, 4: bounding box coordinates
        float[][] scores = new float[1][10]; // Confidence scores for each detection
        float[][] classes = new float[1][10]; // Class IDs for each detection

        // Map output arrays
        output_map.put(0, boxes);
        output_map.put(1, classes);
        output_map.put(2, scores);

        // Perform inference using the detection model
        interpreter.runForMultipleInputsOutputs(input, output_map);

        // Extract detection results from the output_map
        Object value = output_map.get(0); // Bounding boxes
        Object object_class = output_map.get(1); // Classes
        Object score = output_map.get(2); // Confidence scores

        // Loop through each detected object (maximum 10 detections)
        for (int i = 0; i < 10; i++) {
            // Extract class and score for the detected object
            float class_value = (float) Array.get(Array.get(object_class, 0), i);
            float score_value = (float) Array.get(Array.get(score, 0), i);


            // Define a confidence threshold for valid detections
            if (score_value > 0.5) { // Adjust this threshold based on model accuracy
                Object box1 = Array.get(Array.get(value, 0), i);

                // Extract and scale bounding box coordinates
                float y1 = (float) Array.get(box1, 0) * height;
                float x1 = (float) Array.get(box1, 1) * width;
                float y2 = (float) Array.get(box1, 2) * height;
                float x2 = (float) Array.get(box1, 3) * width;

                // Ensure coordinates are within frame boundaries
                if(y1 < 0) { y1 = 0; }
                if(x1 < 0) { x1 = 0; }
                if(x2 > width) { x2 = width; }
                if(y2 > height) { y2 = height; }

                // Define the bounding box dimensions
                float w1 = x2 - x1;
                float h1 = y2 - y1;

                // (x1,y1) is the start point of hand
                // (x2,y2) is the end point of hand

                // Crop the detected region from the frame
                Rect cropped_roi = new Rect((int) x1, (int) y1, (int) w1, (int) h1);
                Mat cropped = new Mat(rotated_mat_image, cropped_roi).clone();

                // Convert cropped region to Bitmap for classification
                Bitmap bitmap1 = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped, bitmap1);

                // Resize cropped Bitmap to match the classification model's input size ( classification input size = 96 )
                Bitmap scaledBitmap1 = Bitmap.createScaledBitmap(bitmap1, classificationInputSize, classificationInputSize, false);
                // Convert to ByteBuffer for classification
                ByteBuffer byteBuffer1 = convertBitmapToByteBuffer1(scaledBitmap1);

                // Run classification model to get predicted sign
                float[][] outputClassValue = new float[1][1];
                interpreter2.run(byteBuffer1, outputClassValue);

                // if you want to outputClassValue
                Log.d("SignLanDetectorClass", "outputClassValue: "+ outputClassValue[0][0]);

                // Map outputClassValue to alphabetic representation
                String signValue = getAlphabets(outputClassValue[0][0]);
                currentText = signValue; // Update the detected text ( so if for ex detected A letter will be currentText = "A" )

                // Display detected sign on the frame
                //              input/output             text               start point             font size
                Imgproc.putText(rotated_mat_image, signValue, new Point(x1 + 10, y1 + 40), 4, 1.5, new Scalar(0, 0, 255, 255), 2);
                // Draw bounding box
                Imgproc.rectangle(rotated_mat_image, new Point(x1, y1), new Point(x2, y2), new Scalar(255, 49, 49), 2);
            }

        }

        // Rotate the frame back to its original orientation
        Mat b = rotated_mat_image.t();
        Core.flip(b, mat_image, 0);
        b.release();

        // Return the processed frame
        return mat_image;
    }

    // Convert bitmap to ByteBuffer for detection model
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int quant=1;
        int size_images = INPUT_SIZE;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    byteBuffer.putFloat((((val >> 16) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val) & 0xFF))/255.0f);
                }
            }
        }
        return byteBuffer;
    }

    // Convert bitmap to ByteBuffer for classification model
    private ByteBuffer convertBitmapToByteBuffer1(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int quant=1;
        int size_images = classificationInputSize;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;
        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    byteBuffer.putFloat((((val >> 16) & 0xFF)));
                    byteBuffer.putFloat((((val >> 8) & 0xFF)));
                    byteBuffer.putFloat((((val) & 0xFF)));
                }
            }
        }
        return byteBuffer;
    }

    private String getAlphabets(float signVal) {
        String val = "";
        if(signVal >= -0.5 & signVal < 0.5) {  // -0.5 < signVal < 0.5
            val = "A";
        }
        else if(signVal >= 0.5 & signVal < 1.5) {
            val = "B";
        }
        else if(signVal >= 1.5 & signVal < 2.5) {
            val = "C";
        }
        else if(signVal >= 2.5 & signVal < 3.5) {
            val = "D";
        }
        else if(signVal >= 3.5 & signVal < 4.5) {
            val = "E";
        }
        else if(signVal >= 4.5 & signVal < 5.5) {
            val = "F";
        }
        else if(signVal >= 5.5 & signVal < 6.5) {
            val = "G";
        }
        else if(signVal >= 6.5 & signVal < 7.5) {
            val = "H";
        }
        else if(signVal >= 7.5 & signVal < 8.5) {
            val = "I";
        }
        else if(signVal >= 8.5 & signVal < 9.5) {
            val = "J";
        }
        else if(signVal >= 9.5 & signVal < 10.5) {
            val = "K";
        }
        else if(signVal >= 10.5 & signVal < 11.5) {
            val = "L";
        }
        else if(signVal >= 11.5 & signVal < 12.5) {
            val = "M";
        }
        else if(signVal >= 12.5 & signVal < 13.5) {
            val = "N";
        }
        else if(signVal >= 13.5 & signVal < 14.5) {
            val = "O";
        }
        else if(signVal >= 14.5 & signVal < 15.5) {
            val = "P";
        }
        else if(signVal >= 15.5 & signVal < 16.5) {
            val = "Q";
        }
        else if(signVal >= 16.5 & signVal < 17.5) {
            val = "R";
        }
        else if(signVal >= 17.5 & signVal < 18.5) {
            val = "S";
        }
        else if(signVal >= 18.5 & signVal < 19.5) {
            val = "T";
        }
        else if(signVal >= 19.5 & signVal < 20.5) {
            val = "U";
        }
        else if(signVal >= 20.5 & signVal < 21.5) {
            val = "V";
        }
        else if(signVal >= 21.5 & signVal < 22.5) {
            val = "W";
        }
        else if(signVal >= 22.5 & signVal < 23.5) {
            val = "X";
        }
        else if(signVal >= 23.5 & signVal < 24.5) {
            val = "Y";
        }
        else
            val = "Z";
        return val;
    }
}
