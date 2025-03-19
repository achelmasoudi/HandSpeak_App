package com.achelm.handspeak;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;

import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;

import com.example.imagepro.R;

import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity {
    // Static block to initialize OpenCV and log the result of the initialization
    static {
        if (OpenCVLoader.initDebug()) {
            Log.d("MainActivity: ", "OpenCV is loaded successfully");
        } else {
            Log.d("MainActivity: ", "OpenCV failed to load");
        }
    }

    private CardView typeWithYourHandsBtn; // Button for navigating to CameraActivity

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Set the status bar to be translucent for a modern UI effect
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_TRANSLUCENT_STATUS);
        // Set the navigation bar color to black for consistency
        getWindow().setNavigationBarColor(Color.parseColor("#000000"));

        setContentView(R.layout.activity_main); // Set the layout for this activity

        // Initialize and set up the button for launching CameraActivity
        typeWithYourHandsBtn = findViewById(R.id.mainActivity_typeWithYourHandsBtn);
        typeWithYourHandsBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Create an intent to start CameraActivity and clear task stack
                Intent intent = new Intent(MainActivity.this, CameraActivity.class);
                intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
                startActivity(intent); // Launch the CameraActivity
            }
        });
    }
}
