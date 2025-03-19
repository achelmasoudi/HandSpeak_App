package com.achelm.handspeak;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;

import androidx.appcompat.app.AppCompatActivity;
import com.example.imagepro.R;

public class SplashScreenActivity extends AppCompatActivity {

    private static final int SPLASH_SPEED = 3500; // Delay duration for the splash screen in milliseconds

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splashscreen);

        // Change the navigation bar color to match the splash screen theme
        getWindow().setNavigationBarColor(getResources().getColor(R.color.splash_screen_color));

        // Delay the transition to the MainActivity for SPLASH_SPEED milliseconds
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                // Intent to navigate from SplashScreenActivity to MainActivity
                Intent intent = new Intent(SplashScreenActivity.this, MainActivity.class);
                // Clear any existing tasks to prevent returning to the splash screen
                intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
                startActivity(intent);
                finish();
            }
        }, SPLASH_SPEED);
    }
}