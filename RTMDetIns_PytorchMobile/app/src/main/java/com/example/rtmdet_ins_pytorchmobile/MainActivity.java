package com.example.rtmdet_ins_pytorchmobile;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.Manifest;
import android.widget.ImageView;

import com.google.android.material.snackbar.Snackbar;


public class MainActivity extends AppCompatActivity {
    private ActivityResultLauncher<Intent> imagePickerActivityResultLauncher;
//    private static final int MY_CAMERA_REQUEST_CODE = 100;
    private static final int MY_GALLERY_REQUEST_CODE = 101;

    private static final int MAX_INPUT_SIZE = 1200;     // avoid OOM for large image

    //    Constants for Object Detection
    private static int INFER_SIZE;
    private static final int[] MASK_COLOR = {255, 0, 0};    // red
    private static final int[] BOX_COLOR = {0, 255, 0};     // green

    private ObjectDetector objectDetector;
    private ImageView inputImageView, outputImageView;
    private Button selectImageBtn, detectBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        AssetManager assetManager = getAssets();

        INFER_SIZE = 640;
        objectDetector = new ObjectDetector(assetManager, "object_det/rtmdetins_s_640.pth", "object_det/classes.txt",  INFER_SIZE, 0.325F, 0.2F);

        initViews();
        setupEvents();
    }

    private void initViews() {
        inputImageView = findViewById(R.id.inputImageView);
        outputImageView = findViewById(R.id.outputImageView);
        selectImageBtn = findViewById(R.id.selectImageBtn);
        detectBtn = findViewById(R.id.detectBtn);
    }

    private void setupEvents() {
        setupImagePicker();
        selectImageBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                runImagePicker();
            }
        });

        detectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Bitmap bm = null;
                try {
                    bm = ((BitmapDrawable)inputImageView.getDrawable()).getBitmap();
                } catch (Exception e) {
                    Snackbar.make(view, "Please select an image first", Snackbar.LENGTH_LONG).show();
                    return;
                }
                try {
                    ObjectDetector.DetectionResult result = objectDetector.infer(bm);
                    Bitmap outputBm = ImageUtils.drawDetectionResult(result, bm, BOX_COLOR, MASK_COLOR, 0.5f);
                    setOutputImage(outputBm);

                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        });
    }

    private void runImagePicker() {
//        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
//            requestPermissions(new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE);
//            return;
//        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (checkSelfPermission(Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.READ_MEDIA_IMAGES}, MY_GALLERY_REQUEST_CODE);
                return;
            }
            if (checkSelfPermission(Manifest.permission.READ_MEDIA_VIDEO) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.READ_MEDIA_VIDEO}, MY_GALLERY_REQUEST_CODE);
                return;
            }
        }
        else if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, MY_GALLERY_REQUEST_CODE);
            return;
        }

        // Pick image from gallery
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        imagePickerActivityResultLauncher.launch(intent);
    }

    private void setupImagePicker() {
        imagePickerActivityResultLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                new ActivityResultCallback<ActivityResult>() {
                    @Override
                    public void onActivityResult(ActivityResult result) {
                        if (result.getResultCode() == Activity.RESULT_OK) {
                            Intent data = result.getData();
                            Bitmap bitmap = ImageUtils.getImageFromPickerIntent(MainActivity.this, data);
                            bitmap = ImageUtils.resizeKeepRatio(bitmap, MAX_INPUT_SIZE);
                            setInputImage(bitmap);
                        }
                    }
                }
        );
    }

    private void setInputImage(Bitmap bitmap) {
        inputImageView.setImageBitmap(bitmap);
        outputImageView.setImageBitmap(null);
    }

    private void setOutputImage(Bitmap bitmap) {
        outputImageView.setImageBitmap(null);
        outputImageView.setImageBitmap(bitmap);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
//            case MY_CAMERA_REQUEST_CODE:
//                System.out.println(grantResults[0]);
//                if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
//                    Snackbar.make(findViewById(R.id.parent), "Can not access to camera", Snackbar.LENGTH_LONG).show();
//                } else {
//                    runImagePicker();
//                }
//                break;
            case MY_GALLERY_REQUEST_CODE:
                if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                    Snackbar.make(findViewById(R.id.parent), "Can not access to image gallery", Snackbar.LENGTH_LONG).show();
                } else {
                    runImagePicker();
                }
                break;

            default:
                break;
        }
    }

}