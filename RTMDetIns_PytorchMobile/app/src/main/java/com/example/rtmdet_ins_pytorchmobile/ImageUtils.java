package com.example.rtmdet_ins_pytorchmobile;

import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;

public class ImageUtils {

    static class PaddedImage {
        public Bitmap image;
        public int padX;
        public int padY;

        public PaddedImage(Bitmap image, int padX, int padY) {
            this.image = image;
            this.padX = padX;
            this.padY = padY;
        }
    }

    public static Bitmap resizeKeepRatio(Bitmap image, int maxSize) {
        int width = image.getWidth();
        int height = image.getHeight();
        if (width <= maxSize && height <= maxSize) {
            return image;
        }

        int newWidth = maxSize;
        int newHeight = maxSize;
        if (width > height) {
            newHeight = (int) ((float)maxSize * (float)height / (float)width);
        } else {
            newWidth = (int) ((float)maxSize * (float)width / (float)height);
        }

        return Bitmap.createScaledBitmap(image, newWidth, newHeight, true);
    }

    public static PaddedImage pad(Bitmap image, int maxSize, int padValue) {
        int width = image.getWidth();
        int height = image.getHeight();
        if (width >= maxSize && height >= maxSize) {
            return new PaddedImage(image, 0, 0);
        }

        int padX = (maxSize - width) / 2;
        int padY = (maxSize - height) / 2;

        Bitmap mutableBm = image.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap newImage = Bitmap.createBitmap(maxSize, maxSize, mutableBm.getConfig());
        Canvas canvas = new Canvas(newImage);
        canvas.drawRGB(padValue, padValue, padValue);
        canvas.drawBitmap(mutableBm, padX, padY, null);

        return new PaddedImage(newImage, padX, padY);
    }

//    public static FloatBuffer normalizeImage(@NotNull Bitmap image, float[] mean, float[] std) {
    public static float[] normalizeImage(@NotNull Bitmap image, float[] mean, float[] std) {
        int width = image.getWidth();
        int height = image.getHeight();
        int bufferSize = 3 * width * height;

        int stride = width * height;
        int[] bmpData = new int[stride];
        image.getPixels(bmpData, 0, width, 0, 0, width, height);

//        FloatBuffer normalizedResult = FloatBuffer.allocate(bufferSize);
//        normalizedResult.rewind();
        float[] normalizedResult = new float[bufferSize];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                int pixelValue = bmpData[idx];
//                normalizedResult.put(idx, ((float) (pixelValue >> 16 & 255) - mean[0]) / std[0]);
//                normalizedResult.put(idx + stride, ((float) (pixelValue >> 8 & 255) - mean[1]) / std[1]);
//                normalizedResult.put(idx + stride * 2, ((float) (pixelValue & 255) - mean[2]) / std[2]);
                normalizedResult[idx] = ((float) (pixelValue >> 16 & 255) - mean[0]) / std[0];
                normalizedResult[idx + stride] = ((float) (pixelValue >> 8 & 255) - mean[1]) / std[1];
                normalizedResult[idx + stride * 2] = ((float) (pixelValue & 255) - mean[2]) / std[2];
            }
        }
//        normalizedResult.rewind();
        return normalizedResult;
    }

    private static Bitmap myDecodeBitmap(Context context, Uri selectedImage) {
        Bitmap bm = null;
        ContentResolver contentResolver = context.getContentResolver();
        try {
            if(Build.VERSION.SDK_INT < 28) {
                bm = MediaStore.Images.Media.getBitmap(contentResolver, selectedImage);
            } else {
                ImageDecoder.Source source = ImageDecoder.createSource(contentResolver, selectedImage);
                bm = ImageDecoder.decodeBitmap(source);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return bm;
    }

    public static Bitmap drawDetectionResult(ObjectDetector.DetectionResult result, Bitmap inputImage, int[] boxColor, int[] maskColor, float maskOpacity) {
        ArrayList<Float> scores = result.scores;
        ArrayList<int[]> boxes = result.boxes;
        ArrayList<String> labels = result.labels;
        ArrayList<Bitmap> masks = result.masks;

        Bitmap outputBm = inputImage.copy(Bitmap.Config.ARGB_8888, true);


        for (int i = 0; i < scores.size(); i++) {
            int[] box = boxes.get(i);
            String label = labels.get(i);
            float score = scores.get(i);

            Canvas canvas = new Canvas(outputBm);
            Paint paint = new Paint();
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(3);
            paint.setColor(Color.rgb(boxColor[0], boxColor[1], boxColor[2]));

            // draw box
            int x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
            canvas.drawRect(x1, y1, x2, y2, paint);

            // draw mask
            Bitmap mask = masks.get(i); // same size as box
            for (int y = 0; y < mask.getHeight(); y++) {
                for (int x = 0; x < mask.getWidth(); x++) {
                    int pixel = mask.getPixel(x, y);
                    int r = Color.red(pixel), g = Color.green(pixel), b = Color.blue(pixel);
                    if (r > 0) {
                        r = (int) (r * maskColor[0]);
                        g = (int) (g * maskColor[1]);
                        b = (int) (b * maskColor[2]);
                        mask.setPixel(x, y, Color.argb((int) (maskOpacity * 255), r, g, b));
                    } else {
                        mask.setPixel(x, y, outputBm.getPixel(x + x1, y + y1));
                    }
                }
            }
            canvas.drawBitmap(mask, x1, y1, null);

            // write label and score
            paint.setStyle(Paint.Style.FILL);
            paint.setTextSize(20);
            paint.setColor(Color.rgb(255, 0, 0));
//            String text = label + ": " + score;
            String text = label;
            canvas.drawText(text, x1, y1 - 10, paint);
        }

        return outputBm;
    }


    public static Bitmap getImageFromPickerIntent(Context context, Intent imageReturnedIntent) {
        Bitmap bm = null;
        Uri selectedImage;
        boolean isCamera = (imageReturnedIntent == null || imageReturnedIntent.getData() == null);

        if (isCamera) {     /** CAMERA **/
            bm = (Bitmap) imageReturnedIntent.getExtras().get("data");
            String path = MediaStore.Images.Media.insertImage(context.getContentResolver(), bm, "Image123", null);
            selectedImage = Uri.parse(path);
        } else {            /** ALBUM **/
            selectedImage = imageReturnedIntent.getData();
        }

        bm = myDecodeBitmap(context, selectedImage);
        return bm;
    }

}


