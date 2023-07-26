package com.example.rtmdet_ins_pytorchmobile;
import android.content.Context;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Color;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;


public class ObjectDetector {
    // constant of current model family
    private static final int PAD_VAL = 114;
    private static final float NMS_BOX_IOU_THRES = 0.6F;
    private static final float BOX_IOU_THRES = 0.7F;
    private static final float MASK_IOU_THRES = 0.7F;
    private static final float OVERLAP_THRES = 0.8F;
    private static final float EPS = 1e-6F;

    private static final float[] MEAN = {103.53F, 116.28F, 123.675F};
    private static final float[] STD = {57.375F, 57.12F, 58.395F};

    // constant of post-processing
    private static final int BOX_THRES = 20;    // ignore too small boxes

    static class DetectionResult {
        public ArrayList<int[]> boxes;      // (n, 4) - format x1, y1, x2, y2
        public ArrayList<Bitmap> masks;    // (n,) - bitmap of mask corresponding to box (size of mask = size of box)
        public ArrayList<Float> scores;     // (n, ) - confidence score between 0 and 1
        public ArrayList<String> labels;    // (n, ) - class label

        public DetectionResult(ArrayList<int[]> boxes, ArrayList<Bitmap> masks, ArrayList<Float> scores, ArrayList<String> labels) {
            this.boxes = boxes;
            this.masks = masks;
            this.scores = scores;
            this.labels = labels;
        }
    }


    private Context context;
    private Resources resources;
    private HashMap<Integer, String> classMapping;
    private final int inferSize;      // input size of the model
    private final float commonThres;  // confidence threshold for common bounding box
    private final float personThres;  // confidence threshold for person (special case)

    private final AssetManager assetManager;
    private Module model;

    public ObjectDetector(AssetManager assetManager, String modelPath, String classPath, int inferSize, float commonThres, float personThres) {
        this.inferSize = inferSize;
        this.commonThres = commonThres;
        this.personThres = personThres;
        this.assetManager = assetManager;

        readClasses(classPath);
        try {
            createModel(modelPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void createModel(String modelPath) throws IOException {
        model = LiteModuleLoader.loadModuleFromAsset(assetManager, modelPath);
        // warm up
        float[] inputData = new float[inferSize * inferSize * 3];
        Tensor inputTensor = Tensor.fromBlob(inputData, new long[]{1, 3, inferSize, inferSize});
        IValue[] outputTensor = model.forward(IValue.from(inputTensor)).toTuple();
    }

    private void readClasses(String labelPath) {
        // read file
        InputStream inputStream = null;
        try {
            inputStream = assetManager.open(labelPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        // read lines
        HashMap<Integer, String> readClasses = new HashMap<>();
        int i = 0;
        try (java.util.Scanner scanner = new java.util.Scanner(inputStream)) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                readClasses.put(i, line);
                i++;
            }
        }
        classMapping = readClasses;
    }

    private static class PreprocessedImage {
        public float[] imageData;
        public int padX;
        public int padY;

        public PreprocessedImage(float[] imageData, int padX, int padY) {
            this.imageData = imageData;
            this.padX = padX;
            this.padY = padY;
        }
    }

    private PreprocessedImage preprocess(Bitmap image) {
        // Resize
        Bitmap resizedBm = ImageUtils.resizeKeepRatio(image, inferSize);

        // Pad
        ImageUtils.PaddedImage paddedImage = ImageUtils.pad(resizedBm, inferSize, PAD_VAL);
        Bitmap paddedBm = paddedImage.image;
        int padX = paddedImage.padX;
        int padY = paddedImage.padY;

        // Convert to float array
//        FloatBuffer imageData = ImageUtils.normalizeImage(paddedBm, MEAN, STD);
        float[] imageData = ImageUtils.normalizeImage(paddedBm, MEAN, STD);

        return new PreprocessedImage(imageData, padX, padY);
    }


    public DetectionResult infer(Bitmap inputBitmap) {
        long startTime = 0L;
        long endTime = 0L;
        long totalTime = 0L;

        int origWidth = inputBitmap.getWidth();
        int origHeight = inputBitmap.getHeight();

        ////////////////////////////////////////
        // Preprocessing
        startTime = System.currentTimeMillis();

        PreprocessedImage preprocessedImage = preprocess(inputBitmap);
        float[] inputData = preprocessedImage.imageData;
        int padX = preprocessedImage.padX;
        int padY = preprocessedImage.padY;

        // Create input tensor
        Tensor inputTensor = Tensor.fromBlob(inputData, new long[]{1, 3, inferSize, inferSize});

        endTime = System.currentTimeMillis();

        totalTime += (endTime - startTime);
        System.out.println("[LOG] 1. Pre-process time: " + (endTime - startTime) + "ms");

        ////////////////////////////////////////
        // Inference

        startTime = System.currentTimeMillis();
        IValue[] outputTensor = model.forward(IValue.from(inputTensor)).toTuple();
        endTime = System.currentTimeMillis();
        totalTime += (endTime - startTime);
        System.out.println("[LOG] 2. Inference time: " + (endTime - startTime) + "ms");

        ////////////////////////////////////////
        // Postprocessing

        // Extract results from session
        startTime = System.currentTimeMillis();
        long[] labels = outputTensor[1].toTensor().getDataAsLongArray();  // in shape (1 * n)
        float[] detsBuffer = outputTensor[0].toTensor().getDataAsFloatArray();  // in shape (1 * n * 5)
        int[][] boxes = new int[detsBuffer.length / 5][4];     // in shape (n, 4) - [x1, y1, x2, y2]
        float[] scores = new float[detsBuffer.length / 5];    // in shape (n)
        for (int i = 0; i < detsBuffer.length / 5; i++) {
            boxes[i] = new int[]{(int) detsBuffer[i * 5], (int) detsBuffer[i * 5 + 1], (int) detsBuffer[i * 5 + 2], (int) detsBuffer[i * 5 + 3]};
            scores[i] = detsBuffer[i * 5 + 4];
        }
        detsBuffer = null;
        byte[] masksBuffer = outputTensor[2].toTensor().getDataAsUnsignedByteArray();      // in shape (n * h * w)
        byte[][][] masks = new byte[masksBuffer.length / (inferSize * inferSize)][inferSize][inferSize];
        for (int i = 0; i < masksBuffer.length / (inferSize * inferSize); i++) {
            for (int j = 0; j < inferSize; j++) {
                for (int k = 0; k < inferSize; k++) {
                    masks[i][j][k] = (byte) Math.round(masksBuffer[i * inferSize * inferSize + j * inferSize + k]);
                }
            }
        }
        masksBuffer = null;
        endTime = System.currentTimeMillis();
        totalTime += (endTime - startTime);
        System.out.println("[LOG] 3. Extract result time: " + (endTime - startTime) + "ms");

        startTime = System.currentTimeMillis();
        DetectionResult result = postprocess(boxes, scores, labels, masks, origWidth, origHeight, padX, padY);
        endTime = System.currentTimeMillis();

        totalTime += (endTime - startTime);
        System.out.println("[LOG] 4. Post-process time: " + (endTime - startTime) + "ms");

        System.out.println("[LOG] Total time: " + totalTime + "ms");

        return result;
    }

    private DetectionResult postprocess(int[][] boxes, float[] scores, long[] labels, byte[][][] masks, int origWidth, int origHeight, int padX, int padY) {
        int n = boxes.length;
        boolean[] isSkipped = new boolean[n];

        // 1. Filter our low score boxes
        for (int i = 0; i < n; i++) {
            if (scores[i] >= commonThres || (labels[i] == 0 && scores[i] >= personThres))
                continue;
            isSkipped[i] = true;
        }

        // 2. NMS
        for (int i = 0; i < n; i++) {
            if (isSkipped[i])
                continue;

            int[] box1 = boxes[i];

            for (int j = i + 1; j < n; j++) {
                if (isSkipped[j] || labels[i] != labels[j])
                    continue;

                int[] box2 = boxes[j];

                float boxIoU = calcBoxIoU(box1, box2);

                if (boxIoU > NMS_BOX_IOU_THRES) {
                    if (scores[i] > scores[j]) {
                        isSkipped[j] = true;
                    } else {
                        isSkipped[i] = true;
                    }
                }
            }
        }

        // 3. Normalize box coordinates (between 0 and infer size - 1)
        for (int i = 0; i < n; i++) {
            if (isSkipped[i])
                continue;

            int x1 = boxes[i][0];
            int y1 = boxes[i][1];
            int x2 = boxes[i][2];
            int y2 = boxes[i][3];

            if (x1 >= x2 || y1 >= y2) {
                isSkipped[i] = true;
                continue;
            }

            x1 = Math.min(Math.max(padX, x1), inferSize - 1 - padX);
            y1 = Math.min(Math.max(padY, y1), inferSize - 1 - padY);
            x2 = Math.min(Math.max(padX, x2), inferSize - 1 - padX);
            y2 = Math.min(Math.max(padY, y2), inferSize - 1 - padY);

            boxes[i][0] = x1; boxes[i][1] = y1; boxes[i][2] = x2; boxes[i][3] = y2;
        }

        // 4. Reduce redundant boxes: moreover NMS + Merged overlapping boxes
        HashMap<Integer, ArrayList<Integer>> mergeDict = new HashMap<>();
        for (int i = 0; i < n; i++) {
            if (isSkipped[i]) {
                continue;
            }
            if (!mergeDict.containsKey(i)) {
                mergeDict.put(i, new ArrayList<Integer>());
            }
            int[] box1 = boxes[i];

            for (int j = i + 1; j < n; j++) {
                if (isSkipped[j]) {
                    continue;
                }
                if (!mergeDict.containsKey(j)) {
                    mergeDict.put(j, new ArrayList<Integer>());
                }
                int[] box2 = boxes[j];

                float boxIoU = calcBoxIoU(box1, box2);

                // crop 2 masks to the same shape
                int x1 = Math.min(box1[0], box2[0]);
                int y1 = Math.min(box1[1], box2[1]);
                int x2 = Math.max(box1[2], box2[2]);
                int y2 = Math.max(box1[3], box2[3]);
                // calculate mask IoU and overlap
                float maskInter = 0, mask1Area = 0, mask2Area = 0;
                for (int yy = y1; yy < y2; yy++) {
                    for (int xx = x1; xx < x2; xx++) {
                        byte mask1Value = masks[i][yy][xx];
                        byte mask2Value = masks[j][yy][xx];
                        mask1Area += mask1Value;
                        mask2Area += mask2Value;
                        maskInter += mask1Value * mask2Value;
                    }
                }

                float maskIoU = (float) maskInter / (mask1Area + mask2Area - maskInter + EPS);
                float mask1Overlap = (float) (maskInter / ((float) mask1Area + EPS));
                float mask2Overlap = (float) (maskInter / ((float) mask2Area + EPS));

                // check condition
                if ((boxIoU > BOX_IOU_THRES && maskIoU > MASK_IOU_THRES) ||
                        (labels[i] == labels[j] && (Math.max(mask1Overlap, mask2Overlap) > OVERLAP_THRES))) {
                    if (scores[i] > scores[j]) {
                        isSkipped[j] = true;
                        mergeDict.get(i).add(j); mergeDict.get(i).addAll(mergeDict.get(j));
                        mergeDict.remove(j);
                    } else {
                        isSkipped[i] = true;
                        mergeDict.get(j).add(i); mergeDict.get(j).addAll(mergeDict.get(i));
                        mergeDict.remove(i);
                    }
                }

                if (isSkipped[i]) {
                    break;
                }
            }
        }

        // 4. Merge masks
        for (int i = 0; i < n; i++) {
            if (isSkipped[i] || !mergeDict.containsKey(i)) {
                continue;
            }

            int[] curBox = boxes[i];
            byte[][] curMask = masks[i];

            for (int j = 0; j < mergeDict.get(i).size(); j++) {
                int idx = mergeDict.get(i).get(j);
                int[] box2 = boxes[idx];
                byte[][] mask2 = masks[idx];

                // merge box
                curBox[0] = Math.min(curBox[0], box2[0]);
                curBox[1] = Math.min(curBox[1], box2[1]);
                curBox[2] = Math.max(curBox[2], box2[2]);
                curBox[3] = Math.max(curBox[3], box2[3]);
                // merge mask
                for (int k = box2[1]; k <= box2[3]; k++) {
                    for (int l = box2[0]; l <= box2[2]; l++) {
                        curMask[k][l] = (byte) Math.max(curMask[k][l], mask2[k][l]);
                    }
                }
            }

            boxes[i] = curBox;
            masks[i] = curMask;
        }
        
        // 5. Refine boxes coordinates
        ArrayList<Float> finalScores = new ArrayList<>();
        ArrayList<int[]> finalBoxes = new ArrayList<>();
        ArrayList<String> finalLabels = new ArrayList<>();
        ArrayList<Bitmap> finalMasks = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            if (isSkipped[i]) {
                continue;
            }
            
            // actual box coordinates
            int x1 = boxes[i][0];
            int y1 = boxes[i][1];
            int x2 = boxes[i][2];
            int y2 = boxes[i][3];
            int actualX1 = (int) ((x1 - padX) / (float) (inferSize - padX * 2) * origWidth);
            int actualY1 = (int) ((y1 - padY) / (float) (inferSize - padY * 2) * origHeight);
            int actualX2 = (int) ((x2 - padX) / (float) (inferSize - padX * 2) * origWidth);
            int actualY2 = (int) ((y2 - padY) / (float) (inferSize - padY * 2) * origHeight);
            // check box size
            if ((actualX2 - actualX1 + 1) + (actualY2 - actualY1 + 1) < BOX_THRES)
                continue;

            // crop current mask (H x W) to final mask (same size with box)
            int maskHeight = y2 - y1;
            int maskWidth = x2 - x1;
            Bitmap maskBitmap = Bitmap.createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888);
            int[] binValues = new int[maskWidth * maskHeight];
            int idx = 0;
            for (int j = 0; j < maskHeight; j++) {
                for (int k = 0; k < maskWidth; k++) {
                    int val = Math.round(masks[i][y1 + j][x1 + k]);
                    binValues[idx++] = Color.rgb(val, val, val);
                }
            }
            int maskNewWidth = actualX2 - actualX1;
            int maskNewHeight = actualY2 - actualY1;
            maskBitmap.setPixels(binValues, 0, maskWidth, 0, 0, maskWidth, maskHeight);
            Bitmap actualMaskBitmap = Bitmap.createScaledBitmap(maskBitmap, maskNewWidth, maskNewHeight, false);

            finalBoxes.add(new int[]{actualX1, actualY1, actualX2, actualY2});
            finalMasks.add(actualMaskBitmap);
            finalScores.add(scores[i]);
            finalLabels.add(classMapping.get((int)labels[i]));
        }
        
        return new DetectionResult(finalBoxes, finalMasks, finalScores, finalLabels);
    }

    private float calcBoxIoU(int[] box1, int[] box2) {
        int x1 = Math.max(box1[0], box2[0]);
        int y1 = Math.max(box1[1], box2[1]);
        int x2 = Math.min(box1[2], box2[2]);
        int y2 = Math.min(box1[3], box2[3]);
        float inter = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
        if (inter == 0) {
            return 0;
        }
        float area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
        float area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
        return inter / (area1 + area2 - inter);
    }
}
