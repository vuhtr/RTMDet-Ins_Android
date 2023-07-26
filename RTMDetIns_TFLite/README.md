# Android - TFLite - RTMDet Instance Segmentation

## Dependencies

```bash
dependencies {
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu-api:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
}
```

## Note

Currently, the class list file and model files are put in folder `app\src\main\assets\object_det`
- Model version: RTMDet-Ins Small 640 (TFLite float16)
- The TFLite model is not working well enough...
