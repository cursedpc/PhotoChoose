# Bundled inference models

## YuNet face detector

- File: `face_detection_yunet_2023mar.onnx`
- Upstream: OpenCV Zoo, `models/face_detection_yunet`
- License: MIT (`YUNET_LICENSE`)
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`

PhotoChoose executes YuNet through OpenVINO CPU. OpenCV DNN is retained as an automatic
load-time and inference-time fallback.

## MediaPipe Face Mesh V2 landmarks

- File: `face_landmarks_detector.tflite`
- Upstream: Google AI Edge MediaPipe `face_landmarker.task`, float16 release
- Source bundle: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
- License: Apache License 2.0 (`MEDIAPIPE_LICENSE`)
- Input: RGB `1×256×256×3`, normalized to 0–1
- Output: 478 three-dimensional face landmarks
- SHA-256: `c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5`

PhotoChoose computes eye aperture from six eyelid landmarks per eye. Ratios at or below
0.14 are treated as confidently closed, ratios at or above 0.19 as open, and the band
between them as uncertain. These deliberately conservative thresholds were checked against
the photographer's local development corpus and should be recalibrated as that corpus grows.
OpenVINO CPU imports the bundled TFLite graph directly; LiteRT is retained as the automatic
load-time and inference-time fallback.

## NanoDet general object detector

- File: `object_detection_nanodet_2022nov.onnx`
- Upstream: OpenCV Zoo, `models/object_detection_nanodet`
- Source: `https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet`
- License: Apache License 2.0 (`NANODET_LICENSE`)
- Input: RGB `1×3×416×416`, letterboxed and normalized
- Output: COCO-80 object labels and bounding boxes
- SHA-256: `4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186`

PhotoChoose runs NanoDet through OpenVINO CPU, with OpenCV DNN as an automatic load-time and
inference-time fallback, and measures focus inside each detected region. If none of its 80
classes is detected, a model-free spectral-saliency fallback may
identify one unlabeled prominent region. Saliency-only evidence can influence relative Best
ranking but cannot independently send a photograph to Review.

Non-person detections below 42% confidence and person detections below 36% are discarded.
The remaining class name is still only the model's most likely COCO-80 guess; PhotoChoose
uses the region and focus evidence independently of whether that name is semantically exact.

The models run locally. PhotoChoose performs no model download at runtime.
