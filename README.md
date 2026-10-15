# Deep Learning for Detection and Classification of Excavated Archaeological Artefacts

MSc dissertation, University of Bradford (2026).

Archaeologists photograph thousands of finds during an excavation, and sorting them by type and material is slow manual work. This project builds a two-stage pipeline for photographs of finds from the Al-Baleed site: a YOLO detector first finds and labels each object, then a CNN classifies what material it is made of.

![End-to-end example](figures/example-full-pipeline-detection-and-classification-1.png)

## Pipeline

![Object detection workflow](figures/object-detection-workflow.jpg)

1. **Data preparation** ([notebook](notebooks/01_data_preparation.ipynb)): reconcile three sources (Excel catalogue labels, cropped images and original photographs), clean more than 30 free-text material labels into 4 material classes (glass, metal, pottery, misc), and build stratified train, validation and test splits.
2. **Detection benchmark** ([notebook](notebooks/02_detection_benchmark.ipynb)): train and compare six YOLO models (YOLOv8 n/s/m and YOLO11 n/s/m) on 8 object classes, then tune the best one with 5-fold cross-validation.
3. **Material classification** ([notebook](notebooks/03_classification.ipynb)): compare a dense network, a custom CNN, GoogLeNet, SqueezeNet and EfficientNet (PyTorch), using class weights to handle imbalance.
4. **End-to-end pipeline** ([notebook](notebooks/04_end_to_end_pipeline.ipynb)): detection, then cropping, then material classification on unseen photographs.

## Dataset

| Sample images | Artefacts by material class |
|---|---|
| ![Sample dataset](figures/image-of-sample-dataset.png) | ![Artefacts by material](figures/example-artefacts-by-material-class.png) |

| Annotated bounding box | Scale bar handling (original vs crop) | Stratified split |
|---|---|---|
| ![Bounding box](figures/image-with-bounding-box.png) | ![Scale bar](figures/scale-bar-illustration-original-vs-crop.png) | ![Stratified split](figures/stratified-split-illustration.png) |

The image dataset belongs to the excavation project and is not included in this repo.

## Results

### Object detection (test set)

| Model | mAP@50 | mAP@50:95 | Precision | Recall |
|---|---|---|---|---|
| **YOLOv8s** | **0.667** | **0.633** | 0.687 | 0.575 |
| YOLO11s | 0.650 | 0.609 | 0.489 | 0.681 |
| YOLOv8n | 0.629 | 0.608 | 0.565 | 0.643 |
| YOLO11n | 0.600 | 0.582 | 0.734 | 0.509 |
| YOLOv8m | 0.571 | 0.550 | 0.562 | 0.550 |
| YOLO11m | 0.436 | 0.374 | 0.236 | 0.603 |

The small models did better than the medium ones. With a dataset this size, the larger models overfit.

| YOLOv8s confusion matrix | YOLOv8s per-class F1 |
|---|---|
| ![YOLOv8s confusion matrix](figures/yolov8s-confusion-matrix.png) | ![Per-class F1](figures/yolov8s-per-class-performance-f1-score-on-test-set.png) |

| Training loss | Example detection |
|---|---|
| ![Loss](figures/loss-function.png) | ![Example detection](figures/example-object-detection-prediction.jpg) |

**Tuning with 5-fold cross-validation.** The baseline configuration scored 0.675 ± 0.071 mAP@50:95, against 0.655 ± 0.077 with strong augmentation. The gap is smaller than the fold-to-fold variation, so I kept the simpler baseline.

| Tuning process | Per-fold scores |
|---|---|
| ![Tuning process](figures/the-tuning-process.png) | ![Cross-validation](figures/cross-validation-per-fold-scores.png) |

### Material classification (test set)

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| **SqueezeNet** | **0.864** | 0.768 | 0.708 | **0.732** |
| EfficientNet | 0.840 | 0.691 | 0.726 | 0.703 |
| CNN (from scratch) | 0.753 | 0.617 | 0.616 | 0.614 |
| GoogLeNet | 0.704 | 0.633 | 0.669 | 0.630 |
| Dense network | 0.160 | 0.255 | 0.325 | 0.122 |

![Model comparison](figures/classification-model-comparison-ranked-by-macro-f1.png)

![Per-class F1 across the five models](figures/per-class-f1-accross-the-five-models.png)

![SqueezeNet architecture](figures/squeeznet-classification-architecture.jpg)

<details>
<summary>Confusion matrices for every classifier</summary>

![All confusion matrices](figures/classification-confusion-matrices.png)

| SqueezeNet | EfficientNet | GoogLeNet |
|---|---|---|
| ![](figures/squeezenet-confusion-matrix.png) | ![](figures/efficientnet-confusion-matrix.png) | ![](figures/googlenet-confusion-matrix.png) |

| CNN | Dense network |
|---|---|
| ![](figures/cnn-confusion-matrix.png) | ![](figures/dense-network-confusion-matrix.png) |

</details>

### End-to-end predictions

| Pipeline output | Single detection |
|---|---|
| ![End to end](figures/end-to-end-exmaple-prediction.png) | ![Bangle](figures/example-prediction-bangle.png) |

## Repo structure

```
notebooks/   four notebooks, run in order
results/     benchmark tables, per-class metrics, CV ledger (CSV/JSON)
configs/     training arguments for the final YOLOv8s model
figures/     all charts and example outputs
```

## Running it

The notebooks were run on Google Colab with a T4 GPU. Set `DATA_DIR` to the folder holding the dataset (the first cell of each notebook reads it), then run the notebooks in order.

```bash
pip install -r requirements.txt
```

**Tech:** Python, PyTorch, torchvision, Ultralytics YOLO, scikit-learn, pandas, matplotlib
