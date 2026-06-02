import os

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = r"D:\cnn\face_detection_DP_Project\face_mask_images_dataset"

MODEL_PATH = r"D:\cnn\face_detection_DP_Project\models\mask_detector_model.h5"

os.makedirs("models", exist_ok=True)