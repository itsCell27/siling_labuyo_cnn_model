from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/segment/runs/labuyo_segmentation_v2/yolo26n_labuyo_leaf_seg_v2/weights/best.pt")

model.export(
    format="tflite",  # Export to TensorFlow Lite format
    imgsz=640,        # Image size for the model
)