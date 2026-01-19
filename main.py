
if __name__ == "__main__":
    from ultralytics import YOLO

    # Load a pretrained YOLO26n (which is in best.pt alr) model
    model = YOLO("best/best/weights/last.pt")

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data="dataset/data.yaml",  # Path to dataset configuration file
        epochs=1000,  # Number of training epochs
        imgsz=640,  # Image size for training
        device="0",
        batch=-1, # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        project="best",
        name="",
        exist_ok=True,
        resume=False,
        early_stopping=False
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("dataset/images/Screenshot_20260115_195301.png")  # Predict on an image
    results[0].show()  # Display results

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model


