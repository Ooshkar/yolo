if __name__ == "__main__":
    from ultralytics import YOLO

    # Load a pretrained YOLO26n (which is in best.pt alr) model
    model = YOLO("best/best/weights/best.pt")

    # Perform detection on a video
    results = model("2026-01-17 19-52-59.mp4",  # path to your video
                    save=True,          # save annotated video
                    show=True,          # optionally display frames while processing
                    conf=0.8)          # confidence threshold (0.25 = 25%)