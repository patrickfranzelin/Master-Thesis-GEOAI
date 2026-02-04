from ultralytics import SAM

model = SAM("models/sam3_weights/sam3.pt")

def run_sam(image_path, inside_pts):
    labels = [1] * len(inside_pts)

    r = model.predict(
        source=str(image_path),
        points=inside_pts,
        labels=labels
    )

    if r[0].masks is None:
        return None

    return r[0].masks.data[0].cpu().numpy()
