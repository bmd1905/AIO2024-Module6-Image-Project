import logging
import os

import timm
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO

# -------------------------------------------------------------------------
# Configuration and Model Setup
# -------------------------------------------------------------------------
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("../results/ocr.log"), logging.StreamHandler()],
)

text_det_model_path = "../runs/detect/train/weights/best.pt"
model_path = "../ocr_crnn_base_best_resnet152.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

chars = "0123456789abcdefghijklmnopqrstuvwxyz-"
vocab_size = len(chars)
char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

hidden_size = 256
n_layers = 3
dropout_prob = 0.2
unfreeze_layers = 3

# Load YOLO text detection model
yolo = YOLO(text_det_model_path)

# Define a single transform for inference
inference_transform = transforms.Compose(
    [
        transforms.Resize((100, 420)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


# -------------------------------------------------------------------------
# Model Definition: CRNN
# -------------------------------------------------------------------------
class CRNN(nn.Module):
    """
    A CRNN model that uses a ResNet152 backbone for feature extraction
    followed by a GRU layer for sequence modeling and a linear output layer
    for character prediction.
    """

    def __init__(
        self, vocab_size, hidden_size, n_layers, dropout=0.2, unfreeze_layers=3
    ):
        super(CRNN, self).__init__()

        # Backbone: ResNet152 in grayscale mode
        backbone = timm.create_model("resnet152", in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        # Unfreeze the last few layers of the backbone
        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        # Map feature dimension to a smaller one suitable for RNN
        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(dropout)
        )

        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Output layer
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        # Extract features
        x = self.backbone(x)
        # Permute to (B, W, C, H)
        x = x.permute(0, 3, 1, 2)
        # Flatten feature map
        x = x.view(x.size(0), x.size(1), -1)
        # Map sequence
        x = self.mapSeq(x)
        # GRU
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        # Predictions
        x = self.out(x)
        # Permute for CTC compatibility
        x = x.permute(1, 0, 2)
        return x


# Load CRNN model and weights
crnn_model = CRNN(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    n_layers=n_layers,
    dropout=dropout_prob,
    unfreeze_layers=unfreeze_layers,
).to(device)
crnn_model.load_state_dict(torch.load(model_path))


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def decode(encoded_sequences, idx_to_char, blank_char="-"):
    """
    Decode the predicted sequences (as indices) into characters.
    Skips the blank character and collapses repeated characters.
    """
    decoded_sequences = []
    for seq in encoded_sequences:
        decoded_label = []
        for token in seq:
            if token != 0:
                char = idx_to_char[token.item()]
                if char != blank_char:
                    decoded_label.append(char)
        decoded_sequences.append("".join(decoded_label))
    return decoded_sequences


def text_detection(img_path, det_model):
    """
    Perform text detection on the input image using the YOLO model.
    Returns bounding boxes, classes, class names, and confidences.
    """
    results = det_model(img_path, verbose=False)[0]
    bboxes = results.boxes.xyxy.tolist()
    classes = results.boxes.cls.tolist()
    confs = results.boxes.conf.tolist()
    names = results.names
    return bboxes, classes, names, confs


def text_recognition(img, transform, reg_model, idx_to_char, device):
    """
    Perform text recognition on the given cropped image using the CRNN model.
    """
    transformed_image = transform(img).unsqueeze(0).to(device)
    reg_model.eval()
    with torch.no_grad():
        logits = reg_model(transformed_image).cpu()
    # Decode predictions
    text = decode(logits.permute(1, 0, 2).argmax(2), idx_to_char)
    return text


def draw_predictions(image, predictions):
    """
    Draw bounding boxes and text predictions on the image.

    Args:
        image: PIL Image object
        predictions: List of (bbox, class_name, confidence, text) tuples

    Returns:
        PIL Image with drawings
    """
    # Create a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Colors for different classes (you can customize these)
    colors = {
        "text": (255, 0, 0),  # Red for text
        "number": (0, 255, 0),  # Green for numbers
        "default": (255, 165, 0),  # Orange for other classes
    }

    for bbox, class_name, confidence, text in predictions:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # Choose color based on class
        color = colors.get(class_name.lower(), colors["default"])

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Prepare label text
        label = f"{class_name} ({confidence:.2f}): {text}"

        # Calculate text position and background
        text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font)

    return draw_image


def predict(
    img_path, transform, det_model, reg_model, idx_to_char, device, output_dir=None
):
    """
    Perform full pipeline:
    1. Text detection with YOLO.
    2. Text recognition with CRNN.
    3. Save results with visualizations.
    """
    logging.info(f"Processing image: {img_path}")

    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory created/verified: {output_dir}")

    # Detection
    bboxes, classes, names, confs = text_detection(img_path, det_model)
    logging.info(f"Detected {len(bboxes)} text regions")

    # Load the original image
    img = Image.open(img_path)
    predictions = []

    # Process each detection
    for i, (bbox, cls_idx, conf) in enumerate(zip(bboxes, classes, confs)):
        x1, y1, x2, y2 = bbox
        name = names[int(cls_idx)]

        # Crop the detected region and recognize text
        cropped_image = img.crop((x1, y1, x2, y2))
        transcribed_text = text_recognition(
            cropped_image, transform, reg_model, idx_to_char, device
        )
        pred = (bbox, name, conf, transcribed_text[0])
        predictions.append(pred)

        logging.info(
            f"Region {i+1}: Class={name}, Confidence={conf:.2f}, Text={transcribed_text[0]}"
        )

    if output_dir:
        base_filename = os.path.splitext(os.path.basename(img_path))[0]

        # Save predictions to text file
        txt_output_path = os.path.join(output_dir, f"{base_filename}_predictions.txt")
        with open(txt_output_path, "w") as f:
            f.write(f"Predictions for {img_path}\n")
            f.write("-" * 50 + "\n")
            for bbox, name, conf, text in predictions:
                f.write(f"Class: {name}\n")
                f.write(f"Confidence: {conf:.2f}\n")
                f.write(f"Text: {text}\n")
                f.write(f"Bounding Box: {bbox}\n")
                f.write("-" * 50 + "\n")
        logging.info(f"Predictions saved to: {txt_output_path}")

        # Save visualization
        vis_output_path = os.path.join(output_dir, f"{base_filename}_visualization.png")
        visualization = draw_predictions(img, predictions)
        visualization.save(vis_output_path, "PNG")
        logging.info(f"Visualization saved to: {vis_output_path}")

    return predictions


# -------------------------------------------------------------------------
# Example Inference Run
# -------------------------------------------------------------------------
if __name__ == "__main__":
    img_dir = "../datasets/SceneTrialTrain/lfsosa_12.08.2002"
    output_dir = "../runs/ocr"

    logging.info("Starting OCR processing")
    logging.info(f"Input directory: {img_dir}")
    logging.info(f"Output directory: {output_dir}")

    # Use the defined inference transform
    for idx, img_filename in enumerate(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_filename)
        logging.info(f"\nProcessing image {idx+1}: {img_filename}")

        try:
            preds = predict(
                img_path,
                transform=inference_transform,
                det_model=yolo,
                reg_model=crnn_model,
                idx_to_char=idx_to_char,
                device=device,
                output_dir=output_dir,
            )
            logging.info(f"Successfully processed {img_filename}")

        except Exception as e:
            logging.error(f"Error processing {img_filename}: {str(e)}")
            continue

        # For demonstration, break after processing a few images
        if idx == 10:
            break

    logging.info("OCR processing completed")
