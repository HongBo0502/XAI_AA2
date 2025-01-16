from flask import Flask, render_template, request, redirect, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import os
from utils import GradCAM , generate_image_tensor , visualize_gradcam , lrp_image, generate_single_perturbed_image
app = Flask(__name__)

# Load the DenseNet model and its weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model architecture and weights
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 1)
model.load_state_dict(torch.load("densenet121_model_weights (1).pth", map_location=device))
model.eval()  # Set the model to evaluation mode
# Initialize Grad-CAM with the loaded model
target_layer_name = "features.denseblock4.denselayer16.conv2"
gradcam = GradCAM(model, target_layer_name)
def predict(image):
    # Ensure the image is in RGB format
    image = image.convert("RGB")
    
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        outputs = model(image_tensor)  # Raw logits
        probabilities = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
        predicted_class = (probabilities >= 0.7).int().item()  # Threshold at 0.5 for binary classification

    return predicted_class, probabilities.item()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/concept", methods=["GET", "POST"])
def concept():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            # Save the image temporarily
            img_path = os.path.join("static", file.filename)
            file.save(img_path)
            
            #image to tensor
            image_tensor = generate_image_tensor(img_path)
            # Generate Grad-CAM heatmap
            heatmap = gradcam.generate_heatmap(image_tensor, target_class=1)
            
            # Visualize the heatmap
            gradcam_image_path = os.path.join("static", "Grad_cam.png")

            
            visualize_gradcam(image_tensor, heatmap, save_path=gradcam_image_path)

            # Perform LRP
            lrp_image_path = os.path.join("static", "LRP.png")
            lrp_image(img_path,image_tensor,model,save_path=lrp_image_path)


            # Open and process the image
            image = Image.open(img_path)
            predicted_class, probability = predict(image)
            
            return render_template(
                "concept.html", 
                image_path=img_path, 
                gradcam_image_path=gradcam_image_path,
                lrp_image_path=lrp_image_path,
                predicted_class=predicted_class,
                probability=probability
            )
    return render_template("concept.html")

@app.route("/counterfactual", methods=["GET", "POST"])
def counterfactual():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            # Save and load the image
            img_path = os.path.join("static", file.filename)
            file.save(img_path)
            image = Image.open(img_path).convert("RGB")
            #image to tensor
            # Generate a single perturbed image
            perturbed_image = generate_single_perturbed_image(image,patch_ratio=0.4,replacement_value=0,offset_x=-200,offset_y=-60)
            perturbed_image_path = os.path.join("static", "perturbed.png")
            perturbed_image.save(perturbed_image_path)

            # Predict for the perturbed image and the original image with the labels
            predicted_class_original, probability_original = predict(image)
            predicted_class_perturbed, probability_perturbed = predict(perturbed_image)

            return render_template(
                "counterfactual.html",
                image_path=img_path,
                perturbed_image_path=perturbed_image_path,
                predicted_class_original=predicted_class_original,
                predicted_class_perturbed=predicted_class_perturbed,
                probability_original=probability_original,
                probability_perturbed=probability_perturbed,
            )
    return render_template("counterfactual.html")


if __name__ == "__main__":
    app.run(debug=True)
