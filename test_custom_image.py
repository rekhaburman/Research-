import torch
import torchvision.transforms as transforms
from PIL import Image
from research.model import LightweightGenerator

# Load the trained model
model = LightweightGenerator()
model.load_state_dict(torch.load("research/super_res_model.pth", weights_only=True))
model.eval()  # Set the model to evaluation mode

# Preprocessing function
def preprocess_image(image_path, target_size=(32, 32)):
    """
    Preprocess the image: Resize, Normalize, and Convert to Tensor.
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize to low-res dimensions
        transforms.ToTensor(),          # Convert to tensor and scale [0, 255] to [0, 1]
    ])
    image = Image.open(image_path).convert("RGB")  # Open image and ensure 3 channels
    return transform(image).unsqueeze(0)          # Add batch dimension

# Postprocessing function
def postprocess_image(tensor, save_path="output_images/super_res_output.jpg"):
    """
    Convert the output tensor back to an image and save it.
    """
    transform = transforms.ToPILImage()
    image = transform(tensor.squeeze(0).clamp(0, 1))  # Clamp values to [0, 1]
    image.save(save_path)
    print(f"Super-resolved image saved at: {save_path}")

# Path to the custom image
custom_image_path = "input_images/bird.jpg"  # Replace with your image file path

# Preprocess the input image
low_res_image = preprocess_image(custom_image_path)

# Forward pass through the model
with torch.no_grad():  # Disable gradient computation for inference
    super_res_image = model(low_res_image)

# Postprocess and save the output image
postprocess_image(super_res_image, save_path="output_images/super_res_output.jpg")
