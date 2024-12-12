import torch
from model import LightweightGenerator
from train import model  # Assuming you trained the model here
from utils import show_images

# Simulate test data
low_res_test = torch.randn(1, 3, 32, 32)  # Low-resolution test image
high_res_test = torch.randn(1, 3, 64, 64)  # High-resolution ground truth

# Generate super-resolution image
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    super_res_test = model(low_res_test)

# Visualize results
show_images(low_res_test.squeeze(), super_res_test.squeeze(), high_res_test.squeeze())
