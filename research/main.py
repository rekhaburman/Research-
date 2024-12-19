import torch
from model import LightweightGenerator
from train import model 
from utils import show_images


low_res_test = torch.randn(1, 3, 32, 32) 
high_res_test = torch.randn(1, 3, 64, 64)  

model.eval()  
with torch.no_grad():
    super_res_test = model(low_res_test)

show_images(low_res_test.squeeze(), super_res_test.squeeze(), high_res_test.squeeze())
