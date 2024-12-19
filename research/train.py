import torch
import torch.optim as optim
from model import LightweightGenerator
from loss import HybridLoss

model = LightweightGenerator()
loss_fn = HybridLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


low_res_data = torch.randn(10, 3, 32, 32) 
high_res_data = torch.randn(10, 3, 64, 64)


num_epochs = 5
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for lr_img, hr_img in zip(low_res_data, high_res_data):
        lr_img = lr_img.unsqueeze(0) 
        hr_img = hr_img.unsqueeze(0)
        
       
        sr_img = model(lr_img)
        loss = loss_fn(sr_img, hr_img)
        
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


torch.save(model.state_dict(), "super_res_model.pth")
print("Model saved as 'super_res_model.pth'")