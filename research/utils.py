import matplotlib.pyplot as plt
import numpy as np

def show_images(low_res, super_res, high_res):
    images = [low_res, super_res, high_res]
    titles = ["Low Resolution", "Super Resolution", "High Resolution"]
    
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, 3, i + 1)
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        plt.title(titles[i])
        plt.axis("off")
    plt.show()
