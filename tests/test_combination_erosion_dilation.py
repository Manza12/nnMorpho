import torch
import time
from nnMorpho.operations import erosion, dilation

size_image = (1000, 1000)
size_strel = (100, 100)

image = torch.rand(size_image, device='cuda:0')
strel = torch.rand(size_strel, device='cuda:0')

if __name__ == '__main__':
    sta = time.time()
    eroded_image = erosion(image, strel)
    end = time.time()
    print("Time elapsed for erosion:", round(1e3 * (end - sta)), "ms")

    sta = time.time()
    dilated_image = dilation(image, strel)
    end = time.time()
    print("Time elapsed for dilation:", round(1e3 * (end - sta)), "ms")

    sta = time.time()
    eroded_image_mistake = -dilation(-image, strel)
    end = time.time()
    print("Time elapsed for mistaken erosion:", round(1e3 * (end - sta)), "ms")

    sta = time.time()
    eroded_image_adapted = -dilation(-image, strel.flip((0, 1)))
    end = time.time()
    print("Time elapsed for adapted erosion:", round(1e3 * (end - sta)), "ms")

    print("Error between mistaken erosion and actual erosion:",
          torch.norm(eroded_image_mistake - eroded_image, 1).item())
    print("Error between adapted erosion and actual erosion:",
          torch.norm(eroded_image_adapted - eroded_image, 1).item())
