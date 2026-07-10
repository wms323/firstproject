from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer=SummaryWriter('logs')
image_path='Data/train/ants/162603798_40b51f1654.jpg'
image_PIL = Image.open(image_path)
img_array = np.array(image_PIL)
print(type(image_PIL))
print(type(img_array))
print(img_array.shape)

writer.add_image('ants',img_array,2,dataformats='HWC')

for i in range(100):
    writer.add_scalar('y=3x',3*i,i)

writer.close()
