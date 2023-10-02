import torch
import torchvision.transforms as tfs
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Pretrainedmodel import uresnet
from skimage import io
from os import listdir
import SimpleITK as sitk

#%% 16bits to 8bit image normalization
# im = io.imread('C:\\Users\\kunning\\Desktop\\CTImages_moomRock\\CTImages\\3Draw_16bit.tif')

# im[im<=16500]=16500  # Lower boundary for 16bits image
# im[im>=44000]=44000 # Higher boundary for 16bits image
# min_16bit2 = 16500
# max_16bit2 = 44000
# image_8bits =(255* ((im - min_16bit2) / (max_16bit2 - min_16bit2)))
# output = np.uint8(image_8bits)

# label_s = sitk.GetImageFromArray(output)
# sitk.WriteImage(label_s,'C:\\Users\\kunning\\Desktop\\CTImages_moomRock\\CTImages\\3Draw_8bit.nii.gz')  

#%% segmentation testing
net = uresnet()
checkpoint = torch.load('C:\\Users\\kunning\\Desktop\\CTImages_moomRock\\X036\\3Draw_8bit.nii\\ck\\test1.pt')
net.load_state_dict(checkpoint['net_state_dict'])

size = 640;size1 = 640
def crop(data,height=size1, width=size):
    st_x = 0
    st_y = 0
    box = (st_x, st_y, st_x+width, st_y+height)
    data = data.crop(box)
    return data


net.eval()

dir = listdir('C:\\Users\\kunning\\Desktop\\CTImages_moomRock\\X038\\slice_8bits')
im_tfs = tfs.Compose([
      tfs.ToTensor(),
      # tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
      ])
# for i in range(len(img1)):
i= 10
# # img = img1[i,:,:]
# # for i in range(1450,1470):
#     # img2 = Image.open('D:\\GBG\\slice\\'+dir[i])
img = io.imread('C:\\Users\\kunning\\Desktop\\CTImages_moomRock\\X038\\slice_8bits\\'+dir[i])  

# im[im<=16500]=16500  # Lower boundary for 16bits image
# im[im>=44000]=44000 # Higher boundary for 16bits image
# min_16bit2 = 16500
# max_16bit2 = 44000
# image_8bits =(255* ((im - min_16bit2) / (max_16bit2 - min_16bit2)))
# img = np.uint8(image_8bits)

img2 = Image.fromarray(img)
cut_image = crop(img2).convert('RGB')
cut_image1 = im_tfs(cut_image)
test_image1 = cut_image1.unsqueeze(0).float()
out = net(test_image1)

pred = out.max(1)[1].squeeze().data.cpu().numpy()
pred = np.uint8(pred)
pred = Image.fromarray(pred)

plt.subplot(121)
plt.imshow(cut_image)
plt.subplot(122)
plt.imshow(pred)


# pred.save('C:\\Users\\kunning\\Desktop\\CTImages_moomRock\\CTImages\\3Draw_8bit.nii\\result\\seg\\' + '%d' % i + '.png')
# cut_image.save('C:\\Users\\kunning\\Desktop\\CTImages_moomRock\\CTImages\\3Draw_8bit.nii\\result\\raw\\' + '%d' % i + '.png')

