import cv2
import numpy as np
import matplotlib.pyplot as plt
img_1 =  cv2.imread('examples/1864109.jpg')

mask = np.zeros(img_1.shape)
# mask = img_1.copy()
mask[206:234,14:288,:] = 255
cv2.imwrite('examples/1864109_masked.jpg',mask)

# plt.subplot(1,2,1)
# plt.imshow(mask)
# plt.subplot(1,2,2)
# plt.imshow(img_1)
# plt.show()