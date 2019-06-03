import os
from generative_inpainting.test import impaint

model_place ='release_place_2_256'
model_image_net = 'release_imagenet_256'

for fname in os.listdir('data/east/clean'):
    img_clean = 'data/east/clean/{}'.format(fname)
    img_mask = 'data/east/mask/{}'.format(fname)
    check_point_dir = 'generative_inpainting/model_logs/{}'.format(model_place)
    output_dir = 'data/east/impainted/{}'.format(fname)
    
    impaint(img_clean,img_mask,output_dir,check_point_dir)