import pandas as pd
from text_localization import EAST_detect_text
from pathlib import Path 
import cv2
import subprocess

img_folder = Path('data/all_images/image_moderation_images')
east_folder = Path('data/east')
train_df_path = Path('data/imgs_train.csv')

subprocess.call('rm -r data/east/clean/*')
subprocess.call('rm -r data/east/mask/*')
subprocess.call('rm -r data/east/impainted/*')

df = pd.read_csv(train_df_path)
df = df[df.labels.str.contains('text')]
fnames = df.images_id.tolist()

net = cv2.dnn.readNet('text_localization/frozen_east_text_detection.pb')

for name in fnames[:20]:
    img_path = str(img_folder/name)
    print(img_path)
    image = cv2.imread(img_path)
    mask = EAST_detect_text.generate_mask(image,net, min_conf = 0.3, show=False)
    cv2.imwrite(str(east_folder/'mask'/name), mask)
    cv2.imwrite(str(east_folder/'clean'/name), image)