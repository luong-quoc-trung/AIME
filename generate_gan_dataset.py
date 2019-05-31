from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import itertools
import random
import os
import pandas as pd

img_path = Path('./Data/all_images/image_moderation_images')
gan_path = Path('./Data/GAN/')
train_df_path = Path('./Data/imgs_train.csv')

font_text = ['Arial']
font_size = [18,20,22,24]
font_type = ['Bold','Italic']
texts = ['9998','12312']
fills = [(255,69,0)]

def create_random_font():
    fonts = list(itertools.product(font_text,font_type,font_size))
    random_font = random.choice(fonts)
    return ImageFont.load_default()

def add_text_to_img(img):
    img = img.copy()
    draw =  ImageDraw.Draw(img)
    h,w = img.size
    draw.text(xy=(random.randrange(0,h),random.randrange(0,w)),
              text= random.choice(texts),
              fill= random.choice(fills),
              font= create_random_font())
    return img

def generate_data(img_path = img_path,
                  gan_path = gan_path,
                  data = pd.read_csv(train_df_path)):
    os.makedirs(gan_path/'clean')
    os.makedirs(gan_path/'text_added')
    
    fnames = data.images_id[~data.labels.str.contains('text')]
    for fname in fnames:
        img = Image.open(img_path/fname).convert('RGB')
        added_text = add_text_to_img(img)

        img.save(gan_path/'clean'/fname)
        added_text.save(gan_path/'text_added'/fname)

if __name__ == '__main__':
    generate_data()