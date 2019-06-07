from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import itertools
import random
import os
import pandas as pd
import shutil
from random_word import RandomWords
r = RandomWords()

img_path = Path('data/all_images/image_moderation_images')
gan_path = Path('data/gan/')
train_df_path = Path('./data/imgs_train.csv')
test_df_path = Path('./data/test_set.csv')
saved_models_folder = Path('./saved_models/')


font_size = [15,18,21,24,27,30,33]
font_text = ['arial']
# font_type = ['Bold','Italic']
fills = [(255,69,0)]
# texts = ['9998','12312']

# img = Image.open('data/all_images/image_moderation_images/661724.jpg')
# draw = ImageDraw.Draw(img)
# font = ImageFont.truetype("sans-serif.ttf", 15)
# draw.text((40,40),'Test',(255,255,255),font=font)
# img.show()

def add_text_to_img(img):
    img = img.copy()
    draw =  ImageDraw.Draw(img)

    fonts = list(itertools.product(font_text,font_size))
    random_font = random.choice(fonts)
    ftext, fsize = random_font
    font = ImageFont.truetype(font = '{}.ttf'.format(ftext), size = fsize)

    h,w = img.size
    x,y = random.randrange(0,h-fsize),random.randrange(0,w-fsize)
    draw.text(xy= (x,y),
              text= r.get_random_word(),
              fill= random.choice(fills),
              font= font)
    return img

def generate_data(img_path = img_path,
                  gan_path = gan_path,
                  data = pd.read_csv(train_df_path)):
    shutil.rmtree(gan_path/'clean')
    shutil.rmtree(gan_path/'text_added')

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