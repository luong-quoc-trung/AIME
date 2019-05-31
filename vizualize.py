from fastai import *
from fastai.vision import *

data = ImageDataBunch.from_df(df = pd.read_csv('Data/imgs_train.csv'),
                              path = '.',
                              folder='Data/all_images/image_moderation_images/',
                              label_delim = ' ')

data.show_batch()