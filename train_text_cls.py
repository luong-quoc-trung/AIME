from fastai import *
from fastai.vision import *
import os
from pathlib import Path
import cv2

from torch.utils.data import Dataset, DataLoader
import torchvision
from skimage import io

img_path = Path('./Data/all_images/image_moderation_images')
gan_path = Path('./Data/GAN/')
train_df_path = Path('./Data/imgs_train.csv')
test_df_path = Path('./Data/test_set.csv')
saved_models_folder = Path('./Saved_models/')

def create_databunch(label):
    train_df = pd.read_csv(train_df_path).sample(frac=1.).reset_index(drop=True)
    test_df = pd.read_csv(test_df_path)
    
    train_df.labels = train_df.labels.apply(lambda x: 1 if label in x else 0)
    src = (ImageList.from_df(df = train_df, path = img_path)
           .split_by_rand_pct(0.2)
           .label_from_df())
    bs= 32
    
    Dataset2 = (src.transform(get_transforms(do_flip = False, max_rotate=0, max_warp=0))
            .databunch(bs=bs))
    Dataset2.add_test(ImageList.from_df(df = test_df, path = img_path))
    return Dataset2

def create_learner(data_bunch):
    def get_model():
        base = models.resnet50(pretrained=True)
        for parms in base.parameters():
            parms.requires_grad = True
        modified_model = nn.Sequential(*[m for m in base.children()][:-2], 
                             nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                           Flatten(),
                                          nn.Linear(2048,2)))
        return modified_model
    
    learn = Learner(data_bunch,get_model(), model_dir=saved_models_folder,
                        metrics=[accuracy])
    return learn
   
class ImageDataSet(Dataset):
    def __init__(self, dataframe, img_folder_path):
        super().__init__()
        self.dataframe = dataframe
        self.path = img_folder_path
        self.transform = torchvision.transforms.ToTensor()
        
    def __getitem__(self,i):
        img_path = Path(self.path) / self.dataframe.images_id[i]
        label = self.dataframe.labels[i]
        img = io.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return (img, label)       
    
ds =ImageDataSet(pd.read_csv(train_df_path),img_path)

data_bunch = create_databunch('text')
learn = create_learner(data_bunch)

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(8,2e-4)

plot_precision_recall_curve(*learn.get_preds())
eval_score_with_threshold(*learn.get_preds())

md = learn.model
pool_layer = md[-1][0]
fc_weight = md[-1][-1].weight

visual = visualize(ds, md,pool_layer,fc_weight,'text')
visual.viz(1231,1)