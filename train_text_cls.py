import copy
import cv2
from fastai import *
from fastai.vision import *
import os
from pathlib import Path
from skimage import io
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset, DataLoader
import torchvision

img_path = Path('data/all_images/image_moderation_images')
gan_path = Path('data/gan/')
train_df_path = Path('data/imgs_train.csv')
test_df_path = Path('data/test_set.csv')
saved_models_folder = Path('saved_models/')

def create_databunch(label,bs=32):
    train_df = pd.read_csv(train_df_path).sample(frac=1.).reset_index(drop=True)
    test_df = pd.read_csv(test_df_path)
    
    train_df.labels = train_df.labels.apply(lambda x: 1 if label in x else 0)
    src = (ImageList.from_df(df = train_df, path = img_path)
           .split_by_rand_pct(0.2)
           .label_from_df())

    Dataset2 = (src.transform(get_transforms(do_flip = False, max_rotate=0, max_warp=0))
            .databunch(bs=bs))
    Dataset2.add_test(ImageList.from_df(df = test_df, path = img_path))
    return Dataset2

def get_model():
        base = models.resnet50(pretrained=True)
        for parms in base.parameters():
            parms.requires_grad = True
        modified_model = nn.Sequential(*[m for m in base.children()][:-2], 
                             nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                           Flatten(),
                                          nn.Linear(2048,2)))
        return modified_model

def create_learner(data_bunch):
    learn = Learner(data_bunch,get_model(), model_dir=saved_models_folder,
                        metrics=[accuracy])
    return learn

def plot_precision_recall_curve(y_scores, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores[:,1],pos_label=1)
    thresholds= thresholds.tolist() + [1]
    plt.plot(thresholds,precision,'-r', label="precision")
    plt.plot(thresholds,recall,'-b', label="recall")
    plt.legend(loc='lower right')

def eval_score_with_threshold(y_score, y_true, thresh=0.3):
    preds_w_thres = [1 if score>=thresh else 0  for score in y_score[:,1]]
    print(np.mean(preds_w_thres))
    print("Mean F1",f1_score(np.array(y_true), np.array(preds_w_thres),pos_label=1))
    print("ACc F1",accuracy_score(np.array(y_true), np.array(preds_w_thres)))
    print(classification_report(np.array(y_true), np.array(preds_w_thres)))

if __name__ == '__main__':
    
    ds =ImageDataSet(pd.read_csv(train_df_path),img_path)
    data_bunch = create_databunch('text')
    learn = create_learner(data_bunch)

    learn.lr_find()
    learn.recorder.plot()

    learn.fit_one_cycle(8,2e-4)

    plot_precision_recall_curve(*learn.get_preds())
    eval_score_with_threshold(*learn.get_preds())

    learn.export(saved_models_folder/'learner_text.p')