from fastai import *
from fastai.vision import *
import os
import cv2
from skimage import io

from torch.utils.data import Dataset, DataLoader
import torchvision
from skimage import io

img_path = Path('./Data/all_images/image_moderation_images')
gan_path = Path('./Data/GAN/')
train_df_path = Path('./Data/imgs_train.csv')
saved_models_folder = Path('./Saved_models/')

class ImageDataSet(Dataset):
    def __init__(self, dataframe, img_folder_path):
        super().__init__()
        self.dataframe = dataframe
        self.path = img_folder_path
        self.transform = torchvision.transforms.ToTensor()
        
    def __getitem__(self,i):
        img_path = Path(self.path) / self.dataframe.images_id.iloc[i]
        label = self.dataframe.labels.iloc[i]
        img = io.imread(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return (img, label)       
    
class visualize():
    def __init__(self, dataset, model, pool_layer,fc_weight,name):
        self.model = model
        self.data = dataset
        self.pool_layer = pool_layer
        self.fc_weight = fc_weight
        self.name = name
        self.model.eval()
        self.pool_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input[0][0]
        
    def viz(self,index,cls, mask_threshold,show=True):
        #N,C,H,W
        img, label = self.data[index]
        output = self.model(img[None].cuda())[0]
        if output[1] >0.5:
            pred = self.name
        else:
            pred = 'Not {}'.format(self.name)
        #C,H,W
        
        img = img.cpu().detach().numpy().transpose(1,2,0)
        last_conv_act = self.input.squeeze(0).cpu().detach().numpy()
        weight = self.fc_weight[cls,:].view(-1,1,1).cpu().detach().numpy()
        
        heat_map = (last_conv_act * weight).sum(0)
        heat_map = (heat_map- np.min(heat_map))/ np.max(heat_map)
        heat_map = np.uint8(255*heat_map)
        heat_map= cv2.resize(heat_map, (299,299))

        masked = np.expand_dims(heat_map<mask_threshold,2) * img
        heat_map = cv2.applyColorMap(heat_map,cv2.COLORMAP_JET)[:,:,[2,1,0]]
        mixed = heat_map * 0.5*(1./255) + img * 0.5
        
        if show:
            toPlot = [mixed,img,heat_map,masked]
            for i,ax in enumerate(plt.subplots(2,2,figsize=(10,10))[1].flatten()):
                ax.imshow(toPlot[i])
                ax.set_title(label='Actual:{} | Pred: {}'.format(label,pred))
                ax.axis('off')
            plt.show()
        return img,masked
    
    def build_mask_dataset(self):
#         shutil.rmtree('.masked_images/')
        os.makedirs( gan_path/'masked_images/')
        for i in range(len(self.data.dataframe)):
            if self.name in self.data.dataframe.labels[i]:
                if i %100 == 0:
                    print(i,end=',')
                _, masked_ = self.viz(i,1,90,show = False)
                masked_.imsave(gan_path/('{}_masked.jpg'.format(self.data.dataframe.images_id[i][:-4])))
        print("done")
    
def get_model():
        base = models.resnet50(pretrained=True)
        for parms in base.parameters():
            parms.requires_grad = False
        modified_model = nn.Sequential(*[m for m in base.children()][:-2], 
                             nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                           Flatten(),
                                          nn.Linear(2048,2)))
        return modified_model


if __name__ == '__main__':
    ds =ImageDataSet(pd.read_csv(train_df_path),
                      img_path)

    md = get_model().cuda()
    md.load_state_dict(torch.load(saved_models_folder/'text.pth',map_location='cpu'))
    pool_layer = md[-1][0]
    fc_weight = md[-1][-1].weight

    visual.viz(1231,1,90)
