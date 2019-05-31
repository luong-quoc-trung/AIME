from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *

img_path = Path('data/all_images/image_moderation_images')
gan_path = Path('data/gan/')
train_df_path = Path('data/imgs_train.csv')
test_df_path = Path('data/test_set.csv')
saved_models_folder = Path('saved_models/')

arch = models.resnet34
src = ImageImageList.from_folder(gan_path/'text_added').split_by_rand_pct(0.1, seed=123)
databunch = (src.label_from_func(lambda x: gan_path/'clean'/x.name)
                .transform(get_transforms(do_flip = False, max_rotate=0, max_warp=0),
                            tfm_y=True,size=(298,298))
                .databunch(bs=32).normalize(imagenet_stats,do_y=True))

generator = unet_learner(databunch, arch, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=(-3.,3.), loss_func=MSELossFlat())
critic    = basic_critic(in_size=299, n_channels=3, n_extra_layers=1)
# generator.fit_one_cycle(2, pct_start=0.8)

learn = GANLearner.wgan(databunch, generator.model, critic, switch_eval=False,
                       opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.)
learn.fit(3,2e-4) 

