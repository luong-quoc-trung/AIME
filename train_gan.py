from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *

path = Path('./Data/GAN')

arch = models.resnet34
src = ImageImageList.from_folder(path/'clean').split_by_rand_pct(0.1, seed=123)
databunch = (src.label_from_func(lambda x: path/'text_added'/x.name)
                .transform(get_transforms(do_flip = False, max_rotate=0, max_warp=0),
                            tfm_y=True,size=(298,298))
                .databunch(bs=32).normalize(imagenet_stats,do_y=True))

generator = unet_learner(databunch, arch, blur=True, norm_type=NormType.Weight,
                         self_attention=True, y_range=(-3.,3.), loss_func=MSELossFlat())
critic    = basic_critic   (in_size=299, n_channels=3, n_extra_layers=1)
generator.fit_one_cycle(2, pct_start=0.8)

# learn = GANLearner.wgan(databunch, generator.model, critic, switch_eval=False,
                        opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.)
# learn.fit(3,2e-4) 

