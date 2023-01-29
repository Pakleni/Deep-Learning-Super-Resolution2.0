from .srgan import get_srgan_loss
from .vgg import vgg_style, vgg_content
from .image import ssim, psnr_mse, psnr_mae


def get_custom_objects(discriminator=None):
    custom_objects = {
        "ssim": ssim,
        "vgg_style": vgg_style,
        "vgg_content": vgg_content,
        "psnr_mse": psnr_mse,
        "psnr_mae": psnr_mae,
    }
    if discriminator:
        return {**custom_objects, **get_srgan_loss(discriminator)}
    else:
        return custom_objects
