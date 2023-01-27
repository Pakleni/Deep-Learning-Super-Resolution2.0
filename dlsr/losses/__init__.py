from .srgan import get_srgan_loss
from .vgg import vgg_style_loss, vgg_content_loss
from .basic import ssim_loss, psnr_loss, psnr_abs_loss


def get_custom_objects(srgan=None):
    custom_objects = {
        "ssim_loss": ssim_loss,
        "vgg_style_loss": vgg_style_loss,
        "vgg_content_loss": vgg_content_loss,
        "psnr_loss": psnr_loss,
        "psnr_abs_loss": psnr_abs_loss,
    }
    if srgan:
        return {**custom_objects, **get_srgan_loss(srgan)}
    else:
        return custom_objects
