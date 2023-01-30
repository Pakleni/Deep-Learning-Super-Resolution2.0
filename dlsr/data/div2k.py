import os
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE


def download_archive(file, target_dir, extract=True):
    source_url = f"http://data.vision.ee.ethz.ch/cvl/DIV2K/{file}"
    target_dir = os.path.abspath(target_dir)
    tf.keras.utils.get_file(file, source_url, cache_subdir=target_dir, extract=extract)
    os.remove(os.path.join(target_dir, file))


def crop(lr_img, hr_img, hr_crop_size, scale):

    lr_crop_size = hr_crop_size // scale

    lr_w = 0
    lr_h = 0

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h : lr_h + lr_crop_size, lr_w : lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h : hr_h + hr_crop_size, hr_w : hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_crop(lr_img, hr_img, hr_crop_size, scale):

    lr_crop_size = hr_crop_size // scale
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(
        shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32
    )
    lr_h = tf.random.uniform(
        shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32
    )

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h : lr_h + lr_crop_size, lr_w : lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h : hr_h + hr_crop_size, hr_w : hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(
        rn < 0.5,
        lambda: (lr_img, hr_img),
        lambda: (tf.image.flip_left_right(lr_img), tf.image.flip_left_right(hr_img)),
    )


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


class DIV2K:
    def __init__(self, type="train", scale=2):
        self.dir = ""
        self.type = type
        if scale not in [2, 3, 4]:
            raise ValueError("Scale should be 2, 3 or 4")
        self.scale = scale

    def dataset(
        self,
        batch_size,
        image_size,
        repeat_count=None,
        random_transform=True,
        crop_images=False,
        modifier=None,
    ) -> tf.data.Dataset:
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))

        ds = ds.repeat(repeat_count)

        if random_transform:
            ds = ds.map(
                lambda lr, hr: random_crop(
                    lr, hr, scale=self.scale, hr_crop_size=image_size
                ),
                num_parallel_calls=AUTOTUNE,
            )
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        elif crop_images:
            ds = ds.map(
                lambda lr, hr: crop(lr, hr, scale=self.scale, hr_crop_size=image_size),
                num_parallel_calls=AUTOTUNE,
            )
        if modifier:
            ds = ds.map(modifier, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(batch_size, num_parallel_calls=AUTOTUNE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        if not os.path.exists(self._hr_images_dir()):
            download_archive(self._hr_images_archive(), self.dir, extract=True)

        ds = self._images_dataset(self._hr_image_files())

        ds = ds.cache()

        return ds

    def lr_dataset(self):
        if not os.path.exists(self._lr_images_dir()):
            download_archive(self._lr_images_archive(), self.dir, extract=True)

        ds = self._images_dataset(self._lr_image_files())

        ds = ds.cache()

        return ds

    def _hr_image_files(self):
        images_dir = self._hr_images_dir()
        return [
            os.path.join(images_dir, filename)
            for filename in sorted(os.listdir(images_dir))
        ]

    def _lr_image_files(self):
        images_dir = self._lr_images_dir()
        return [
            os.path.join(images_dir, filename)
            for filename in sorted(os.listdir(images_dir))
        ]

    def _hr_images_dir(self):
        return os.path.join(self.dir, f"DIV2K_{self.type}_HR")

    def _lr_images_dir(self):
        return os.path.join(self.dir, f"DIV2K_{self.type}_LR_bicubic", f"X{self.scale}")

    def _hr_images_archive(self):
        return f"DIV2K_{self.type}_HR.zip"

    def _lr_images_archive(self):
        return f"DIV2K_{self.type}_LR_bicubic_X{self.scale}.zip"

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(
            lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE
        )
        return ds
