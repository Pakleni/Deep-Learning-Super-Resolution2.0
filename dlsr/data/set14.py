import os
import tensorflow as tf

from tensorflow.python.data.experimental import AUTOTUNE


def download_archive(target_dir, extract=True):
    file = "Set14_SR.zip"
    source_url = (
        f"https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip"
    )
    target_dir = os.path.abspath(target_dir)
    tf.keras.utils.get_file(file, source_url, cache_subdir=target_dir, extract=extract)
    os.remove(os.path.join(target_dir, file))


class Set14:
    valid_types = [
        "LR",
        "HR",
        "bicubic",
        "glasner",
        "Kim",
        "nearest",
        "ScSR",
        "SelfExSR",
        "SRCNN",
    ]

    def __init__(self, scale=2):
        self.dir = ""
        self.scale = scale

        if self.scale not in [2, 3, 4]:
            raise ValueError("Scale should be 2, 3 or 4")

    def dataset(self, type="LR"):
        if type not in self.valid_types:
            raise ValueError(f"Type should be one of {self.valid_types}")

        if not os.path.exists(self._images_dir()):
            download_archive(self.dir, extract=True)

        ds = self._images_dataset(self._image_files(type))
        return ds

    def _image_files(self, type):
        images_dir = self._images_dir()
        return [
            os.path.join(images_dir, filename)
            for filename in sorted(os.listdir(images_dir))
            if filename.endswith(f"_{type}.png")
        ]

    def _images_dir(self):
        return os.path.join(self.dir, "Set14", f"image_SRF_{self.scale}")

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(
            lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE
        )
        return ds
