import tensorflow as tf
import matplotlib.pyplot as plt
from .data import DIV2K
import os
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

# normalize low resolution images to [0,1]
def normalize_lr(x):
    return x / 255


# normalize high resolution images to [-1,1]
def normalize_hr(x):
    val = 255 / 2
    return (tf.cast(x, tf.float32) - val) / val


# denormalize high resolution images to [0,255]
def denormalize_hr(x):
    val = 255 / 2
    adjusted = x * val + val
    clipped = tf.clip_by_value(adjusted, 0, 255)
    return tf.cast(tf.math.round(clipped), tf.uint8)


class History:
    def __init__(self, params: list = ["val_loss", "loss"]):
        self.params = params
        self.history = {param: [] for param in params}

    def extend(self, x: tf.keras.callbacks.History):
        for param in self.params:
            self.history[param].extend(x.history[param])

    def plot(
        self,
        map: dict = {
            "val_loss": "Validation Loss",
            "loss": "Loss",
        },
    ):
        for param in map:
            plt.plot(self.history[param], label=map[param])

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="lower right")
        plt.show()


def load_dataset(
    type: str,
    image_size,
    batch_size: int = 1,
    random_transform: bool = True,
    repeat_count=None,
    scale=2,
):
    loader = DIV2K(type=type, scale=scale)
    return loader.dataset(
        batch_size=batch_size,
        random_transform=random_transform,
        image_size=image_size,
        modifier=lambda lr, hr: (normalize_lr(lr), normalize_hr(hr)),
        repeat_count=repeat_count,
    )


def get_training_data(
    image_size,
    random_transform: bool = True,
    batch_size: int = 1,
    repeat_count=None,
    scale=2,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    valiation_dataset = load_dataset(
        type="valid",
        random_transform=random_transform,
        repeat_count=1,
        image_size=image_size,
        batch_size=batch_size,
        scale=scale,
    )
    training_dataset = load_dataset(
        type="train",
        random_transform=random_transform,
        repeat_count=repeat_count,
        image_size=image_size,
        batch_size=batch_size,
        scale=scale,
    )

    return training_dataset, valiation_dataset


def dataset_from_folder(input: str):
    images_dir = os.path.join(input)
    image_files = [
        os.path.join(images_dir, filename)
        for filename in sorted(os.listdir(images_dir))
    ]

    ds = tf.data.Dataset.from_tensor_slices(image_files)
    ds = ds.map(tf.io.read_file)
    ds = ds.map(
        lambda x: tf.io.decode_image(x, channels=3), num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(1)

    return ds


def config(use_gpu: bool, eager: bool = False):
    if eager:
        tf.config.run_functions_eagerly(True)
    if use_gpu:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        tf.config.set_visible_devices([], "GPU")
