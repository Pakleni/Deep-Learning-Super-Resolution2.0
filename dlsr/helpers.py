def norm(x):
    return x / 255


def denorm(x):
    return (x * 255).astype("int32")


def plot_history(history):
    import matplotlib.pyplot as plt

    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="lower right")
    plt.show()


def load_dataset(
    num: int, type: str, random_transform: bool = True, batch_size: int = 1
):
    import numpy as np
    from .data import DIV2K

    loader = DIV2K(type=type)
    ds = loader.dataset(batch_size=batch_size, random_transform=random_transform)

    lr = []
    hr = []

    for x in ds.take(num):
        lr.extend([norm(x[0][0])])
        hr.extend([norm(x[1][0])])

    lr = np.array(lr)
    hr = np.array(hr)

    return lr, hr


def get_training_data(
    num: int = 3200,
    valid_num: int = 100,
    random_transform: bool = True,
    batch_size: int = 1,
):
    print("loading validation...")
    the_lr_vl, the_hr_vl = load_dataset(
        num=valid_num,
        type="valid",
        batch_size=batch_size,
        random_transform=random_transform,
    )
    print("finished loading validation")

    print("loading training...")
    the_lr_tr, the_hr_tr = load_dataset(
        num=num, type="train", batch_size=batch_size, random_transform=random_transform
    )
    print("finished loading training")

    return the_lr_tr, the_hr_tr, the_lr_vl, the_hr_vl


def dataset_from_folder(input: str):
    import os
    import tensorflow as tf
    from tensorflow.python.data.experimental import AUTOTUNE

    images_dir = os.path.join(input)
    image_files = [
        os.path.join(images_dir, filename)
        for filename in sorted(os.listdir(images_dir))
    ]

    ds = tf.data.Dataset.from_tensor_slices(image_files)
    ds = ds.map(tf.io.read_file)
    ds = ds.map(
        lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(1)

    return ds


def config(use_gpu: bool, vgg_problems: bool = False):
    import tensorflow as tf

    if vgg_problems:
        tf.config.experimental_run_functions_eagerly(True)
        tf.config.run_functions_eagerly(True)
    if use_gpu:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        tf.config.set_visible_devices([], "GPU")
