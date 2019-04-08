# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import functools
import tensorflow as tf
import progressbar
from string import Formatter

tf.enable_eager_execution()

CELEBA_PATH_AMZ = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"
NOISE_SIZE = 100

if not tf.test.is_gpu_available():
    print("No GPU")
else:
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

def custom_progress_text(message):

    message_ = message.replace('(', '{')
    message_ = message_.replace(')', '}')

    keys = [key[1] for key in Formatter().parse(message_)]

    ids = {}
    for key in keys:
        if key is not None:
            ids[key] = float('nan')

    msg = progressbar.FormatCustomText(message, ids)
    return msg


def create_progress_bar(text=None):

    if text is None:
        text = progressbar.FormatCustomText('')
    bar = progressbar.ProgressBar(widgets=[
        progressbar.Percentage(),
        progressbar.Bar(),
        progressbar.AdaptiveETA(), '  ',
        text,
    ])
    return bar


def dowload_images(url=CELEBA_PATH_AMZ):

    path_to_faces = tf.keras.utils.get_file("celebs.zip", url, extract=True)
    img_path = os.path.join(os.path.split(path_to_faces)[0], 'img_align_celeba')

    image_list = []
    for each in os.listdir(img_path):
        full_path = os.path.join(img_path, each)
        if os.path.isfile(full_path):
            image_list.append(full_path)

    return image_list


def create_dataset(image_list, batch_size=50, randomize=True):

    dataset = tf.data.Dataset.from_tensor_slices(image_list)

    # mapping function to load and resize the images
    def load_image(filename, width=56, height=56):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string)
        image = tf.image.convert_image_dtype(image, tf.float32)

        if image.shape[:2] != [width, height]:
            image = tf.image.crop_to_bounding_box(image, 40, 20, 218 - 80, 178 - 40)
            image = tf.image.resize(image, [width, height])

        return image

    dataset = dataset.map(load_image)
    dataset = dataset.repeat()

    if randomize:
        dataset = dataset.shuffle(len(image_list))

    if batch_size > 0:
        dataset = dataset.batch(batch_size)

    return dataset


def create_generator(noise_shape=(100, 0), show=False):

    BatchNormalization = tf.keras.layers.BatchNormalization
    Dense = tf.keras.layers.Dense
    Conv2DTranspose = tf.keras.layers.Conv2DTranspose
    LeakyReLU = tf.keras.layers.LeakyReLU

    model = tf.keras.Sequential([
        Dense(4 * 4 * 1024, use_bias=False, input_shape=noise_shape, activation='relu'),
        BatchNormalization(),
        tf.keras.layers.Reshape((4, 4, 1024)),

        Conv2DTranspose(filters=512, kernel_size=[4, 4], strides=1),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(filters=256, kernel_size=[5, 5], strides=2, padding="same"),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(filters=128, kernel_size=[5, 5], strides=2, padding="same"),

        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(filters=3, kernel_size=[5, 5], strides=2, padding="same")

    ], name="Generator")

    if show:
        print(model.summary())

    return model


def create_discriminator(show=False, drop_rate=0.3):

    BatchNormalization = tf.keras.layers.BatchNormalization
    Dense = tf.keras.layers.Dense
    Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same')  
    LeakyReLU = tf.keras.layers.LeakyReLU
    DropOut = tf.keras.layers.Dropout

    model = tf.keras.Sequential([
        Conv2D(filters=64, kernel_size=[5, 5], strides=[2, 2], input_shape=(56, 56, 3)),
        BatchNormalization(),
        LeakyReLU(),
        DropOut(drop_rate),

        Conv2D(filters=128, kernel_size=[5, 5], strides=[2, 2]),
        BatchNormalization(),
        LeakyReLU(),
        DropOut(drop_rate),

        Conv2D(filters=256, kernel_size=[5, 5], strides=[2, 2]),
        BatchNormalization(),
        LeakyReLU(),
        DropOut(drop_rate),

        tf.keras.layers.Flatten(name="Disc_Flat"),

        Dense(1, name="Disc_Logit")

    ], name="Discriminator")
    if show:
        print(model.summary())
    return model


def train_model(nb_epochs, batch_size, disp_freq, save_freq, destination, learning_rate = 2.5e-4, beta1 = 0.45):

    iamges_list = dowload_images()
    nb_images = len(iamges_list)

    dataset = create_dataset(iamges_list, batch_size=batch_size)

    iterator = dataset.make_one_shot_iterator()

    iterator.get_next().shape

    generator = create_generator((NOISE_SIZE,))
    discriminator = create_discriminator()

    gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1,
                                           name="Generator_Optimizer")  # define our optimizer
    disc_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1,
                                            name="Discriminator_Optimizer")  # define our optimizer

    checkpoint = tf.train.Checkpoint(gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer, generator=generator,
                                     discriminator=discriminator)
    checkpoint_prefix = os.path.join(destination, "ckpt")

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))

    gen_labels = np.ones((batch_size, 1)).astype(np.float32)
    disc_labels = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))]).astype(np.float32)

    for epoch in range(nb_epochs):

        custom_msg = custom_progress_text("Epoch: %(epoch).0f Gen loss: %(gen_loss)2.2f Disc loss: %(disc_loss)2.2f")
        bar = create_progress_bar(custom_msg)

        for idx in bar(range(nb_images // batch_size)):
            batch = iterator.get_next()
            # draw a batch of random
            noise = np.random.uniform(-1, 1, (batch_size, NOISE_SIZE)).astype(np.float32)

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_images = generator(noise)

                gen_logits = discriminator(gen_images)
                gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gen_labels, logits=gen_logits)
                # print("loss shape {}".format(gen_loss.shape))

                real_logits = discriminator(batch)

                disc_logits = tf.concat([gen_logits, real_logits], axis=0)
                disc_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=disc_labels, logits=disc_logits)

            grads = gen_tape.gradient(gen_loss, generator.variables)
            gen_optimizer.apply_gradients(zip(grads, generator.variables))

            grads = disc_tape.gradient(disc_loss, discriminator.variables)
            disc_optimizer.apply_gradients(zip(grads, discriminator.variables))

            custom_msg.update_mapping(epoch=epoch, gen_loss=gen_loss.numpy().mean(), disc_loss=disc_loss.numpy().mean())

            if (idx + 1) % (nb_images // batch_size // disp_freq) == 0:
                noise = np.random.uniform(-1, 1, (25, NOISE_SIZE)).astype(np.float32)
                gen_images = generator(noise)
                plt.figure(figsize=(10, 10))
                for i in range(25):
                    plt.subplot(5, 5, i + 1)
                    plt.imshow(gen_images[i])
                    plt.grid(False)
                    plt.axis("off")
                filename = os.path.join(destination, "Picture-{}-{}.png".format(epoch, idx))
                plt.savefig(filename)

            if (idx + 1) % (nb_images // batch_size // save_freq) == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train gan on celebrity faces.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--disp-freq', type=int, default=1)
    parser.add_argument('--save-freq', type=int, default=2)
    parser.add_argument('--dest', required=True)

    args = parser.parse_args()

    assert(os.path.isdir(args.dest))

    train_model(nb_epochs=args.epochs, batch_size=args.batch_size, disp_freq=args.disp_freq,
                save_freq=args.save_freq, destination=args.dest)