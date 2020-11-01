from imageio import imread
import numpy as np
import skimage.color as skimage
from tensorflow.python.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from sol5_utils import *
from scipy import ndimage

LARGE_CROP = 3
NORM = 0.5
SPLIT = 0.8
CONV_KERNEL_SIZE = (3, 3)
GRAY_SCALE = 1
NUM_PIXELS = 255
NUM_PIXELS_D = 255.0
ROW = 0
COL = 1


def read_image(filename, representation):
    """
    This function reads an image file and converts it into a given representation.
    :param filename: the filename of an image on disk(could be grayscale or RGB)
    :param representation: representation code, either 1 or 2 defining whether the
        output should be a grayscale image(1) or an RGB image(2).
    :return: An image, represented by a matrix of type np.float64 with intensities.
    """
    image = imread(filename)

    # checks if the image is already from type float64
    if not isinstance(image, np.float64):
        image.astype(np.float64)
        image = image / NUM_PIXELS

    # checks if the output image should be grayscale
    if representation == GRAY_SCALE:
        image = skimage.rgb2gray(image)
    return image


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    This function builds a dataset of pairs of cropped image batches, original and corrupted.
    :param filenames:  A list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
        and returns a randomly corrupted version of the input image.
    :param crop_size:  A tuple (height, width) specifying the crop size of the patches to extract.
    :return: A Python’s generator object which outputs random tuples of the form (source_batch, target_batch), where
    each output variable is an array of shape (batch_size, height, width, 1), target_batch is made of clean images, and
    source_batch is their respective randomly corrupted version according to corruption_func(im)
    """
    image_dict = {}
    while True:
        clean_patches = np.zeros(shape=(batch_size, crop_size[ROW], crop_size[COL], 1))
        corrupt_patches = np.zeros(shape=(batch_size, crop_size[ROW], crop_size[COL], 1))
        for i in range(batch_size):

            # randomly chose an image, and read it using the image dictionary
            rand_image = np.random.randint(len(filenames))
            if filenames[rand_image] not in image_dict:
                image_dict[filenames[rand_image]] = read_image(filenames[rand_image], GRAY_SCALE)[..., np.newaxis]
            image = image_dict[filenames[rand_image]]

            # randomly chose a large patch index
            rand_x_index_big_crop = np.random.randint(image.shape[ROW] - (LARGE_CROP * crop_size[ROW]))
            rand_y_index_big_crop = np.random.randint(image.shape[COL] - (LARGE_CROP * crop_size[COL]))

            # crop the image according to the matching patch
            large_crop_image = image[rand_x_index_big_crop:rand_x_index_big_crop + (LARGE_CROP * crop_size[ROW]),
                               rand_y_index_big_crop:rand_y_index_big_crop + (LARGE_CROP * crop_size[COL])]
            large_crop_image_corrupt = corruption_func(large_crop_image)

            # randomly chose a patch index
            rand_x_index_crop = np.random.randint(large_crop_image.shape[ROW] - crop_size[ROW])
            rand_y_index_crop = np.random.randint(large_crop_image.shape[COL] - crop_size[COL])

            # crop the large patch into the wanted patch size
            clean_patch = large_crop_image[rand_x_index_crop:rand_x_index_crop + crop_size[ROW],
                          rand_y_index_crop:rand_y_index_crop + crop_size[COL]]
            corrupt_patch = large_crop_image_corrupt[rand_x_index_crop:rand_x_index_crop + crop_size[ROW],
                            rand_y_index_crop:rand_y_index_crop + crop_size[COL]]

            clean_patches[i] = clean_patch.reshape((clean_patch.shape[ROW], clean_patch.shape[COL], 1)) - NORM
            corrupt_patches[i] = corrupt_patch.reshape((corrupt_patch.shape[ROW], corrupt_patch.shape[COL], 1)) - NORM
        yield (corrupt_patches, clean_patches)


def resblock(input_tensor, num_channels):
    """
    This function takes as input a symbolic input tensor and the number of channels for each of its
    convolutional layers, and returns the symbolic output tensor of the layer configuration described in the exercise.
    :param input_tensor: a symbolic input tensor
    :param num_channels: number of channels for each of its convolutional layers
    :return: the symbolic output tensor of the layer configuration described in the exercise
    """
    conv = Conv2D(num_channels, CONV_KERNEL_SIZE, padding='same')(input_tensor)
    rel = Activation('relu')(conv)
    conv_2 = Conv2D(num_channels, CONV_KERNEL_SIZE, padding='same')(rel)
    return Activation('relu')(Add()([input_tensor, conv_2]))


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    This function returns an untrained Keras model
    :param height: the height shape of the untrained Keras model
    :param width: the width shape of the untrained Keras model
    :param num_channels: the number of output channels, except the very last convolutional layer which should have a
        single output channel
    :param num_res_blocks: the number of residual blocks
    :return: an untrained Keras model
    """
    input_tensor = Input(shape=(height, width, 1))
    conv = Conv2D(num_channels, CONV_KERNEL_SIZE, padding='same')(input_tensor)
    res_output = Activation('relu')(conv)
    for i in range(num_res_blocks):
        res_output = resblock(res_output, num_channels)
    final_conv = Conv2D(1, CONV_KERNEL_SIZE, padding='same')(res_output)
    final = Add()([input_tensor, final_conv])
    return Model(inputs=input_tensor, outputs=final)


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    This function divides the images into a training set and validation set, using an 80-20 split,
    and generate from each set a dataset with the given batch size and corruption function
    :param model:  a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files. You should assume these paths are complete, and
        should append anything to them.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
        and returns a randomly corrupted version of the input image.
    :param batch_size: the size of the batch of examples for each iteration of SGD
    :param steps_per_epoch: The number of update steps in each epoch
    :param num_epochs: The number of epochs for which the optimization will run
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch
    """
    train_images = images[:int(len(images) * SPLIT)]
    validation_images = images[int(len(images) * SPLIT):]
    train_generator = load_dataset(train_images, batch_size, corruption_func, model.input_shape[1:3])
    valid_generator = load_dataset(validation_images, batch_size, corruption_func, model.input_shape[1:3])
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=valid_generator,
                        validation_steps=num_valid_samples, use_multiprocessing=True)


def restore_image(corrupted_image, base_model):
    """
    This function restores full images of any size
    :param corrupted_image: a grayscale image of shape (height, width) and with values in the [0, 1] range of
        type float64
    :param base_model: a neural network trained to restore small patches
    :return: the restored image
    """

    a = Input(shape=(corrupted_image.shape[ROW], corrupted_image.shape[COL], 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    im = corrupted_image.reshape(1, corrupted_image.shape[ROW], corrupted_image.shape[COL], 1) - NORM
    restored_im = (new_model.predict(im)[0] + NORM).clip(0, 1)

    return np.squeeze(restored_im).astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    This function adds a random gaussian noise for a giving image
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing the maximal
        variance of the gaussian distribution
    :return: the corrupted image after adding the noise
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0, sigma, image.shape)
    return ((np.round((image + noise) * NUM_PIXELS)) / NUM_PIXELS_D).clip(0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    This function creates a trained denoising model
    :param num_res_blocks: the number of residual blocks
    :param quick_mode: a boolean indicating whether doing a short or long training
    :return: a trained denoising model
    """
    images = images_for_denoising()
    model = build_nn_model(24, 24, 48, num_res_blocks)
    if quick_mode:
        train_model(model, images, lambda im: add_gaussian_noise(im, 0, 0.2), 10, 3, 2, 30)
    else:
        train_model(model, images, lambda im: add_gaussian_noise(im, 0, 0.2), 100, 100, 5, 1000)
    return model


def add_motion_blur(image, kernel_size, angle):
    """
    This function simulate motion blur on the given image
    :param image: a grayscale image with values in the [0, 1] range of type float64
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: – an angle in radians in the range [0, π).
    :return: a corrupted image
    """
    kernel = motion_blur_kernel(kernel_size, angle)
    if image.ndim == 3:
        kernel = kernel[..., np.newaxis]
    return ndimage.filters.convolve(image, kernel).astype(np.float64)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    This function operates the function add_motion_blur on a giving image with random parameters
    :param image: a grayscale image with values in the [0, 1] range of type float64
    :param list_of_kernel_sizes: a list of odd integers.
    :return:a corrupted image
    """
    angle = np.random.uniform(0, np.pi)
    kernel_size_index = np.random.randint(0, len(list_of_kernel_sizes))
    corrupted_im = add_motion_blur(image, list_of_kernel_sizes[kernel_size_index], angle)
    return ((np.round(corrupted_im * NUM_PIXELS)) / NUM_PIXELS_D).clip(0, 1)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    This function creates a trained deblurring model
    :param num_res_blocks: the number of residual blocks
    :param quick_mode: a boolean indicating whether doing a short or long training
    :return: a trained deblurring model
    """
    images = images_for_deblurring()
    model = build_nn_model(16, 16, 32, num_res_blocks)

    if quick_mode:
        train_model(model, images, lambda im: random_motion_blur(im, [7]), 10, 3, 2, 30)
    else:
        train_model(model, images, lambda im: random_motion_blur(im, [7]), 100, 100, 10, 1000)
    return model

