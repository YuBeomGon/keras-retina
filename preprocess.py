# Preprocessing the images involves two steps:

# Resizing the image: Images are resized such that the shortest size is equal to 800 px, after resizing if the longest side of the image exceeds 1333 px, the image is resized such that the longest size is now capped at 1333 px.
# Applying augmentation: Random scale jittering and random horizontal flipping are the only augmentations applied to the images.
# Along with the images, bounding boxes are rescaled and flipped if required.

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from box_utils import swap_xy, convert_to_xywh
import albumentations as A
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate
)

def switch_image(img) :
    h, w = img.shape[:2]
    if (h, w) == (4032, 1960) or (h, w) == (4000, 1800) :
        img = np.flip(img, 1)
        img = np.transpose(img, (1, 0, 2))      
    return img

transforms = A.Compose([
    A.CenterCrop(1408,1408, True,1),
    A.Resize(320, 320, p=1),
], p=1.0, bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.8)) 

train_transforms = A.Compose([
    A.CenterCrop(1280,1280, True,1),
    A.Resize(320, 320, p=1),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.RandomResizedCrop(height=320,width=320,scale=[0.75,1.0],ratio=[0.75,1.33],p=0.3),
#     A.ColorJitter(p=0.3),
#     A.GaussianBlur(p=0.1)
], p=1.0, bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.8))

val_transforms = A.Compose([
    A.CenterCrop(1280,1280, True,1),
    A.Resize(320, 320, p=1),
], p=1.0, bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.8))


def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio


def preprocess_data(sample):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    print('sample', type(sample))
    print(sample.keys())
    print(sample['objects'].keys())
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image, image_shape, _ = resize_and_pad_image(image)

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id

def paps_process(image, bbox, label):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
#     bbox = swap_xy(bbox)
    image, bbox = random_flip_horizontal(image, bbox)
    class_id = tf.cast(label, dtype=tf.int32)

    bbox = tf.stack(
        [
            bbox[:, 0],
            bbox[:, 1],
            bbox[:, 2],
            bbox[:, 3],            
        ],
        axis=-1,
    )
    
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id