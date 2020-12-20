import tensorflow as tf
from tensorflow.contrib import slim
import cv2
import os, random
import numpy as np

class ImageData:

    def __init__(self, load_size, channels, augment_flag):
        self.load_size = load_size#64
        self.channels = channels#3
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        #读取图片
        
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        #将图像使用JPEG的格式解码从而得到图像对应的三维矩阵
        #解码后是一个tensor张量
        
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1
        #将其转化成float格式
        #为什么要/127.5 ，将其映射到-1到1之间

        if self.augment_flag :
            #augment_flag=True
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            #augment_size=15
            p = random.random()
            #随机一个p，在（0，1）之间？
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img

def load_test_data(image_path, size=256):
    img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, dsize=(size, size))

    img = np.expand_dims(img, axis=0)
    img = img/127.5 - 1

    return img

def augmentation(image, augment_size):
    #**图片增强
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    #随机水平翻转图片？？？
    image = tf.image.resize_images(image, [augment_size, augment_size])
    #放大图片（原大小+augment_size）
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    #随机裁剪图片？？？
    return image

def save_images(images, size, image_path):
    print('shape1:',np.array(images[0]).shape)
    return imsave(inverse_transform(images[0]), size, image_path)

def inverse_transform(images):
    return (np.array(images)+1./2)*255.0


def imsave(images, size, path):
    images=np.squeeze(merge(images, size))
    print('shape2:',np.array(images).shape)
    #scipy.misc.imsave(path, image)
    images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)

    return cv2.imwrite(path, images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  print('images.shape:',images.shape)
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

# def merge(images, size):
    # h, w = images.shape[1], images.shape[2]
    # img = np.zeros((h * size[0], w * size[1], 3))
    # for idx, image in enumerate(images):
        # i = idx % size[1]
        # j = idx // size[1]
        # img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    # return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')
