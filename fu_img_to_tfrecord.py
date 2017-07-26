import os 
import os.path 
import tensorflow as tf 
from PIL import Image  
import matplotlib.pyplot as plt 
import sys
import pprint
pp = pprint.PrettyPrinter(indent = 2)

data_dir=sys.argv[1]
train_dir=sys.argv[2]
classes=[]
for dir in os.listdir(data_dir):
    path = os.path.join(data_dir, dir)
    if os.path.isdir(path):
        classes.append(dir)


train= tf.python_io.TFRecordWriter(train_dir+"/iss_train.tfrecord") 
test= tf.python_io.TFRecordWriter(train_dir+"/iss_test.tfrecord") 


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={ 
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))

def get_extension(path):
    return os.path.splitext(path)[1] 

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
      # Initializes function that decodes RGB JPEG data.
      self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
      self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
      image = self.decode_jpeg(sess, image_data)
      return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
      image = sess.run(self._decode_jpeg,
                       feed_dict={self._decode_jpeg_data: image_data})
      assert len(image.shape) == 3
      assert image.shape[2] == 3
      return image

def write_label_file(labels_to_class_names, dataset_dir,
                     filename='lables.txt'):
    """Writes a file with the list of class names.

    Args:
      labels_to_class_names: A map of (integer) labels to class names.
      dataset_dir: The directory in which the labels file should be written.
      filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
      for label in labels_to_class_names:
        class_name = labels_to_class_names[label]
        f.write('%d:%s\n' % (label, class_name))

lable_file=train_dir+'/lable.txt'
lable_input=open(lable_file, 'w')

info_file=train_dir+'/meta_info.txt'
test_num=0;
train_num=0;

with tf.Graph().as_default():
    image_reader = ImageReader()
    with tf.Session('') as sess: 

        for index,name in enumerate(classes):
            lable_input.write('%d:%s\n' % (index, name))  
            class_path=data_dir+'/'+name+'/'
            for num, img_name in enumerate(os.listdir(class_path)): 
                img_path=class_path+img_name 
                
                format=get_extension(img_name)
                image_data = tf.gfile.FastGFile(img_path, 'rb').read()
                height, width = image_reader.read_image_dims(sess, image_data)
                example = image_to_tfexample(image_data, b'jpg', height, width, index)
                if num % 5 == 0:
                    test_num= test_num+1
                    #pass
                    #print img_path + " " + str(index) + " " + name
                    test.write(example.SerializeToString()) 
                else:
                    train_num=train_num+1
                    train.write(example.SerializeToString())
                    #print img_path + " " + str(index) + " " + name

train.close()
test.close()

info_input=open(info_file,'w')
info_input.write("train_num:"+str(train_num)+'\n')
info_input.write("test_num:"+str(test_num)+'\n')
info_input.close()

lable_input.close()