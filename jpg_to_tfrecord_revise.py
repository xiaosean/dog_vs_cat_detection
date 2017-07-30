#coding:utf-8  
import tensorflow as tf  
import os  
import os.path  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
  
def _int64_feature(value):  
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  
  
def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  
  
#cwd = os.getcwd()
mode = "train"
cwd = "C:/Users/VIPLAB/Desktop/dog_vs_cat_detection/dataset/self_divide/" + mode + "/"
rootdir = cwd
TFRfilename = "C:/Users/VIPLAB/Desktop/dog_vs_cat_detection/" + mode + ".tfrecords"
classes = ["cat", "dog"]


# http://blog.csdn.net/zsg2063/article/details/75646677
  
  
writer = tf.python_io.TFRecordWriter(TFRfilename)  
count=0  
with tf.Session() as sess:  
    for index, name in enumerate(classes):
        class_path = cwd + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            if "jpg" in img_name:  
          
                image_raw_data_jpg = tf.gfile.FastGFile(img_path, 'rb').read()  
                img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)  
                img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)  
                resized_image = tf.image.resize_images(img_data_jpg, [256, 256])  
                image_raw_data = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()  
                    
                    
                if(len(image_raw_data)==0):  
                    continue  
                example = tf.train.Example(features=tf.train.Features(feature={  
                  # 包装为可以训练的数据  
                  'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),  
                  'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw_data]))  
                       
                }))  
                count=count+1  
                if(count % 100 == 0):
                    print("success count = %d " % count)  
                writer.write(example.SerializeToString())  
    print("success count = %d " % count)  

    writer.close()  
    print ("TFRecord done count = %d" % count) 