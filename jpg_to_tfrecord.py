#coding:utf-8
import tensorflow as tf
import os
import os.path
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mode = "train"
# mode = "eval"

rootdir = "C:/Users/VIPLAB/Desktop/dog_vs_cat_detection/dataset/self_divide/" + mode

TFRfilename = "C:/Users/VIPLAB/Desktop/dog_vs_cat_detection/dataset/TFRECORD/dog_cat_"+ mode + ".tfrecords"

defined_label = [
'cat',
'dog'
]
row = 256
col = 256

# get the labelID (0 ~ category_num -1) or -1 if label not found
def convert_filename_to_labelID(filename,defined_label):
    # get the label numbers
    label_num = len(defined_label)
    labelid = -1;
    # loop the defined labels to find the label name that matches current filename
    for i in range(0,label_num):
       if defined_label[i] in filename:
            labelid=i
            break
    return labelid


writer = tf.python_io.TFRecordWriter(TFRfilename)
count=0
with tf.Session() as sess:
    for parent,dirnames,filenames in os.walk(rootdir):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for filename in filenames:                        #输出文件信息
        
           if "jpg" in filename:
              labelID = convert_filename_to_labelID(filename,defined_label)
    
              if (labelID>=0) and (labelID<len(defined_label)):
    
                  image_dir = parent+"\\"+filename
                  image_raw_data_jpg = tf.gfile.FastGFile(image_dir, 'rb').read()
                  img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
                  img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)
                  resized_image = tf.image.resize_images(img_data_jpg, [row, col])
                  image_raw_data = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
                  
                  
                  if(len(image_raw_data)==0):
                     continue
                  example = tf.train.Example(features=tf.train.Features(feature={
                      # 包装为可以训练的数据
                      'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labelID])),
                      'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw_data]))
                         
                  }))
                  count=count+1
                  # print("文件"+filename+"生成成功，已生成%d个文件"%count)
                  writer.write(example.SerializeToString())
    writer.close()
    # print ("TFRecord文件已保存。共%d个文件"%count)    
    print ("TFRecord save。count = %d"%count)    