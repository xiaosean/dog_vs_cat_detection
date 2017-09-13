import os
import tensorflow as tf 
import io
from object_detection.utils import dataset_util
from PIL import Image
from time import time
#cwd = os.getcwd()
# new a tensorflow model
sess = tf.Session()
# make a placeholder more flexible
encoded_jpg_ph = tf.placeholder(tf.string, shape=[])
# set resize layer
height = 256
width = 256	
# resizing the image here
decoded_image = tf.image.decode_jpeg(encoded_jpg_ph)
decoded_image_resized = tf.image.resize_images(decoded_image, [height, width]) # this returns float32
decoded_image_resized_uint = tf.cast(decoded_image_resized, tf.uint8)
resize_image   = tf.image.encode_jpeg(decoded_image_resized_uint) # expects uint8
# reset all variables
sess.run(tf.global_variables_initializer())

def create_cat_tf_example(label, label_text, img_path, img_name):
	"""Creates a tf.Example proto from sample cat image.

	Args:
	encoded_cat_image_data: The jpg encoded data of the cat image.

	Returns:
	example: The created tf.Example.
	"""
	
	with tf.gfile.FastGFile(img_path + img_name, 'rb') as fid:
	    encoded_image = fid.read() 

	encoded_image_data = sess.run(resize_image, {encoded_jpg_ph: encoded_image}) #  I think this may not be the right way of doing this
	b_filename = str.encode(img_name)

	image_format = b'jpg'
	xmins = [10.0 / width]
	xmaxs = [(width - 10) / width]
	ymins = [10.0 / height]
	ymaxs = [(height - 10.0) / height]
	# classes_text = [str.encode(label_text)]
	classes_text = []
	if label_text:
		classes_text.append(label_text.encode('utf8'))
	classes = []
	# if label == 1:
	classes.append(int(label))
	# print(classes_text, classes, b_filename)
	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(b_filename),
		'image/source_id': dataset_util.bytes_feature(b_filename),
		'image/encoded': dataset_util.bytes_feature(encoded_image_data),
		# 'image/encoded': dataset_util.bytes_feature(encoded_jpg),
		'image/format': dataset_util.bytes_feature(image_format),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))
	return tf_example


if __name__ == '__main__':
	start_time = time()
	each_batch_time = time()
	
	# collect the dirs
	mode_list = ["train", "eval"]
	for mode in mode_list:
		cwd = "C:/Users/VIPLAB/Desktop/dog_vs_cat_detection/dataset/self_divide/" + mode + "/"
		# classes = ["cat", "dog"]
		classes = ["dog", "cat"]

		writer = tf.python_io.TFRecordWriter("C:/Users/VIPLAB/Desktop/dog_vs_cat_detection/dataset/TFRECORD/" + mode + ".tfrecords")
		for index, name in enumerate(classes):
			class_path = cwd + name + "/"
			for img_count, img_name in enumerate(os.listdir(class_path)):
				if (img_count % 100 == 0):
					output_str = mode + " step -- " + str(img_count)
					print(output_str, " compute 100 image _ batch time = ", time() - each_batch_time)
					print("id =", int(index+1),name, img_name, class_path)
					each_batch_time = time()
					# sess.close()
					# # reset session otherwise it will run slowly
					# tf.reset_default_graph()
					# sess = tf.Session()
				# img_path = class_path + img_name
				each_record = create_cat_tf_example(label = index + 1 , label_text = name, img_path = class_path, img_name = img_name)

				# if(name == "dog"):
					# each_record = create_cat_tf_example(label = 1, label_text = name, img_path = class_path, img_name = img_name)
				# else:
					# each_record = create_cat_tf_example(label = None, label_text = None, img_path = class_path, img_name = img_name)

				writer.write(each_record.SerializeToString())  #序列化为字符串
		writer.close()
		print(mode , "is finished.")
	sess.close()
	print("cost time =", time() - start_time)