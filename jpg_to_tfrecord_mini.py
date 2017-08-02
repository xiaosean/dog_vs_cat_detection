import os
import tensorflow as tf 
import io
from object_detection.utils import dataset_util
from PIL import Image
#cwd = os.getcwd()

def create_cat_tf_example(label, label_text, img_path, img_name):
	"""Creates a tf.Example proto from sample cat image.

	Args:
	encoded_cat_image_data: The jpg encoded data of the cat image.

	Returns:
	example: The created tf.Example.
	"""
	height = 256
	width = 256
	with tf.gfile.FastGFile(img_path + img_name, 'rb') as fid:
	    encoded_jpg = fid.read() 

	# encoded_jpg_io = io.BytesIO(encoded_jpg)
	# image = Image.open(encoded_jpg_io)
	# width, height = image.size
	# resizing the image here
	decoded_image = tf.image.decode_jpeg(encoded_jpg)
	decoded_image_resized = tf.image.resize_images(decoded_image, [height, width]) # this returns float32
	decoded_image_resized = tf.cast(decoded_image_resized, tf.uint8)
	encoded_jpg   = tf.image.encode_jpeg(decoded_image_resized) # expects uint8
	encoded_image_data = tf.Session().run(encoded_jpg) #  I think this may not be the right way of doing this


	# with tf.Session() as sess:  
	# 	image_raw_data_jpg = tf.gfile.FastGFile(img_path + img_name, 'rb').read()  
	# 	img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
	# 	img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.float32)  
	# 	resized_image = tf.image.resize_images(img_data_jpg, [height, width])  
	# 	encoded_image_data = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()  
	# # img = Image.open(img_path + img_name)
	# img = img.resize((256,256))
	b_filename = str.encode(img_name)
	# encoded_image_data = img.tobytes()            #将图片转化为原生bytes
	# example = tf.train.Example(features=tf.train.Features(feature={
	#     "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
	#     'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
	# }))


	image_format = b'jpg'

	xmins = [30.0 / width]
	xmaxs = [(width - 30) / width]
	ymins = [30.0 / height]
	ymaxs = [(height - 30.0) / height]
	# classes_text = [str.encode(label_text)]
	classes_text = [label_text.encode('utf8')]
	classes = []
	classes.append(int(label))
	print(classes, classes_text)

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
	mode_list = ["train", "eval"]
	# mode_list = ["train"]
	for mode in mode_list:
		cwd = "C:/Users/VIPLAB/Desktop/dog_vs_cat_detection/dataset/self_divide/" + mode + "/"
		classes = ["cat", "dog"]
		writer = tf.python_io.TFRecordWriter("C:/Users/VIPLAB/Desktop/dog_vs_cat_detection/dataset/TFRECORD/" + mode + ".tfrecords")
		for index, name in enumerate(classes):
			class_path = cwd + name + "/"
			for img_count, img_name in enumerate(os.listdir(class_path)):
				if (img_count % 50 == 49):
					output_str = mode + " step -- " + str(img_count)
					print(output_str)
					break
				if (img_count % 100 == 0):
					output_str = mode + " step -- " + str(img_count)
					print(output_str)

				# img_path = class_path + img_name
				each_record = create_cat_tf_example(label = index + 1, label_text = name, img_path = class_path, img_name = img_name)
				writer.write(each_record.SerializeToString())  #序列化为字符串
		writer.close()
		print(mode , "is finished.")