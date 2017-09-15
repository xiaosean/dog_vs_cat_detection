Use Google detection API:<br>
	https://github.com/tensorflow/models/tree/master/object_detection<br>

dataset- Dogs vs. Cats Redux: Kernels Edition [Kaggle]:<br>
	https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data <br>

The train folder contains 25,000 images of dogs and cats. <br>
Each image in this folder has the label as part of the filename.<br>
The test folder contains 12,500 images, named according to a numeric id.<br>
For each image in the test set, you should predict a probability that the image is a dog (1 = dog, 0 = cat).<br>

<h3>step1. prepare the dataset</h3><br>
<br>
	here can download dataset.<br>
	https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data<br>
	+ dataset<br>
		----train<br>
			+ dog<br>
			+ cat<br>
	

<h3>step1. let dataset divide to each group, use split_train_data.ipynb</h3><br>
	it will add the dirs like below.<br>
	+ dataset<br><br>
	    ----self_divide (new use split_train_data.ipynb and then u got it.)<br>
			----train<br>
				+ dog<br>
				+ cat<br>
			----eval<br>
				+ dog<br>
				+ cat<br>
		----train<br>
			+ dog<br>
			+ cat<br>
<h3>[optional] use coco model to recongnize it is dog or cat </h3><br>
	auto_bounding_use_coco.ipynb <br>
	u will get a dir - auto_box<br>
	----auto_box (new)<br>
		auto_box.csv<br>
		multi_box_filename.csv<br>
		+box_image<br>
			*.jpg (it will show the coco model box the picture)<br>
<h3>step2. image transform to tfrecord </h3><br>
	to_tfrecord_load_box.ipynb <br>
	u will get tfrecords<br>
	----dataset<br>
		+TFRECORD<br>
			train.tfrecords (new)<br>
			eval.tfrecords (new)<br>
			
<h3>step3. modify config </h3><br>
	 \xiao_pet_res\xiao_resnet50_pet.config <br>
	 u must change input_path and label_map_path <br>

	 
<h3>step4. train your model </h3><br>
	cd $YOUR_MODELS_PATH\models  <br>
	[it was clone google tensorflow -> https://github.com/tensorflow/models]<br>


	****below is my path u should change to yours<br>

	python object_detection/train.py --logtostderr --pipeline_config_path=C:\Users\VIPLAB\Desktop\dog_vs_cat_detection\xiao_pet_res\xiao_resnet50_pet.config --train_dir=C:\Users\VIPLAB\Desktop\dog_vs_cat_detection\xiao_pet_res\train
	<br>
 <br>
