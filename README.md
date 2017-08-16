Use Google detection API:<br>
	https://github.com/tensorflow/models/tree/master/object_detection<br>

dataset- Dogs vs. Cats Redux: Kernels Edition [Kaggle]:<br>
	https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data <br>

The train folder contains 25,000 images of dogs and cats. <br>
Each image in this folder has the label as part of the filename.<br>
The test folder contains 12,500 images, named according to a numeric id.<br>
For each image in the test set, you should predict a probability that the image is a dog (1 = dog, 0 = cat).<br>

<h3>step1. let dataset divide to each group</h3><br>
	<br>
	----train<br>
		+ dog<br>
		+ cat<br>

<h3>step2. image transform to tfrecord </h3><br>
	python jpg_to_tfrecord_mini.py <br>


<h3>step3. modify config </h3><br>
	 \xiao_pet_res\xiao_resnet50_pet.config <br>

<h3>step4. image transform to tfrecord </h3><br>
	cd C:\Users\VIPLAB\Desktop\models

	python object_detection/train.py --logtostderr --pipeline_config_path=C:\Users\VIPLAB\Desktop\dog_vs_cat_detection\xiao_pet_res\xiao_resnet50_pet.config --train_dir=C:\Users\VIPLAB\Desktop\dog_vs_cat_detection\xiao_pet_res\train
 <br>
