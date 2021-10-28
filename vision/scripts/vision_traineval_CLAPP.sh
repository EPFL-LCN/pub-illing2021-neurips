# Commands to 
# (i) train a model with CLAPP (use 600 epochs because of asynchronous positive and negative updates)
# (ii) evaluate the trained model with linear downstream classification on last layer

#!/bin/sh

echo "Training the model on vision data (stl-10)"
python -m CLAPPVision.vision.main_vision --download_dataset --save_dir CLAPP --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --contrast_mode 'hinge' --asymmetric_W_pred --num_epochs 600 --negative_samples 1 --sample_negs_locally --sample_negs_locally_same_everywhere --either_pos_or_neg_update

echo "Testing the model for image classification"
python -m CLAPPVision.vision.downstream_classification --model_path ./logs/CLAPP --model_num 599 --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --module_num 6 --asymmetric_W_pred
