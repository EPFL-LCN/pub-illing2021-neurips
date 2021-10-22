# Commands to 
# (i) train a model with Hinge Loss CPC, the end-to-end version of CLAPP (use 600 epochs because of asynchronous positive and negative updates)
# (ii) evaluate the trained model with linear downstream classification on last layer

#!/bin/sh

echo "Training the model on vision data (stl-10)"
python -m GreedyInfoMax.vision.main_vision --download_dataset --save_dir HingeLossCPC --num_epochs 600 --encoder_type 'vgg_like' --model_splits 1 --train_module 1 --contrast_mode 'hinge' --negative_samples 1 --sample_negs_locally --sample_negs_locally_same_everywhere --either_pos_or_neg_update --asymmetric_W_pred

echo "Testing the model for image classification"
python -m GreedyInfoMax.vision.downstream_classification --model_path ./logs/HingeLossCPC --model_num 599 --encoder_type 'vgg_like' --model_splits 1 --train_module 1 --module_num 1 --asymmetric_W_pred

