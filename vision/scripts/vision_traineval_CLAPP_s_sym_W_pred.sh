# Commands to 
# (i) train a model with CLAPP-s, i.e. synchronous positive and negative updates. Negatives are sampled from all locations and transposed W_pred is used (not fully local compared to CLAPP)
# (ii) evaluate the trained model with linear downstream classification on last layer

#!/bin/sh

echo "Training the model on vision data (stl-10)"
python -m GreedyInfoMax.vision.main_vision --download_dataset --save_dir CLAPP_s --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --contrast_mode 'hinge'

echo "Testing the model for image classification"
python -m GreedyInfoMax.vision.downstream_classification --model_path ./logs/CLAPP_s --model_num 299 --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --module_num 6
