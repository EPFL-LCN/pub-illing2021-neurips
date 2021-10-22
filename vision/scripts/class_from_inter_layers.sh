# Script for evaluating intermediate layers through linear classification from hidden layers.

#!/bin/sh

savepath=./logs/YOURMODEL
n_epochs=299 # 599
n_in_channels=(128 256 256 512 1024 1024)

for i in 1 2 3 4 5 6
do
    echo "Testing the model for linear image classification from layer/module $i"
    python -m GreedyInfoMax.vision.downstream_classification --model_path $savepath --model_num $n_epochs --encoder_type 'vgg_like' --model_splits 6 --train_module 6 --module_num $i --in_channels ${n_in_channels[$i-1]}
    mv $savepath/classification_results.txt  $savepath/classification_results_layer_$i.txt
    mv $savepath/classification_results_values.npy  $savepath/classification_results_values_$i.npy
done