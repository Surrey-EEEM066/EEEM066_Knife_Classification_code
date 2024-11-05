#!/bin/bash

STUDENT_ID=xxxxxxx STUDENT_NAME="xxxxxx" python main.py \
--model_mode tf_efficientnet_b0 \
--dataset_location ../EEEM066_Knife_Classification_dataset \
--train_datacsv dataset/train.csv \
--test_datacsv dataset/test.csv \
--saved_checkpoint_path Knife-Effb0 \
--epochs 20 \
--batch_size 32 \
--n_classes 192 \
--learning_rate 0.00005 \
--resized_img_weight 224 \
--resized_img_height 224 \
--seed 0 \
--brightness 0.2 \
--contrast 0.2 \
--saturation 0.2 \
--hue 0.2 \
--optim rmsprop \
--lr-scheduler CosineAnnealingLR
