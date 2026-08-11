# CONDITIONAL U-NET FOR PCB STRESS HEATMAP PREDICTION

This project trains a 5-level encoder-decoder conditional U-Net architecture to predict stress heatmap images from board-level condition features. The model takes a 6-channel board condition tensor as input and outputs a 3-channel RGB heatmap image representing the predicted stress distribution.

## PROJECT GOAL

The goal of this project is to predict stress heatmaps without giving the model direct stress values as input. Instead, the model learns the relationship between board layout features and the final stress heatmap.

The model learns this mapping:

6-channel board condition tensor  →  RGB stress heatmap


## PREPROCESSED_CONDITIONS/CONDITION_X.PT
    Each condition tensor has shape: 6 x 256 x 64
    Channel 0: material map
    Channel 1: distance to chip
    Channel 2: distance to wire
    Channel 3: distance to board center
    Channel 4: distance to Y edge / vertical location
    Channel 5: edge degree / connectivity
    
     #HOW TO RUN PREPROCESSED_CONDITIONS
    cd "$HOME/Desktop/UNET TEST/Pytorch-UNet-master"
    PYTHONNOUSERSITE=1 /home/nabanga2/miniconda3/envs/test/bin/python preprocess_conditions.py
    
    
    
    
Target Output - DAC_FHE/clustergcn_reg/src/boards_recreated_pixels_5/board_X.png
    
    
#Project Structure   
Pytorch-UNet-master/
│
├── train_board_unet_kfold.py          # Main K-fold training and generation script
├── evaluate_heatmap.py                # Evaluates one predicted heatmap
├── evaluate_all_folds.py              # Evaluates all folds and test boards
├── board_dataset.py                   # Dataset loader for board condition tensors and target heatmap images
├── train_board_unet.py                # Standard direct U-Net training and generation script without K-fold
│
├── unet/
│   ├── __init__.py
│   ├── unet_model.py                  # U-Net architecture
│   └── unet_parts.py                  # U-Net blocks
│
├── preprocessed_conditions/
│   ├── condition_1.pt
│   ├── condition_2.pt
│   └── ...
│
├── DAC_FHE/
│   └── clustergcn_reg/
│       └── src/
│           └── boards_recreated_pixels_5/
│               ├── board_1.png
│               ├── board_2.png
│               └── ...
│
├── unetcon_checkpoints/               # Saved K-fold checkpoints
├── unetcon_images/                    # Training preview images
└── README.md



# PACKAGES TO INSTALL
enviroment_packages.txt (use my packages)
torch
torchvision
numpy
pillow
tqdm
scikit-image



Boards 1–46  = K-fold training/validation pool
Boards 47–52 = final held-out test boards




# 1 is training/testing on one board
# 2 is training/testing using the K-fold

# TRAIN DIRECT U-NET WITHOUT K-FOLD (1.1)
cd "$HOME/Desktop/UNET TEST/Pytorch-UNet-master"

rm -rf unetcon_checkpoints
rm -rf unetcon_images
rm -f board_unet_generated.png
rm -f board_unet_generated_big.png

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/nabanga2/miniconda3/envs/test/bin/python train_board_unet.py \
--mode train \
--target-folder boards_recreated_pixels_5 \
--batch-size 2 \
--epochs 300 \
--lr 0.0001 \
--device cuda




#TO TRAIN ALL 5 K-FOLDS: (2.1)
rm -rf unetcon_checkpoints
rm -rf unetcon_images
rm -f board_*_fold_*_generated.png
rm -f board_*_fold_*_generated_big.png
rm -f board_*_kfold_generated.png
rm -f board_*_kfold_generated_big.png
rm -f board_unet_generated.png
rm -f board_unet_generated_big.png

cd "$HOME/Desktop/UNET TEST/Pytorch-UNet-master"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/nabanga2/miniconda3/envs/test/bin/python train_board_unet_kfold.py \
--mode kfold \
--target-folder boards_recreated_pixels_5 \
--k-folds 5 \
--seed 42 \
--batch-size 2 \
--epochs 300 \
--lr 0.0001 \
--device cuda


#TO GENERATE A PREDICTION FOR ONE BOARD WITHOUT A KFOLD: (1.2)
CKPT=$(find unetcon_checkpoints -type f | sort -V | tail -1)
echo "$CKPT"

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/nabanga2/miniconda3/envs/test/bin/python train_board_unet.py \
--mode generate \
--target-folder boards_recreated_pixels_5 \
--checkpoint "$CKPT" \
--board-num 50 \
--device cuda

#TO GENERATE PREDICTIONS FOR BOARDS 47–52 USING ALL 5 FOLD CHECKPOINTS: (2.2)
cd "$HOME/Desktop/UNET TEST/Pytorch-UNet-master"
for f in 1 2 3 4 5; do
    CKPT="unetcon_checkpoints/kfold_${f}_best.pth"

    for b in 47 48 49 50 51 52; do
        echo "Generating board $b using fold $f"

        PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/nabanga2/miniconda3/envs/test/bin/python train_board_unet_kfold.py \
        --mode generate \
        --target-folder boards_recreated_pixels_5 \
        --checkpoint "$CKPT" \
        --board-num "$b" \
        --device cuda

        mv board_${b}_kfold_generated.png board_${b}_fold_${f}_generated.png
        mv board_${b}_kfold_generated_big.png board_${b}_fold_${f}_generated_big.png
    done
done



#TO EVALUATE ONE GENERATED IMAGE: (1.3)
PYTHONNOUSERSITE=1 /home/nabanga2/miniconda3/envs/test/bin/python evaluate_heatmap.py \
--pred board_unet_generated_big.png \
--true DAC_FHE/clustergcn_reg/src/boards_recreated_pixels_5/board_50.png \
--threshold 0.20


#TO EVALUATE ALL FOLDS AND ALL FINAL TEST BOARDS: (2.3)
PYTHONNOUSERSITE=1 /home/nabanga2/miniconda3/envs/test/bin/python evaluate_all_folds.py \
--boards 47 48 49 50 51 52 \
--folds 1 2 3 4 5 \
--threshold 0.20 | tee evaluation_results_threshold_020.txt


#REPORTS INDIVIDUAL RESULTS AND AVERAGES FOR:
IoU			#Intersection over Union measures how much the predicted heatmap region overlaps with the true heatmap region.
Dice/F1			#Dice score is another overlap metric.
Precision		#Precision measures how much of the predicted heatmap region was actually correct.
Recall			#Recall measures how much of the true heatmap region was captured.	
MSE			#These measure pixel-level error between the predicted and true heatmaps.
MAE			#These measure pixel-level error between the predicted and true heatmaps.
Weighted MAE		#These measure pixel-level error between the predicted and true heatmaps.
Pixel Similarity	#Pixel similarity is based on MAE: Pixel Similarity = 100 * (1 - MAE)
Visual Similarity SSIM	#SSIM measures structural visual similarity between the predicted and true images.
Pearson Correlation	#Pearson correlation checks whether the predicted heat pattern follows the true heat pattern.
Top-10% Hotspot IoU	#This compares the highest-intensity regions in the predicted and true heatmaps.
Centroid Distance	#This measures how far the center of the predicted heatmap region is from the center of the true heatmap region.
