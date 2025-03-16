
#export nnUNet_N_proc_DA=36
export nnUNet_N_proc_DA=36
export nnUNet_codebase="/public/home/lixueping/backups/nnUNet-1.7.1" # replace to your codebase
export nnUNet_raw_data_base="/public/home/lixueping/backups/nnUNet-1.7.1/DATASET/nnUNet_raw" # replace to your database
export nnUNet_preprocessed="/public/home/lixueping/backups/nnUNet-1.7.1/DATASET/nnUNet_preprocessed"
export RESULTS_FOLDER="/public/home/lixueping/backups/nnUNet-1.7.1/DATASET/nnUNet_trained_models"


CONFIG=/public/home/lixueping/backups/3D-TransUNet/configs/Atlas/p3_SwinUNETR_with_PWAM.yaml
# CONFIG=/media/lichangyong/Flash/服务器文件备份/lxp_temp/3D-TransUNet/configs/Atlas/p3_SwinUNETR_with_PWAM.yaml
fold=3
echo "run on fold: ${fold}"
nnunet_use_progress_bar=1 CUDA_VISIBLE_DEVICES=1 \
        python3 ./train_0726.py --fold=${fold} --config=$CONFIG --resume=''




# python train_0726.py --fold=0 --config=configs/Atlas/p3_SwinUNETR_with_PWAM.yaml --resume='' -c

