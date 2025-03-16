export nnUNet_N_proc_DA=36
#export nnUNet_codebase="/media/lichangyong/Flash/服务器文件备份/lxp_temp/nnUNet-1.7.1" # replace to your codebase
#export nnUNet_raw_data_base="/media/lichangyong/Flash/服务器文件备份/lxp_temp/nnUNet-1.7.1/DATASET/nnUNet_raw" # replace to your database
#export nnUNet_preprocessed="/media/lichangyong/Flash/服务器文件备份/lxp_temp/nnUNet-1.7.1/DATASET/nnUNet_preprocessed"
#export RESULTS_FOLDER="/media/lichangyong/Flash/服务器文件备份/lxp_temp/nnUNet-1.7.1/DATASET/nnUNet_trained_models"

#export nnUNet_codebase="F:/服务器文件备份/lxp_temp/nnUNet-1.7.1" # replace to your codebase
#export nnUNet_raw_data_base="F:/服务器文件备份/lxp_temp/nnUNet-1.7.1/DATASET/nnUNet_raw" # replace to your database
#export nnUNet_preprocessed="F:/服务器文件备份/lxp_temp/nnUNet-1.7.1/DATASET/nnUNet_preprocessed"
#export RESULTS_FOLDER="F:/服务器文件备份/lxp_temp/nnUNet-1.7.1/DATASET/nnUNet_trained_models"


SET nnUNet_codebase=F:/backups/lxp_temp/nnUNet-1.7.1
SET nnUNet_raw_data_base=F:/backups/lxp_temp/nnUNet-1.7.1/DATASET/nnUNet_raw
SET nnUNet_preprocessed=F:/backups/lxp_temp/nnUNet-1.7.1/DATASET/nnUNet_preprocessed
SET RESULTS_FOLDER=F:/backups/lxp_temp/nnUNet-1.7.1/DATASET/nnUNet_trained_models

config='configs/Atlas/p3_SwinUNETR_with_PWAM.yaml'
export save_folder=./results/inference/test/Atlas01/p3_SwinUNETR_with_PWAM

# python inference_0726.py --config=configs/Atlas/p3_SwinUNETR_with_PWAM.yaml --fold=0 --save_folder=D:\



NUM_GPUS=1

inference() {
	fold=$1
	gpu=$2
	extra=${@:3}

	echo "inference: fold ${fold} on gpu ${gpu}, extra ${extra}"
	CUDA_VISIBLE_DEVICES=${gpu} \
		python3 inference_0726.py --config=${config} \
		--fold=${fold} --raw_data_folder ${subset} \
		--save_folder=${save_folder}/fold_${fold} ${extra}
}

compute_metric() {
	fold=$1
	pred_dir=${2:-${save_folder}/fold_${fold}/}
	extra=${@:3}

	echo "compute_metric: fold ${fold}, extra ${extra}"
	python3 measure_dice.py \
		--config=${config} --fold=${fold} \
		--raw_data_dir=${raw_data_dir} \
		--pred_dir=${pred_dir} ${extra}
}

#  python measure_dice.py --config=configs/Atlas/p3_SwinUNETR_with_PWAM.yaml --fold=0 --pred_dir=results/inference/test/Atlas02/p3_SwinUNETR_with_PWAM/fold_0/



fold=$1
if [[ ${fold} == "all" ]]; then
	gpu=${2:-${gpu}}
else
	gpu=$((${fold} % ${NUM_GPUS}))
	gpu=${2:-${gpu}}
fi
extra=${@:3}

echo "extra: ${extra}"

# 5 fold eval
subset='imagesTr'

inference ${fold} ${gpu} ${extra}
# compute_metric ${fold}

echo "finished: inference: fold ${fold} on ${config}"
exit

# # test set eval
# subset='imagesTs'
inference ${fold} ${gpu} ${extra} --disable_split
compute_metric ${fold} ${save_folder}/fold_${fold}/ --eval_mode Ts

# multi_save_folder=./results/inference/test/task201/encoderonly/fold_${fold},./results/inference/test/task201/decoderonly/fold_${fold}
# compute_metric ${fold} ${multi_save_folder} --eval_mode Ts
