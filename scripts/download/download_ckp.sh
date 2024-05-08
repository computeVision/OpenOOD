# sh ./scripts/download/dowanload.sh

# download the up-to-date benchmarks and checkpoints
# provided by OpenOOD v1.5
python ./scripts/download/download.py \
	--contents  'checkpoints' \
	--checkpoints 'imagenet200_res18_v1.5' \
	--save_dir './data' './results' \
	--dataset_mode 'benchmark'
