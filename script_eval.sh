
python main.py --dataset-name I-RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1 \
               --image-size 80 -a dsrf_raven \
               -e --resume your_checkpoint_dir/model_best.pth.tar \
               --show-detail