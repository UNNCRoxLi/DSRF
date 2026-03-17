python main.py --dataset-name RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1 --fp16 \
               --image-size 80 --epochs 200 --seed 12345 --batch-size 128 --lr 0.001 --muon-lr 0.003 --wd 1e-5 \
               -a dsrf_raven --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt your_checkpoint_dir \
               --dsrf-scale M\
