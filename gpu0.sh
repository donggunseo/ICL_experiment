# for seed in 38 39 40 41 42
# do
#     CUDA_VISIBLE_DEVICES=0 python3 icl.py --data mnli --seed $seed  --n_shot 8 --max_train_size 10000 --max_dev_size 500 --selection "random_stratify_sampling"
# done

CUDA_VISIBLE_DEVICES=0 python3 icl.py --data mnli --seed 42 --n_shot 8 --max_train_size 10000 --max_dev_size 500 --selection similar
CUDA_VISIBLE_DEVICES=0 python3 icl.py --data mnli --seed 42 --n_shot 8 --max_train_size 10000 --max_dev_size 500 --selection pair_similar
CUDA_VISIBLE_DEVICES=0 python3 icl.py --data mnli --seed 42 --n_shot 8 --max_train_size 10000 --max_dev_size 500 --selection pair_one_similar
CUDA_VISIBLE_DEVICES=0 python3 icl.py --data mnli --seed 42 --n_shot 8 --max_train_size 10000 --max_dev_size 500 --selection pair_one_similar2

