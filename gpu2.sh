# for seed in 38 39 40 41 42
# do
#     CUDA_VISIBLE_DEVICES=2 python3 icl.py --data mrpc --seed $seed  --n_shot 8 --selection "random_stratify_sampling"
# done

# CUDA_VISIBLE_DEVICES=2 python3 icl.py --data mrpc --seed 42 --n_shot 8 --selection similar
# CUDA_VISIBLE_DEVICES=2 python3 icl.py --data mrpc --seed 42 --n_shot 8 --selection pair_similar
# CUDA_VISIBLE_DEVICES=2 python3 icl.py --data mrpc --seed 42 --n_shot 8 --selection pair_one_similar
# CUDA_VISIBLE_DEVICES=2 python3 icl.py --data mrpc --seed 42 --n_shot 8 --selection pair_one_similar2
CUDA_VISIBLE_DEVICES=2 python3 icl.py --data mrpc --seed 42 --selection zero_shot