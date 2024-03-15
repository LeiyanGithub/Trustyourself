# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 0 --task query_rewrite --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target Qwen1.5-14B-Chat --pos_num 0 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 

# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --source Qwen1.5-14B-Chat --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 0 --task rewrite_qa --promptfile myprompt --nums 500 
CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --source Qwen1.5-14B-Chat --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --source Qwen1.5-14B-Chat --target Qwen1.5-14B-Chat --pos_num 0 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 