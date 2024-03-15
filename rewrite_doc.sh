# python main.py --dataset hotpotqa --model gpt-3.5-turbo --task rewrite --promptfile myprompt
# python main.py --dataset hotpotqa --model gpt-3.5-turbo --task rewrite --promptfile myprompt

# python main.py --dataset hotpotqa --model llama-2-7b-chat-hf --promptfile myprompt
# python main.py --dataset hotpotqa --model llama-2-7b-chat-hf --task rewrite --promptfile myprompt


# python main.py --dataset hotpotqa --model llama-2-13b-chat-hf --promptfile myprompt
# python main.py --dataset hotpotqa --model llama-2-13b-chat-hf --task rewrite --promptfile myprompt

# python preprocess.py --dataset hotpotqa --model llama-2-7b-chat-hf --task rewrite --promptfile myprompt



# CUDA_VISIBLE_DEVICES=0 python batch_generate.py \
#     --model_name  llama-2-7b-chat-hf \
#     --model_path /home/liulian/yan/pytorch_model/llama-2-7b-chat-hf \
#     --task rewrite \

# CUDA_VISIBLE_DEVICES=1 python batch_generate.py --model_name  llama-2-13b-chat-hf --model_path /home/liulian/yan/pytorch_model/llama-2-13b-chat-hf --task rewrite

# CUDA_VISIBLE_DEVICES=1 python batch_generate.py --model_name  Qwen1.5-7B-Chat --model_path /home/liulian/yan/pytorch_model/Qwen1.5-7B-Chat --task rewrite


# CUDA_VISIBLE_DEVICES=1s python batch_generate.py \
#     --model_name  llama-2-7b-chat-hf \
#     --model_path /home/liulian/yan/pytorch_model/llama-2-7b-chat-hf \
#     --task rewrite \

# CUDA_VISIBLE_DEVICES=1s python batch_generate.py \
#     --model_name  llama-2-7b-chat-hf \
#     --model_path /home/liulian/yan/pytorch_model/llama-2-7b-chat-hf \
#     --task rewrite \


# python generate.py --dataset hotpotqa --target gpt-3.5-turbo --pos_num 2 --neg_num 0 --task qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target gpt-3.5-turbo --pos_num 2 --neg_num 2 --task qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target gpt-3.5-turbo --pos_num 0 --neg_num 2 --task qa --promptfile myprompt --nums 500 

# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/gpt-3.5-turbo_pos_2_neg_0.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/gpt-3.5-turbo_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/gpt-3.5-turbo_pos_0_neg_2.json

# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 0 --task qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 2 --task qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 0 --neg_num 2 --task qa --promptfile myprompt --nums 500 

# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/llama-2-7b-chat-hf_pos_2_neg_0.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/llama-2-7b-chat-hf_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/llama-2-7b-chat-hf_pos_0_neg_2.json

# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 2 --neg_num 0 --task qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 2 --neg_num 2 --task qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 0 --neg_num 2 --task qa --promptfile myprompt --nums 500 

# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/llama-2-13b-chat-hf_pos_2_neg_0.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/llama-2-13b-chat-hf_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/llama-2-13b-chat-hf_pos_0_neg_2.json

# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target Qwen1.5-7B-Chat --pos_num 2 --neg_num 0 --task qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=0 python generate.py --dataset hotpotqa --target Qwen1.5-7B-Chat --pos_num 2 --neg_num 2 --task qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target Qwen1.5-7B-Chat --pos_num 0 --neg_num 2 --task qa --promptfile myprompt --nums 500 

# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/Qwen1.5-7B-Chat_pos_2_neg_0.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/Qwen1.5-7B-Chat_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/Qwen1.5-7B-Chat_pos_0_neg_2.json

# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 0 --task qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 2 --task qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target Qwen1.5-14B-Chat --pos_num 0 --neg_num 2 --task qa --promptfile myprompt --nums 500 

# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/Qwen1.5-14B-Chat_pos_2_neg_0.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/Qwen1.5-14B-Chat_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/Qwen1.5-14B-Chat_pos_0_neg_2.json


# python generate.py --dataset hotpotqa --target gpt-3.5-turbo --pos_num 2 --neg_num 0 --task query_rewrite --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target gpt-3.5-turbo --pos_num 2 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target gpt-3.5-turbo --pos_num 0 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 

# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 0 --task query_rewrite --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 0 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 

# CUDA_VISIBLE_DEVICES=0 python generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 2 --neg_num 0 --task query_rewrite --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 2 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=0 python generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 0 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 

# python generate.py --dataset hotpotqa --target Qwen1.5-7B-Chat --pos_num 2 --neg_num 0 --task query_rewrite --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target Qwen1.5-7B-Chat --pos_num 2 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target Qwen1.5-7B-Chat --pos_num 0 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 

# python generate.py --dataset hotpotqa --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 0 --task query_rewrite --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target Qwen1.5-14B-Chat --pos_num 0 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 


# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target gpt-3.5-turbo --pos_num 2 --neg_num 0 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target gpt-3.5-turbo --pos_num 2 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target gpt-3.5-turbo --pos_num 0 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 

# python generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 0 --task query_rewrite --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 0 --neg_num 2 --task query_rewrite --promptfile myprompt --nums 500 


# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/gpt-3.5-turbo_gpt-3.5-turbo_pos_2_neg_0.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/gpt-3.5-turbo_gpt-3.5-turbo_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/gpt-3.5-turbo_gpt-3.5-turbo_pos_0_neg_2.json


# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --source llama-2-7b-chat-hf --target llama-2-7b-chat-hf --pos_num 2 --neg_num 0 --task rewrite_qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --source llama-2-7b-chat-hf --target llama-2-7b-chat-hf --pos_num 2 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=0 python generate.py --dataset hotpotqa --source llama-2-7b-chat-hf --target llama-2-7b-chat-hf --pos_num 0 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 

# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/llama-2-7b-chat-hf_llama-2-7b-chat-hf_pos_2_neg_0.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/llama-2-7b-chat-hf_llama-2-7b-chat-hf_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/llama-2-7b-chat-hf_llama-2-7b-chat-hf_pos_0_neg_2.json

# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --source llama-2-13b-chat-hf --target llama-2-13b-chat-hf --pos_num 2 --neg_num 0 --task rewrite_qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=0 python generate.py --dataset hotpotqa --source llama-2-13b-chat-hf --target llama-2-13b-chat-hf --pos_num 2 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python generate.py --dataset hotpotqa --source llama-2-13b-chat-hf --target llama-2-13b-chat-hf --pos_num 0 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 

# python generate.py --dataset hotpotqa --source Qwen1.5-7B-Chat --target Qwen1.5-7B-Chat --pos_num 2 --neg_num 0 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source Qwen1.5-7B-Chat --target Qwen1.5-7B-Chat --pos_num 2 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source Qwen1.5-7B-Chat --target Qwen1.5-7B-Chat --pos_num 0 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 

# python generate.py --dataset hotpotqa --source Qwen1.5-14B-Chat --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 0 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source Qwen1.5-14B-Chat --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source Qwen1.5-14B-Chat --target Qwen1.5-14B-Chat --pos_num 0 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 

# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/llama-2-13b-chat-hf_llama-2-13b-chat-hf_pos_2_neg_0.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/llama-2-13b-chat-hf_llama-2-13b-chat-hf_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/llama-2-13b-chat-hf_llama-2-13b-chat-hf_pos_0_neg_2.json

# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/Qwen1.5-7B-Chat_Qwen1.5-7B-Chat_pos_2_neg_0.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/Qwen1.5-7B-Chat_Qwen1.5-7B-Chat_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/Qwen1.5-7B-Chat_Qwen1.5-7B-Chat_pos_0_neg_2.json

# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/Qwen1.5-14B-Chat_Qwen1.5-14B-Chat_pos_2_neg_0.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/Qwen1.5-14B-Chat_Qwen1.5-14B-Chat_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/rewrite_qa/Qwen1.5-14B-Chat_Qwen1.5-14B-Chat_pos_0_neg_2.json

python rewrite_doc.py --dataset hotpotqa --target gpt-3.5-turbo --task rewrite_doc --promptfile myprompt --nums 500 

# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target gpt-3.5-turbo --pos_num 2 --neg_num 0 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target gpt-3.5-turbo --pos_num 2 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target gpt-3.5-turbo --pos_num 0 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 

# python generate.py --dataset hotpotqa --source llama-2-7b-chat-hf --target llama-2-7b-chat-hf --pos_num 2 --neg_num 0 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source llama-2-7b-chat-hf --target llama-2-7b-chat-hf --pos_num 2 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source llama-2-7b-chat-hf --target llama-2-7b-chat-hf --pos_num 0 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 

# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target llama-2-13b-chat-hf --pos_num 2 --neg_num 0 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target llama-2-13b-chat-hf --pos_num 2 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target llama-2-13b-chat-hf --pos_num 0 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 

# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target Qwen1.5-7B-Chat --pos_num 2 --neg_num 0 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target Qwen1.5-7B-Chat --pos_num 2 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target Qwen1.5-7B-Chat --pos_num 0 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 

# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 0 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target Qwen1.5-14B-Chat --pos_num 2 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
# python generate.py --dataset hotpotqa --source gpt-3.5-turbo --target Qwen1.5-14B-Chat --pos_num 0 --neg_num 2 --task rewrite_qa --promptfile myprompt --nums 500 
