# CUDA_VISIBLE_DEVICES=1 python doc_generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 0 --task doc_rewrite_qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=1 python doc_generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 8 --task doc_rewrite_qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=0 python doc_generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 2 --task doc_rewrite_qa --promptfile myprompt --nums 500
# CUDA_VISIBLE_DEVICES=1 python doc_generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 4 --task doc_rewrite_qa --promptfile myprompt --nums 500 
# CUDA_VISIBLE_DEVICES=0 python doc_generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 2 --neg_num 6 --task doc_rewrite_qa --promptfile myprompt --nums 500
# CUDA_VISIBLE_DEVICES=1 python doc_generate.py --dataset hotpotqa --target llama-2-7b-chat-hf --pos_num 0 --neg_num 8 --task doc_rewrite_qa --promptfile myprompt --nums 500

# python doc_generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 2 --neg_num 0 --task doc_rewrite_qa --promptfile myprompt --nums 500 
# python doc_generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 2 --neg_num 8 --task doc_rewrite_qa --promptfile myprompt --nums 500 
# python doc_generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 2 --neg_num 2 --task doc_rewrite_qa --promptfile myprompt --nums 500
# python doc_generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 2 --neg_num 4 --task doc_rewrite_qa --promptfile myprompt --nums 500 
# python doc_generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 2 --neg_num 6 --task doc_rewrite_qa --promptfile myprompt --nums 500
# python doc_generate.py --dataset hotpotqa --target llama-2-13b-chat-hf --pos_num 0 --neg_num 8 --task doc_rewrite_qa --promptfile myprompt --nums 500

# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/llama-2-13b-chat-hf_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/doc_rewrite_qa/llama-2-13b-chat-hf_pos_2_neg_2.json
# python evaluate_result.py --data_path ./results/hotpotqa/500/qa/llama-2-13b-chat-hf_pos_0_neg_2.json