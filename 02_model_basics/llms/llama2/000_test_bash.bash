# =========================
# Module Check
# =========================

# # Sanity
# python sanity_check.py

# # Optimizer
# python optimizer_test.py

# # RoPE
# python rope_test.py


# =========================
# Text Generation
# =========================
# python run_llama.py --option generate --use_gpu

# =========================
# Zero-Shot Prompting
# =========================

# SST Dataset
python run_llama.py \
  --option prompt \
  --batch_size 10 \
  --train data/sst-train.txt \
  --dev data/sst-dev.txt \
  --test data/sst-test.txt \
  --label-names data/sst-label-mapping.json \
  --dev_out sst-dev-prompting-output.txt \
  --test_out sst-test-prompting-output.txt \
  --use_gpu

# CFIMDB Dataset
python run_llama.py \
  --option prompt \
  --batch_size 10 \
  --train data/cfimdb-train.txt \
  --dev data/cfimdb-dev.txt \
  --test data/cfimdb-test.txt \
  --label-names data/cfimdb-label-mapping.json \
  --dev_out cfimdb-dev-prompting-output.txt \
  --test_out cfimdb-test-prompting-output.txt \
  --use_gpu

# =========================
# Classification Fine-tuning
# =========================

# SST Dataset
python run_llama.py \
  --option finetune \
  --epochs 5 \
  --lr 2e-5 \
  --batch_size 80 \
  --train data/sst-train.txt \
  --dev data/sst-dev.txt \
  --test data/sst-test.txt \
  --label-names data/sst-label-mapping.json \
  --dev_out sst-dev-finetuning-output.txt \
  --test_out sst-test-finetuning-output.txt \
  --use_gpu

# CFIMDB Dataset
python run_llama.py \
  --option finetune \
  --epochs 5 \
  --lr 2e-5 \
  --batch_size 10 \
  --train data/cfimdb-train.txt \
  --dev data/cfimdb-dev.txt \
  --test data/cfimdb-test.txt \
  --label-names data/cfimdb-label-mapping.json \
  --dev_out cfimdb-dev-finetuning-output.txt \
  --test_out cfimdb-test-finetuning-output.txt \
  --use_gpu

# =========================
# LoRA Fine-tuning
# =========================

# SST Dataset
python run_llama.py \
  --option lora \
  --epochs 5 \
  --lr 2e-5 \
  --batch_size 80 \
  --train data/sst-train.txt \
  --dev data/sst-dev.txt \
  --test data/sst-test.txt \
  --label-names data/sst-label-mapping.json \
  --dev_out sst-dev-lora-output.txt \
  --test_out sst-test-lora-output.txt \
  --lora_rank 4 \
  --lora_alpha 1.0 \
  --use_gpu

# CFIMDB Dataset
python run_llama.py \
  --option lora \
  --epochs 5 \
  --lr 2e-5 \
  --batch_size 10 \
  --train data/cfimdb-train.txt \
  --dev data/cfimdb-dev.txt \
  --test data/cfimdb-test.txt \
  --label-names data/cfimdb-label-mapping.json \
  --dev_out cfimdb-dev-lora-output.txt \
  --test_out cfimdb-test-lora-output.txt \
  --lora_rank 4 \
  --lora_alpha 1.0 \
  --use_gpu