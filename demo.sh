python train_fn.py \
  --infer "Qual é a descrição do produto 'iPhone 12 128GB'?" \
  --use_trained ./models/flan_t5_a10_lora && \
python baseline_inference.py --limit 3 && python trained_inference.py --limit 3