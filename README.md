# Processo de Fine-Tuning (FLAN-T5 + LoRA)

Este documento descreve o fluxo utilizado para ajustar o modelo `google/flan-t5-base` ao conjunto `trn.clean.jsonl`, incluindo parâmetros de treinamento e ajustes aplicados.

## Visão Geral
- **Modelo base:** `google/flan-t5-base` (seq2seq).
- **Frameworks:** `transformers`, `datasets`, `peft`, `bitsandbytes`, `evaluate`.
- **Objetivo:** gerar descrições de produto em português a partir de títulos (`title` → `content`).
- **Script principal:** `train_fn.py`.
- **Artefatos salvos:** adaptador LoRA em `models/flan_t5_a10_lora/` e, opcionalmente, pesos mesclados em `models/flan_t5_a10_lora/merged/`.

## Pré-processamento e Dataset
- Fonte: arquivo `trn.clean.jsonl` (JSON Lines), campos obrigatórios `title` e `content`.
- Leitura e filtragem implementadas em `train_fn._read_json_any` e `train_fn.make_hf_dataset`: arquivos JSON/JSONL são carregados, linhas inválidas são descartadas e cada entrada precisa conter `title` + `content` não vazios.
- Normalização: função `_norm` (em `train_fn.py`) substitui múltiplos espaços por um único espaço, remove quebras de linha redundantes e aparas antes/depois do texto.
- Subamostragem opcional (`--train_subset`) e semente fixa (`--seed`) garantem reprodutibilidade; o embaralhamento usa `random.Random(seed)`.
- Divisão: 80% treino / 20% validação realizada em `make_hf_dataset` após o embaralhamento.
- Prompt usado em todas as gerações (PT-BR):
  ```text
  Pergunta: Qual é a descrição do produto "{title}"?
  Contexto: título = {title}
  Responda apenas com a descrição.
  ```
- Alvos: campo `content` original (sem mudança estrutural).

## Geração de Baseline
- Antes do treinamento, o script pode gerar respostas do modelo base para amostras da validação (`--run_baseline true`).
- Saída: `baseline_samples.jsonl` dentro de `--output_dir`, contendo prompt, alvo e resposta do modelo base.
- Scripts auxiliares para comparação manual:
  - `baseline_inference.py`: gera respostas com o modelo base.
  - `trained_inference.py`: gera respostas com o adaptador LoRA ou com o modelo mesclado (`--use_merged`).

## Configuração de Treinamento
O comando padrão (ver `run.sh`) foi:
```bash
python train_fn.py \
  --data_path trn.clean.jsonl \
  --model_name google/flan-t5-base \
  --output_dir ./models/flan_t5_a10_lora \
  --peft true \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules "q, k, v, o, wi_0, wi_1, wo" \
  --load_in_8bit true \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --max_source_length 256 \
  --max_target_length 256 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --train_ratio 0.8 \
  --train_subset 1.0 \
  --run_baseline true \
  --do_eval true \
  --merge_lora true
```
Principais parâmetros:
- **Quantização:** carregamento em 8 bits (`bitsandbytes`), reduzindo memória.
- **Batch lógico:** `per_device_train_batch_size=8` com `gradient_accumulation_steps=2` (equivalente a 16 exemplos antes do otimizador).
- **Comprimento máximo:** 256 tokens para prompt e alvo.
- **Taxa de aprendizado:** `2e-4` (AdamW padrão do `Seq2SeqTrainer`).
- **Épocas:** 1 (ajuste inicial); pode ser aumentado conforme necessidade.
- **Avaliação:** realizada a cada 1000 etapas (`eval_strategy=steps`, `eval_steps=1000`), métrica SacreBLEU (`evaluate.load("sacrebleu")`).
- **Dispositivo:** escolhe automaticamente CUDA se disponível (`torch.cuda.is_available()`).
- **Precisão:** BF16 habilitado quando CUDA presente; FP16 desativado.
- **Checkpointing:** `checkpoint-3000`, `checkpoint-3706` na pasta de saída.

## Ajustes Específicos (LoRA / PEFT)
- **LoRA ativa:** `--peft true`.
- **Configuração:** `r=16`, `alpha=32`, `dropout=0.05`, campos alvo (`q,k,v,o,wi_0,wi_1,wo`).
- **Bias:** `none` (apenas pesos LoRA treináveis).
- **Treináveis vs. total:** registrado em tempo de execução via `print_trainable_params`.
- **Merge opcional:** `--merge_lora true` gera pesos completos mesclados em `models/flan_t5_a10_lora/merged/` para exportação sem PEFT.

## Execução e Reprodutibilidade
1. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Opcional: rode `baseline_inference.py` para guardar respostas do modelo base.
3. Execute `run.sh` ou adapte os parâmetros conforme sua infraestrutura.
4. Após o treinamento, utilize `trained_inference.py` para comparar respostas com a versão fine-tunada:
   ```bash
   python trained_inference.py --limit 3 --adapter_dir ./models/flan_t5_a10_lora
   ```
5. Utilize a pasta `merged/` para carregar o modelo completo sem PEFT:
   ```python
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
   tok = AutoTokenizer.from_pretrained("./models/flan_t5_a10_lora/merged")
   model = AutoModelForSeq2SeqLM.from_pretrained("./models/flan_t5_a10_lora/merged")
   ```

## Saídas Gerais
- `models/flan_t5_a10_lora/adapter_model.safetensors` + `adapter_config.json`: pesos LoRA.
- `models/flan_t5_a10_lora/tokenizer.*`: tokenizer reutilizado.
- `models/flan_t5_a10_lora/baseline_samples.jsonl`: respostas do modelo base pré-treino.
- `models/flan_t5_a10_lora/checkpoint-*`: checkpoints intermediários.
- `models/flan_t5_a10_lora/merged/`: pesos completos (se `--merge_lora true`).

Para ajustes futuros, recomenda-se experimentar épocas adicionais, variações de `train_subset`, e avaliar sacreBLEU na validação para monitorar ganhos reais sobre o baseline.
