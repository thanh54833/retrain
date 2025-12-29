# Hướng Dẫn Retrain Qwen 2.5 7B với Dataset Rephrase

## Tổng Quan

Tài liệu này hướng dẫn chi tiết các bước để fine-tune **Qwen 2.5 7B Instruct** với dataset Rephrase sử dụng **LoRA (Low-Rank Adaptation)**. Dataset này yêu cầu model học cách tạo JSON output từ input query tiếng Việt.

### Thông Tin Dự Án

- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **PEFT Method**: LoRA (r=8 hoặc r=16)
- **Dataset**: Rephrase (1,000 samples)
- **Task**: Text-to-JSON Generation
- **Language**: Tiếng Việt

### Yêu Cầu

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (khuyến nghị 16GB+ VRAM)
- Transformers, PEFT, Datasets libraries

## Bước 1: Chuẩn Bị Môi Trường

### 1.1 Cài Đặt Dependencies

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt các thư viện cần thiết
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.35.0
pip install peft>=0.6.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0
pip install bitsandbytes>=0.41.0
pip install trl>=0.7.0
pip install evaluate>=0.4.0
pip install scikit-learn
```

### 1.2 Kiểm Tra GPU

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB" if torch.cuda.is_available() else "")
```

## Bước 2: Load và Xử Lý Dataset

### 2.1 Load Dataset

```python
import json
from datasets import Dataset
from typing import Dict, List

def load_rephrase_dataset(file_path: str) -> List[Dict]:
    """Load dataset từ file JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Load dataset
dataset_path = "src/lora/dataset/01_simple/01_dataset_rephrase.json"
raw_data = load_rephrase_dataset(dataset_path)
print(f"Total samples: {len(raw_data)}")
```

### 2.2 Format Dataset cho Training

```python
def format_prompt(query: str) -> str:
    """Format input prompt cho model"""
    return f"Query: {query}\n\nOutput JSON:"

def format_output(data: Dict) -> str:
    """Format output JSON từ data"""
    output = {
        "keyword": data.get("keyword", ""),
        "is_in_scope": data.get("is_in_scope", False),
        "reasoning": data.get("reasoning", ""),
        "message_banner": data.get("message_banner", ""),
        "message_no_result": data.get("message_no_result", "")
    }
    return json.dumps(output, ensure_ascii=False, indent=None)

def prepare_dataset(raw_data: List[Dict]) -> List[Dict]:
    """Chuẩn bị dataset cho training"""
    formatted_data = []
    
    for item in raw_data:
        query = item.get("query", "")
        prompt = format_prompt(query)
        output = format_output(item)
        
        formatted_data.append({
            "prompt": prompt,
            "output": output,
            "query": query,
            "is_in_scope": item.get("is_in_scope", False)
        })
    
    return formatted_data

# Format dataset
formatted_data = prepare_dataset(raw_data)
print(f"Formatted samples: {len(formatted_data)}")
print("\nExample:")
print(f"Prompt: {formatted_data[0]['prompt']}")
print(f"Output: {formatted_data[0]['output'][:100]}...")
```

### 2.3 Chia Dataset (Train/Validation/Test)

```python
from sklearn.model_selection import train_test_split
import numpy as np

# Chia dataset: 80% train, 10% validation, 10% test
train_data, temp_data = train_test_split(
    formatted_data,
    test_size=0.2,
    random_state=42,
    stratify=[item['is_in_scope'] for item in formatted_data]  # Stratified split
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    stratify=[item['is_in_scope'] for item in temp_data]
)

print(f"Train: {len(train_data)} samples")
print(f"Validation: {len(val_data)} samples")
print(f"Test: {len(test_data)} samples")

# Convert to HuggingFace Dataset
from datasets import Dataset

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
test_dataset = Dataset.from_list(test_data)
```

## Bước 3: Load Model và Tokenizer

### 3.1 Load Base Model và Tokenizer

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

model_name = "Qwen/Qwen2.5-7B-Instruct"

# Cấu hình quantization (tùy chọn, để tiết kiệm bộ nhớ)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right"
)

# Đảm bảo có pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,  # Bỏ nếu không dùng quantization
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

print(f"Model loaded: {model_name}")
print(f"Model device: {next(model.parameters()).device}")
```

### 3.2 Kiểm Tra Model

```python
# Kiểm tra số tham số
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
```

## Bước 4: Cấu Hình LoRA

### 4.1 Tạo LoRA Config

```python
from peft import LoraConfig, get_peft_model, TaskType

# Cấu hình LoRA - Option 1: Conservative (r=8)
lora_config = LoraConfig(
    r=8,                          # Rank
    lora_alpha=16,               # Alpha = 2 * r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
    lora_dropout=0.1,            # Dropout để tránh overfitting
    bias="none",                  # Không train bias
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False
)

# Hoặc Option 2: Balanced (r=16) - Khuyến nghị cho hiệu suất tốt hơn
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM,
#     inference_mode=False
# )

print("LoRA Config:")
print(lora_config)
```

### 4.2 Áp Dụng LoRA vào Model

```python
# Áp dụng LoRA
model = get_peft_model(model, lora_config)

# Kiểm tra tham số có thể train
model.print_trainable_parameters()

# Output mong đợi:
# trainable params: ~4M || all params: ~7.6B || trainable%: ~0.05%
```

## Bước 5: Chuẩn Bị Dữ Liệu Training

### 5.1 Tokenization Function

```python
def tokenize_function(examples):
    """Tokenize prompt và output"""
    # Combine prompt và output
    texts = []
    for prompt, output in zip(examples["prompt"], examples["output"]):
        text = f"{prompt} {output}"
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=512,  # Điều chỉnh dựa trên độ dài output
        padding=False,
        return_tensors=None
    )
    
    # Labels cho training (same as input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Tokenize datasets
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

val_tokenized = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

print(f"Train tokenized: {len(train_tokenized)} samples")
print(f"Val tokenized: {len(val_tokenized)} samples")
```

### 5.2 Data Collator

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, không phải masked LM
    pad_to_multiple_of=8  # Tối ưu cho GPU
)
```

## Bước 6: Cấu Hình Training

### 6.1 Training Arguments

```python
from transformers import TrainingArguments

output_dir = "./results/qwen2.5-7b-rephrase-lora"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=4,  # Điều chỉnh dựa trên GPU memory
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,  # Mixed precision training
    gradient_checkpointing=True,  # Tiết kiệm memory
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    logging_dir=f"{output_dir}/logs",
    remove_unused_columns=False,
)
```

### 6.2 Custom Metrics (Tùy chọn)

```python
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    """Compute custom metrics"""
    predictions, labels = eval_pred
    
    # Tính perplexity hoặc loss
    # (Có thể thêm metrics khác như JSON validity, field completeness)
    
    return {
        "perplexity": np.exp(eval_pred.metrics.get("eval_loss", 0))
    }
```

## Bước 7: Training

### 7.1 Khởi Tạo Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,  # Uncomment nếu có custom metrics
)

print("Trainer initialized")
print(f"Training samples: {len(train_tokenized)}")
print(f"Validation samples: {len(val_tokenized)}")
```

### 7.2 Bắt Đầu Training

```python
# Training
print("Starting training...")
train_result = trainer.train()

# Lưu model cuối cùng
trainer.save_model()
tokenizer.save_pretrained(output_dir)

print(f"Training completed! Model saved to {output_dir}")
print(f"Training loss: {train_result.training_loss:.4f}")
```

### 7.3 Monitor Training

```bash
# Trong terminal khác, chạy tensorboard để theo dõi
tensorboard --logdir ./results/qwen2.5-7b-rephrase-lora/logs
```

## Bước 8: Evaluation

### 8.1 Đánh Giá trên Test Set

```python
# Evaluate trên test set
test_tokenized = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=test_dataset.column_names
)

eval_results = trainer.evaluate(eval_dataset=test_tokenized)
print("Test Results:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")
```

### 8.2 Đánh Giá JSON Validity

```python
import json
from tqdm import tqdm

def evaluate_json_validity(model, tokenizer, test_data, max_samples=100):
    """Đánh giá tỷ lệ JSON hợp lệ"""
    model.eval()
    valid_count = 0
    total_count = 0
    
    for i, item in enumerate(tqdm(test_data[:max_samples])):
        prompt = item["prompt"]
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON (sau "Output JSON:")
        if "Output JSON:" in generated_text:
            json_text = generated_text.split("Output JSON:")[-1].strip()
        else:
            json_text = generated_text
        
        # Kiểm tra JSON validity
        try:
            json.loads(json_text)
            valid_count += 1
        except:
            pass
        
        total_count += 1
    
    validity_rate = valid_count / total_count if total_count > 0 else 0
    print(f"JSON Validity Rate: {validity_rate:.2%} ({valid_count}/{total_count})")
    return validity_rate

# Chạy evaluation
validity_rate = evaluate_json_validity(model, tokenizer, test_data, max_samples=50)
```

## Bước 9: Lưu và Tải Model

### 9.1 Lưu LoRA Adapter

```python
# Lưu adapter (chỉ LoRA weights, nhỏ)
adapter_path = "./results/qwen2.5-7b-rephrase-lora/adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

print(f"Adapter saved to {adapter_path}")
```

### 9.2 Tải LoRA Adapter

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Load adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
print("Model loaded successfully!")
```

### 9.3 Merge và Unload (Tùy chọn)

```python
# Merge adapter vào base model (để inference nhanh hơn)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./results/qwen2.5-7b-rephrase-lora/merged")
tokenizer.save_pretrained("./results/qwen2.5-7b-rephrase-lora/merged")
```

## Bước 10: Inference

### 10.1 Inference Function

```python
def generate_json_response(model, tokenizer, query: str, max_length=512):
    """Generate JSON response từ query"""
    prompt = f"Query: {query}\n\nOutput JSON:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON
    if "Output JSON:" in generated_text:
        json_text = generated_text.split("Output JSON:")[-1].strip()
    else:
        json_text = generated_text
    
    # Parse JSON
    try:
        result = json.loads(json_text)
        return result
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Generated text: {json_text}")
        return None

# Test inference
test_query = "sữa cho bé 6 tháng"
result = generate_json_response(model, tokenizer, test_query)
print(f"Query: {test_query}")
print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
```

### 10.2 Batch Inference

```python
def batch_inference(model, tokenizer, queries: List[str]):
    """Inference cho nhiều queries"""
    results = []
    for query in queries:
        result = generate_json_response(model, tokenizer, query)
        results.append({
            "query": query,
            "result": result
        })
    return results

# Test batch
test_queries = [
    "sữa cho bé 6 tháng",
    "tã bỉm size M",
    "đồ chơi cho trẻ sơ sinh"
]

batch_results = batch_inference(model, tokenizer, test_queries)
for item in batch_results:
    print(f"\nQuery: {item['query']}")
    print(f"Result: {json.dumps(item['result'], ensure_ascii=False, indent=2)}")
```

## Bước 11: Tối Ưu Hóa và Fine-tuning

### 11.1 Xử Lý Class Imbalance

```python
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class WeightedLossTrainer(Trainer):
    """Custom Trainer với weighted loss cho class imbalance"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Weight cho out-of-scope (3.9%) cao hơn in-scope (96.1%)
        self.class_weights = torch.tensor([1.0, 10.0]).to(self.model.device)  # [in-scope, out-of-scope]
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Shift để align với labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Sử dụng WeightedLossTrainer thay vì Trainer
# trainer = WeightedLossTrainer(...)
```

### 11.2 Early Stopping

```python
from transformers import EarlyStoppingCallback

# Thêm vào TrainingArguments
training_args = TrainingArguments(
    # ... các tham số khác
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Thêm callback
trainer = Trainer(
    # ... các tham số khác
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

### 11.3 Hyperparameter Tuning

```python
# Thử nghiệm với các learning rate khác nhau
learning_rates = [1e-4, 2e-4, 5e-4]

for lr in learning_rates:
    training_args.learning_rate = lr
    training_args.output_dir = f"./results/qwen2.5-7b-rephrase-lora-lr{lr}"
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"LR {lr}: Eval Loss = {eval_results['eval_loss']:.4f}")
```

## Bước 12: Production Deployment

### 12.1 Tối Ưu Model cho Inference

```python
# Quantization cho inference nhanh hơn
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model với quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# Load adapter
model = PeftModel.from_pretrained(model, adapter_path)
```

### 12.2 Tạo Inference API (Flask Example)

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    query = data.get('query', '')
    
    result = generate_json_response(model, tokenizer, query)
    
    return jsonify({
        'status': 'success',
        'result': result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Troubleshooting

### Vấn Đề Thường Gặp

1. **Out of Memory (OOM)**
   - Giảm `per_device_train_batch_size`
   - Tăng `gradient_accumulation_steps`
   - Sử dụng `gradient_checkpointing=True`
   - Sử dụng quantization (4-bit hoặc 8-bit)

2. **JSON Invalid**
   - Tăng `max_new_tokens` trong generation
   - Thêm JSON examples vào prompt
   - Sử dụng constrained decoding hoặc post-processing

3. **Overfitting**
   - Giảm learning rate
   - Tăng `lora_dropout`
   - Giảm số epochs
   - Thêm regularization

4. **Underfitting**
   - Tăng `r` (rank) trong LoRA config
   - Tăng số epochs
   - Tăng learning rate
   - Thêm nhiều target modules

## Metrics Đánh Giá

### Các Metrics Quan Trọng

1. **JSON Validity Rate**: >95%
2. **Field Completeness**: >98%
3. **is_in_scope Accuracy**: >90% (cả hai classes)
4. **Training Loss**: Giảm ổn định
5. **Validation Loss**: Không tăng (tránh overfitting)

## Kết Luận

Tài liệu này cung cấp hướng dẫn đầy đủ để fine-tune Qwen 2.5 7B với dataset Rephrase sử dụng LoRA. Các bước chính:

1. ✅ Chuẩn bị môi trường và dependencies
2. ✅ Load và xử lý dataset
3. ✅ Load model và tokenizer
4. ✅ Cấu hình LoRA
5. ✅ Setup training
6. ✅ Training và monitoring
7. ✅ Evaluation
8. ✅ Saving và loading model
9. ✅ Inference
10. ✅ Tối ưu hóa và deployment

**Lưu ý**: Điều chỉnh các hyperparameters dựa trên kết quả thực tế và tài nguyên có sẵn.

## Tài Liệu Tham Khảo

- [Dataset Analysis](../02_analysis/01_dataset/01_rephrase.md)
- [PEFT Method Selection](../02_analysis/01_dataset/03_peft_method_selection_guide_with_dataset.md)
- [LoRA Guide](../01_research/04_peft_method_lora_method.md)
- [Qwen 2.5 Documentation](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [PEFT Documentation](https://huggingface.co/docs/peft)

