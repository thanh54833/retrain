# Hugging Face Transformers - Tổng Quan và Hướng Dẫn

## Giới Thiệu

**Hugging Face Transformers** là một framework toàn diện cho các mô hình machine learning state-of-the-art trong nhiều lĩnh vực: text, computer vision, audio, video, và multimodal models, cho cả inference và training. Thư viện này tập trung hóa định nghĩa mô hình để đảm bảo tính nhất quán trên toàn bộ hệ sinh thái.

Transformers đóng vai trò là điểm trung tâm (pivot) giữa các framework: nếu một định nghĩa mô hình được hỗ trợ, nó sẽ tương thích với đa số các training framework (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), inference engines (vLLM, SGLang, TGI, ...), và các thư viện modeling liên quan (llama.cpp, mlx, ...) sử dụng định nghĩa mô hình từ `transformers`.

### Cam Kết

Transformers cam kết hỗ trợ các mô hình state-of-the-art mới và dân chủ hóa việc sử dụng chúng bằng cách làm cho định nghĩa mô hình trở nên đơn giản, có thể tùy chỉnh và hiệu quả.

### Quy Mô

Có hơn **1 triệu model checkpoints** trên [Hugging Face Hub](https://huggingface.com/models) mà bạn có thể sử dụng với Transformers.

## Tính Năng Chính

Transformers cung cấp mọi thứ bạn cần cho inference hoặc training với các mô hình pretrained state-of-the-art. Một số tính năng chính bao gồm:

### 1. Pipeline - Inference Đơn Giản và Tối Ưu

**Pipeline** là một lớp inference đơn giản và được tối ưu hóa cho nhiều tác vụ machine learning:

- ✅ **Text Generation:** Tạo văn bản với LLM
- ✅ **Image Segmentation:** Phân đoạn hình ảnh
- ✅ **Automatic Speech Recognition (ASR):** Nhận dạng giọng nói tự động
- ✅ **Document Question Answering:** Trả lời câu hỏi từ tài liệu
- ✅ **Và nhiều tác vụ khác...**

**Ví dụ sử dụng Pipeline:**

```python
from transformers import pipeline

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("Hello, I am a language model", max_length=30)

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love using Transformers!")

# Image classification
vision_classifier = pipeline("image-classification")
result = vision_classifier("path/to/image.jpg")

# Automatic speech recognition
asr = pipeline("automatic-speech-recognition")
result = asr("path/to/audio.wav")
```

### 2. Trainer - Huấn Luyện Toàn Diện

**Trainer** là một trainer toàn diện hỗ trợ nhiều tính năng:

- ✅ **Mixed Precision:** Huấn luyện với độ chính xác hỗn hợp để tăng tốc
- ✅ **torch.compile:** Tích hợp PyTorch 2.0 compilation
- ✅ **FlashAttention:** Tối ưu hóa attention mechanism
- ✅ **Distributed Training:** Huấn luyện phân tán cho PyTorch models
- ✅ **Gradient Accumulation:** Tích lũy gradient cho batch lớn
- ✅ **Learning Rate Scheduling:** Lập lịch learning rate
- ✅ **Checkpointing:** Lưu và tải checkpoint

**Ví dụ sử dụng Trainer:**

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Load dataset
dataset = load_dataset("glue", "sst2")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Train
trainer.train()
```

### 3. Generate - Tạo Văn Bản Nhanh

**Generate** cho phép tạo văn bản nhanh với LLM và VLM:

- ✅ **Streaming:** Tạo văn bản theo dòng (streaming)
- ✅ **Multiple Decoding Strategies:** Nhiều chiến lược decoding
  - Greedy decoding
  - Sampling với temperature
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Beam search
- ✅ **Custom Stopping Criteria:** Tiêu chí dừng tùy chỉnh
- ✅ **Logits Processor:** Xử lý logits tùy chỉnh

**Ví dụ sử dụng Generate:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare input
text = "The future of AI is"
inputs = tokenizer(text, return_tensors="pt")

# Generate with different strategies
# 1. Greedy decoding
outputs = model.generate(
    inputs.input_ids,
    max_length=50,
    num_return_sequences=1
)

# 2. Sampling with temperature
outputs = model.generate(
    inputs.input_ids,
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

# 3. Beam search
outputs = model.generate(
    inputs.input_ids,
    max_length=50,
    num_beams=5,
    early_stopping=True
)

# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Thiết Kế và Nguyên Tắc

Transformers được thiết kế cho developers, machine learning engineers và researchers. Các nguyên tắc thiết kế chính:

### 1. Nhanh và Dễ Sử Dụng

Mỗi mô hình được triển khai từ chỉ **ba lớp chính**:
- **Configuration:** Cấu hình mô hình
- **Model:** Mô hình thực tế
- **Preprocessor:** Tiền xử lý dữ liệu (tokenizer, feature extractor, processor)

Điều này cho phép sử dụng nhanh chóng cho inference hoặc training với `Pipeline` hoặc `Trainer`.

**Ví dụ cấu trúc đơn giản:**

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Configuration
config = AutoConfig.from_pretrained("bert-base-uncased")

# Model
model = AutoModel.from_pretrained("bert-base-uncased")

# Preprocessor (Tokenizer)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### 2. Pretrained Models

Sử dụng pretrained models giúp:
- ✅ **Giảm carbon footprint:** Không cần train từ đầu
- ✅ **Giảm chi phí tính toán:** Tiết kiệm thời gian và tài nguyên
- ✅ **Tiết kiệm thời gian:** Sử dụng ngay các mô hình đã được train
- ✅ **State-of-the-art performance:** Mỗi pretrained model được tái tạo càng gần với mô hình gốc càng tốt

### 3. Tương Thích Rộng Rãi

Transformers đảm bảo tương thích với:

**Training Frameworks:**
- Axolotl
- Unsloth
- DeepSpeed
- FSDP (Fully Sharded Data Parallel)
- PyTorch-Lightning
- Và nhiều framework khác...

**Inference Engines:**
- vLLM
- SGLang
- TGI (Text Generation Inference)
- Và nhiều engine khác...

**Adjacent Libraries:**
- llama.cpp
- mlx
- Và nhiều thư viện khác...

## Các Thành Phần Chính

### Auto Classes

Transformers cung cấp các Auto classes để tự động tải mô hình phù hợp:

```python
from transformers import (
    AutoConfig,      # Tự động tải configuration
    AutoModel,       # Tự động tải model
    AutoTokenizer,  # Tự động tải tokenizer
    AutoProcessor,  # Tự động tải processor (cho multimodal)
)

# Ví dụ với BERT
config = AutoConfig.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Ví dụ với GPT-2
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Ví dụ với Vision Transformer
from transformers import AutoImageProcessor, AutoModelForImageClassification
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
```

### Model Classes Theo Task

Transformers cung cấp các lớp model chuyên biệt cho từng task:

#### Text Models

```python
# Causal Language Modeling (Text Generation)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Masked Language Modeling
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Sequence Classification
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Question Answering
from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Token Classification (NER, POS tagging)
from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
```

#### Vision Models

```python
# Image Classification
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Image Segmentation
from transformers import AutoModelForImageSegmentation
model = AutoModelForImageSegmentation.from_pretrained("facebook/detr-resnet-50")

# Object Detection
from transformers import AutoModelForObjectDetection
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
```

#### Audio Models

```python
# Automatic Speech Recognition
from transformers import AutoModelForSpeechSeq2Seq
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base")

# Audio Classification
from transformers import AutoModelForAudioClassification
model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
```

#### Multimodal Models

```python
# Vision-Language Models
from transformers import AutoProcessor, AutoModelForVision2Seq
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip-image-captioning-base")
```

## Workflow Cơ Bản

### 1. Inference Workflow

```python
from transformers import pipeline

# 1. Tạo pipeline
classifier = pipeline("sentiment-analysis")

# 2. Sử dụng pipeline
result = classifier("I love Transformers!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

### 2. Training Workflow

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# 1. Load model và tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Load và chuẩn bị dataset
dataset = load_dataset("glue", "sst2")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. Cấu hình training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
)

# 4. Tạo Trainer và train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()
trainer.save_model("./fine-tuned-model")
```

### 3. Custom Training Workflow

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Chuẩn bị dữ liệu
texts = ["Your training data here"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# 3. Setup optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# 4. Training loop
model.train()
for epoch in range(3):
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")
```

## Tích Hợp Với PEFT

Transformers tích hợp mạnh mẽ với PEFT (Parameter-Efficient Fine-Tuning):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Now you can use Trainer with PEFT model
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora-results",
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

## Tối Ưu Hóa và Hiệu Suất

### 1. Quantization

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=quantization_config
)

# 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=quantization_config
)
```

### 2. Flash Attention

```python
from transformers import AutoModelForCausalLM, TrainingArguments

# Flash Attention được tự động sử dụng nếu có sẵn
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    attn_implementation="flash_attention_2"  # Yêu cầu flash-attn package
)
```

### 3. torch.compile

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Compile model với PyTorch 2.0
model = torch.compile(model)
```

### 4. Gradient Checkpointing

```python
from transformers import AutoModelForCausalLM, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.gradient_checkpointing_enable()  # Tiết kiệm bộ nhớ

# Hoặc trong TrainingArguments
training_args = TrainingArguments(
    gradient_checkpointing=True,
    # ...
)
```

## Best Practices

### 1. Sử Dụng Pretrained Models

Luôn bắt đầu với pretrained models thay vì train từ đầu:

```python
# ✅ Tốt: Sử dụng pretrained model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# ❌ Không tốt: Train từ đầu (trừ khi có lý do đặc biệt)
from transformers import BertConfig, BertForSequenceClassification
config = BertConfig()
model = BertForSequenceClassification(config)  # Random weights
```

### 2. Sử Dụng Auto Classes

Sử dụng Auto classes để tự động tải đúng model:

```python
# ✅ Tốt: Auto class tự động chọn đúng class
model = AutoModelForCausalLM.from_pretrained("gpt2")

# ❌ Không tốt: Phải biết chính xác class
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")
```

### 3. Xử Lý Tokenization Đúng Cách

```python
# ✅ Tốt: Sử dụng tokenizer của model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world", return_tensors="pt", padding=True, truncation=True)

# ❌ Không tốt: Sử dụng tokenizer không khớp
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Không khớp với BERT
```

### 4. Quản Lý Bộ Nhớ

```python
# Sử dụng device_map="auto" để tự động phân bổ GPU
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16  # Sử dụng half precision
)

# Hoặc sử dụng quantization
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=quantization_config
)
```

## Tài Nguyên Học Tập

### Khóa Học LLM

Hugging Face cung cấp [LLM Course](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt) toàn diện bao gồm:

- ✅ **Fundamentals:** Cách transformer models hoạt động
- ✅ **Practical Applications:** Ứng dụng thực tế trên nhiều tác vụ
- ✅ **Complete Workflow:** Từ curating datasets đến fine-tuning LLM
- ✅ **Reasoning Capabilities:** Triển khai khả năng lý luận
- ✅ **Theoretical & Hands-on:** Cả lý thuyết và bài tập thực hành

### Tài Liệu Tham Khảo

- [Transformers Documentation](https://huggingface.co/docs/transformers/en/index)
- [Hugging Face Hub](https://huggingface.co/models)
- [Models Timeline](https://huggingface.co/docs/transformers/en/models_timeline)
- [Transformers Philosophy](https://huggingface.co/docs/transformers/en/philosophy)

## Kết Luận

Hugging Face Transformers là một framework mạnh mẽ và linh hoạt cho việc làm việc với các mô hình machine learning state-of-the-art. Với hơn 1 triệu model checkpoints, tích hợp rộng rãi với các framework khác, và API đơn giản, Transformers giúp việc sử dụng và fine-tuning các mô hình trở nên dễ dàng và hiệu quả.

**Điểm mạnh chính:**
- ✅ API đơn giản và nhất quán
- ✅ Hỗ trợ nhiều modality (text, vision, audio, multimodal)
- ✅ Tích hợp với nhiều framework và engine
- ✅ Hơn 1 triệu pretrained models
- ✅ Tài liệu và tài nguyên học tập phong phú

**Use Cases:**
- ✅ Text generation và NLP tasks
- ✅ Computer vision applications
- ✅ Audio processing và speech recognition
- ✅ Multimodal applications
- ✅ Fine-tuning và transfer learning
- ✅ Research và experimentation

