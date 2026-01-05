# RAM và LLM

## 1. LLM tương tác với RAM như nào?

LLM (Large Language Model) tương tác với RAM theo các cách sau:

### 1.1. Model Loading (Tải model vào bộ nhớ)
- **Model weights**: Toàn bộ tham số của model được load vào RAM/VRAM
- **Architecture**: Cấu trúc model (layers, attention mechanisms) cần được lưu trong bộ nhớ
- **Tokenizer**: Vocabulary và tokenizer config cũng chiếm bộ nhớ

### 1.2. Inference (Suy luận)
- **Input tokens**: Sequence input được lưu trong RAM
- **Hidden states**: Các hidden states qua từng layer được tính toán và lưu tạm
- **Attention matrices**: Ma trận attention (Q, K, V) chiếm bộ nhớ đáng kể
- **Intermediate activations**: Các giá trị trung gian trong quá trình forward pass

### 1.3. Training (Huấn luyện)
- **Gradients**: Gradient của từng tham số cần lưu (gấp đôi model size)
- **Optimizer states**: Adam/AdamW lưu momentum và variance (gấp đôi model size)
- **Activations**: Lưu activations để tính backward pass (rất lớn với batch size và sequence length)

### 1.4. Memory Hierarchy
```
CPU RAM → GPU VRAM (nếu có GPU)
- Model weights có thể ở CPU RAM hoặc GPU VRAM
- Inference thường cần model + activations trong VRAM
- Training cần model + gradients + optimizer states + activations
```

## 2. Công thức tính RAM cần thiết cho từng model

### 2.1. Model Size (Kích thước model)

**Công thức cơ bản:**
```
Model Size (GB) = (Số tham số × Bytes per parameter) / (1024³)

Với:
- FP32: 4 bytes/parameter
- FP16/BF16: 2 bytes/parameter
- INT8: 1 byte/parameter
- INT4: 0.5 bytes/parameter
```

**Ví dụ:**
- **Llama 2 7B (FP16)**: 7B × 2 bytes = 14 GB
- **Llama 2 13B (FP16)**: 13B × 2 bytes = 26 GB
- **Qwen 2.5 7B (FP16)**: 7B × 2 bytes = 14 GB

### 2.2. Inference Memory Requirements

**Công thức inference:**
```
Total RAM/VRAM = Model Size + Activation Memory

Activation Memory ≈ Batch Size × Sequence Length × Hidden Size × Num Layers × 2 bytes
```

**Chi tiết hơn:**
```
Inference Memory = 
  Model Weights (FP16/FP32)
  + KV Cache (Batch × Seq × Hidden × Layers × 2 bytes)
  + Input Embeddings (Batch × Seq × Hidden × 2 bytes)
  + Attention Matrices (Batch × Heads × Seq² × 2 bytes)
  + Intermediate Activations (Batch × Seq × Hidden × Layers × 2 bytes)
```

**Ví dụ tính toán:**
- Model: Llama 2 7B (FP16) = 14 GB
- Batch size: 1
- Sequence length: 2048 tokens
- Hidden size: 4096
- Num layers: 32
- KV Cache: 1 × 2048 × 4096 × 32 × 2 bytes ≈ 0.5 GB
- Total: ~14.5 GB (tối thiểu)

### 2.3. Training Memory Requirements

**Công thức training:**
```
Training Memory = 
  Model Weights (FP32/FP16)
  + Gradients (same size as model)
  + Optimizer States (Adam: 2× model size, SGD: 1× model size)
  + Activations (Batch × Seq × Hidden × Layers × 2 bytes)
  + Temporary buffers
```

**Với Adam optimizer:**
```
Training Memory ≈ Model Size × 4 + Activation Memory

Activation Memory ≈ Batch × Seq × Hidden × Layers × 2 bytes × 2 (forward + backward)
```

**Ví dụ:**
- Model: Llama 2 7B (FP16)
- Batch size: 4
- Sequence length: 2048
- Training Memory ≈ 14 GB × 4 + (4 × 2048 × 4096 × 32 × 2 × 2 bytes)
- ≈ 56 GB + 4.3 GB ≈ **60+ GB**

### 2.4. PEFT Memory Requirements

**LoRA (Low-Rank Adaptation):**
```
LoRA Memory = Base Model + LoRA Weights
LoRA Weights ≈ Rank × (Input Dim + Output Dim) × 2 bytes × Num Layers

Ví dụ LoRA rank=8:
- LoRA weights: 8 × (4096 + 4096) × 2 bytes × 32 ≈ 4 MB
- Total: Base Model + 4 MB (rất nhỏ!)
```

**Adapter:**
```
Adapter Memory = Base Model + Adapter Weights
Adapter Weights ≈ Hidden Size × Adapter Dim × 2 bytes × Num Layers
```

## 3. Các chỉ số LLM và độ dài input tác động vào RAM như nào?

### 3.1. Các chỉ số LLM quan trọng

#### 3.1.1. Number of Parameters (Số tham số)
- **Tác động trực tiếp**: Tỷ lệ thuận với RAM
- **Công thức**: RAM ≈ Parameters × Bytes per parameter
- **Ví dụ**: 7B model cần ~14 GB (FP16), 13B cần ~26 GB

#### 3.1.2. Hidden Size (Kích thước ẩn)
- **Tác động**: Ảnh hưởng đến activation memory
- **Công thức**: Activation ∝ Hidden Size × Sequence Length × Batch Size
- **Ví dụ**: Hidden size 4096 vs 8192 → activation memory gấp đôi

#### 3.1.3. Number of Layers (Số lớp)
- **Tác động**: Tỷ lệ thuận với activation memory
- **Công thức**: Activation ∝ Num Layers × Hidden Size × Seq Length
- **Ví dụ**: 32 layers vs 64 layers → activation memory gấp đôi

#### 3.1.4. Number of Attention Heads
- **Tác động**: Ảnh hưởng đến attention matrix size
- **Công thức**: Attention Memory ∝ Heads × Seq²
- **Ví dụ**: 32 heads vs 64 heads → attention memory tăng

#### 3.1.5. Vocabulary Size
- **Tác động**: Ảnh hưởng đến embedding layer
- **Công thức**: Embedding Memory = Vocab Size × Hidden Size × 2 bytes
- **Ví dụ**: Vocab 50K vs 100K → embedding memory gấp đôi

### 3.2. Độ dài input (Sequence Length) tác động

#### 3.2.1. Linear Scaling với Sequence Length

**KV Cache Memory:**
```
KV Cache = Batch × Seq × Hidden × Layers × 2 bytes
→ Tỷ lệ thuận với Seq
```

**Attention Matrix:**
```
Attention Memory = Batch × Heads × Seq² × 2 bytes
→ Tỷ lệ thuận với Seq² (quadratic!)
```

**Activation Memory:**
```
Activation Memory = Batch × Seq × Hidden × Layers × 2 bytes
→ Tỷ lệ thuận với Seq
```

#### 3.2.2. Ví dụ cụ thể

**Sequence Length 512 tokens:**
- KV Cache: 1 × 512 × 4096 × 32 × 2 bytes ≈ 0.13 GB
- Attention: 1 × 32 × 512² × 2 bytes ≈ 0.02 GB
- Total activation: ~0.15 GB

**Sequence Length 2048 tokens:**
- KV Cache: 1 × 2048 × 4096 × 32 × 2 bytes ≈ 0.5 GB
- Attention: 1 × 32 × 2048² × 2 bytes ≈ 0.27 GB
- Total activation: ~0.77 GB (**tăng ~5x**)

**Sequence Length 8192 tokens:**
- KV Cache: 1 × 8192 × 4096 × 32 × 2 bytes ≈ 2 GB
- Attention: 1 × 32 × 8192² × 2 bytes ≈ 4.3 GB
- Total activation: ~6.3 GB (**tăng ~42x từ 512!**)

### 3.3. Batch Size tác động

**Công thức:**
```
Memory ∝ Batch Size × Sequence Length × Hidden Size × Layers
```

**Ví dụ:**
- Batch size 1: ~0.15 GB (seq=512)
- Batch size 4: ~0.6 GB (seq=512) - **tăng 4x**
- Batch size 8: ~1.2 GB (seq=512) - **tăng 8x**

### 3.4. Tổng hợp tác động

**Bảng so sánh memory requirements:**

| Model | Seq Length | Batch | Model Size | Activation | Total |
|-------|------------|-------|------------|------------|-------|
| 7B FP16 | 512 | 1 | 14 GB | 0.15 GB | 14.15 GB |
| 7B FP16 | 2048 | 1 | 14 GB | 0.77 GB | 14.77 GB |
| 7B FP16 | 8192 | 1 | 14 GB | 6.3 GB | 20.3 GB |
| 7B FP16 | 2048 | 4 | 14 GB | 3.1 GB | 17.1 GB |
| 13B FP16 | 2048 | 1 | 26 GB | 1.5 GB | 27.5 GB |

### 3.5. Optimization Techniques

**Giảm memory usage:**
1. **Gradient Checkpointing**: Trade compute for memory
   - Giảm activation memory ~50-70%
   - Tăng thời gian training ~20-30%

2. **Mixed Precision Training**: FP16/BF16 thay vì FP32
   - Giảm memory ~50%
   - Model + Gradients + Optimizer states

3. **Sequence Parallelism**: Chia sequence across GPUs
   - Giảm memory per GPU
   - Cần communication overhead

4. **PEFT (LoRA/Adapter)**: Chỉ train một phần nhỏ parameters
   - Giảm gradient + optimizer memory đáng kể
   - Base model vẫn cần load

5. **Quantization**: INT8/INT4 thay vì FP16
   - Giảm model size 2-4x
   - Có thể giảm accuracy

## 4. Kết luận

- **Model size** là yếu tố quan trọng nhất (tỷ lệ thuận)
- **Sequence length** có tác động lớn, đặc biệt với attention (Seq²)
- **Batch size** tác động tuyến tính
- **Training** cần memory gấp 4-8x so với inference
- **PEFT** giúp giảm memory đáng kể cho training
- **Quantization** giúp giảm model size nhưng có trade-off về accuracy
