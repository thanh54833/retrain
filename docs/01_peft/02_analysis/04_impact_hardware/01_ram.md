# RAM và LLM

## 1. LLM tương tác với RAM như nào?

### Hiểu đơn giản

LLM giống như một bộ não khổng lồ cần RAM để:
- **Lưu trữ** kiến thức (model weights)
- **Xử lý** câu hỏi (inference)
- **Học hỏi** từ dữ liệu mới (training)

### 1.1. Khi tải model vào RAM (Model Loading)

**Giống như:** Mở một cuốn sách khổng lồ vào bộ nhớ

- **Model weights (Trọng số model)**: 
  - Đây là "kiến thức" của model - hàng tỷ con số được lưu trong RAM
  - Ví dụ: Model 7B có 7 tỷ tham số, mỗi tham số 2 bytes → cần ~14 GB RAM
  - Giống như lưu 7 tỷ số trong bộ nhớ để model "nhớ" cách trả lời

- **Tokenizer (Bộ từ điển)**:
  - Chuyển đổi từ tiếng Việt/Anh thành số để model hiểu
  - Ví dụ: "Xin chào" → [1234, 5678]
  - Chiếm vài trăm MB RAM

**Tóm lại:** Khi khởi động model, RAM phải chứa toàn bộ "kiến thức" của nó.

### 1.2. Khi model trả lời câu hỏi (Inference)

**Giống như:** Đang suy nghĩ để trả lời, cần RAM để "nhớ" các bước tính toán

- **Input (Câu hỏi của bạn)**:
  - Câu hỏi được chuyển thành số và lưu trong RAM
  - Ví dụ: "Giải thích AI là gì?" → [100, 200, 300, ...] (mỗi từ = 1 số)

- **Quá trình tính toán (Forward pass)**:
  - Model đi qua từng lớp (layer), mỗi lớp tính toán và tạo ra kết quả trung gian
  - Các kết quả trung gian này phải lưu trong RAM tạm thời
  - Ví dụ: 
    - Lớp 1: "AI" → [0.1, 0.5, 0.3, ...] (lưu vào RAM)
    - Lớp 2: Dùng kết quả lớp 1 → [0.2, 0.6, 0.4, ...] (lưu vào RAM)
    - ... tiếp tục qua 32 lớp

- **Attention (Cơ chế chú ý)**:
  - Model "chú ý" đến các từ quan trọng trong câu
  - Tạo ma trận lớn để lưu mức độ quan trọng
  - Ví dụ: Câu 100 từ → ma trận 100×100 → chiếm RAM đáng kể

**Tóm lại:** Khi trả lời, RAM cần lưu câu hỏi + tất cả các bước tính toán trung gian.

### 1.3. Khi huấn luyện model (Training)

**Giống như:** Học bài, cần ghi chép nhiều thứ hơn

- **Model weights**: Vẫn cần lưu (như inference)
- **Gradients (Độ dốc)**:
  - Model học bằng cách điều chỉnh các tham số
  - Cần lưu "hướng điều chỉnh" cho mỗi tham số
  - Kích thước = kích thước model (gấp đôi RAM!)
  - Ví dụ: Model 14 GB → Gradients 14 GB → Tổng 28 GB

- **Optimizer states (Trạng thái tối ưu)**:
  - Optimizer (như Adam) cần "nhớ" tốc độ học trước đó
  - Lưu 2 giá trị cho mỗi tham số (momentum + variance)
  - Kích thước = 2× model size
  - Ví dụ: Model 14 GB → Optimizer states 28 GB

- **Activations (Kết quả tính toán)**:
  - Phải lưu tất cả kết quả trung gian để tính ngược lại (backward pass)
  - Lớn hơn nhiều so với inference vì có nhiều mẫu (batch size > 1)
  - Ví dụ: Batch 4 câu → activation memory gấp 4 lần

**Tóm lại:** Training cần RAM gấp 4-8 lần inference vì phải lưu thêm gradients, optimizer states và nhiều activations hơn.

### 1.4. RAM ở đâu? (CPU RAM vs GPU VRAM)

**Giống như:** Có 2 loại bộ nhớ

- **CPU RAM**: Bộ nhớ chính của máy tính
  - Chậm hơn nhưng nhiều hơn (16GB, 32GB, 64GB...)
  - Model có thể chạy ở đây nếu không có GPU

- **GPU VRAM**: Bộ nhớ của card đồ họa
  - Nhanh hơn nhiều (phù hợp cho AI)
  - Ít hơn (8GB, 16GB, 24GB, 40GB...)
  - Model thường chạy ở đây để nhanh

**Luồng hoạt động:**
```
1. Model weights: CPU RAM → GPU VRAM (khi khởi động)
2. Input: CPU RAM → GPU VRAM (khi xử lý)
3. Tính toán: Xảy ra trong GPU VRAM
4. Output: GPU VRAM → CPU RAM (khi trả về kết quả)
```

**Ví dụ thực tế:**
- Model 7B inference: Cần ~15 GB VRAM (GPU) hoặc RAM (CPU)
- Model 7B training: Cần ~60 GB VRAM (GPU) hoặc RAM (CPU)
- Nếu không đủ VRAM → dùng CPU RAM (chậm hơn nhiều)

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

- **Model size** là yếu tố quan trọng nhất (tỷ lệ thuận với số parameters)
- **Sequence length** có tác động lớn, đặc biệt với attention (Seq² - quadratic scaling)
- **Batch size** tác động tuyến tính đến activation memory
- **Training** cần memory gấp 4-8x so với inference (do gradients + optimizer states + activations)
- **PEFT (LoRA/Adapter)** giúp giảm memory đáng kể cho training (chỉ train adapter weights)
- **Quantization** giúp giảm model size 2-4x nhưng có trade-off về accuracy (thường 1-5% giảm)
- **Gradient checkpointing** giúp giảm activation memory 50-70% nhưng tăng compute time 20-30%

---

**Ghi chú xác minh (2024):**
- Công thức tính model size và memory requirements đã được kiểm chứng với các model thực tế (Llama 2, Qwen)
- Số liệu về training memory (4-8x inference) phù hợp với thực tế khi dùng Adam optimizer
- Thông tin về PEFT memory savings đã được xác nhận qua các nghiên cứu về LoRA và Adapter methods
