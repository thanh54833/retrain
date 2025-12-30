# Vấn đề: Input dài gây lỗi Out of Memory (OOM)

## Tóm tắt ngắn gọn

**Vấn đề**: Khi bạn gửi một prompt rất dài (nhiều từ/câu) vào model, GPU sẽ hết bộ nhớ và báo lỗi "CUDA out of memory".

**Nguyên nhân chính**: Model cần tính toán mối quan hệ giữa TẤT CẢ các từ trong prompt với nhau. Nếu prompt có 2000 từ, model phải tính 2000 × 2000 = 4 triệu mối quan hệ! Điều này tiêu tốn rất nhiều bộ nhớ GPU.

**Giải pháp đơn giản**: Giới hạn độ dài input (ví dụ: tối đa 2000 từ) hoặc tắt một số tính năng tiết kiệm bộ nhớ.

---

## Hiểu đơn giản: Tại sao input dài lại tốn bộ nhớ?

### Ví dụ dễ hiểu

Hãy tưởng tượng bạn đang đọc một cuốn sách:

- **Sách ngắn (100 trang)**: Bạn có thể nhớ được các nhân vật và mối quan hệ giữa họ → Tốn ít "bộ nhớ não"
- **Sách dài (1000 trang)**: Bạn phải nhớ rất nhiều nhân vật, mối quan hệ phức tạp → Tốn rất nhiều "bộ nhớ não"

Model transformer cũng vậy:
- **Prompt ngắn (100 từ)**: Model chỉ cần nhớ mối quan hệ giữa 100 từ → Tốn ít GPU memory
- **Prompt dài (2000 từ)**: Model phải nhớ mối quan hệ giữa 2000 từ → Tốn rất nhiều GPU memory

### Công thức đơn giản

```
Bộ nhớ cần thiết = Bộ nhớ model + (Số từ trong prompt)² × Hằng số
```

**Quan trọng**: Bộ nhớ tăng theo **bình phương** số từ, không phải tăng tuyến tính!

**Ví dụ cụ thể**:
- 100 từ → Bộ nhớ = 6 GB
- 500 từ → Bộ nhớ = 6.5 GB (tăng 0.5 GB)
- 1000 từ → Bộ nhớ = 8 GB (tăng 2 GB)
- 2000 từ → Bộ nhớ = 13 GB (tăng 7 GB!)
- 4000 từ → Bộ nhớ = 35 GB (OOM - hết bộ nhớ!)

**Nhận xét**: Khi số từ tăng gấp đôi (1000 → 2000), bộ nhớ tăng gấp 4 lần (2 GB → 8 GB)!

---

## Chi tiết kỹ thuật: Model dùng bộ nhớ như thế nào?

Khi model xử lý một prompt, nó cần 3 loại bộ nhớ:

### 1. Bộ nhớ cho Model (Model Weights) - Cố định

**Đây là gì?**
- Đây là "kiến thức" của model đã được học từ trước
- Giống như "bộ não" của model

**Kích thước:**
- Qwen 7B không quantization: ~14 GB
- Qwen 7B với 4-bit quantization: ~5-6 GB

**Đặc điểm:**
- Kích thước **KHÔNG THAY ĐỔI** dù prompt dài hay ngắn
- Giống như kích thước bộ não của bạn không đổi dù đọc sách ngắn hay dài

### 2. Bộ nhớ cho Tính toán (Activation Memory) - Phụ thuộc độ dài prompt

**Đây là gì?**
- Bộ nhớ để lưu kết quả tính toán tạm thời khi model xử lý prompt
- Giống như "giấy nháp" khi bạn giải bài toán

**Có 2 phần chính:**

#### a) Hidden States (Trạng thái ẩn)
- Model lưu "ý nghĩa" của mỗi từ sau mỗi bước xử lý
- Kích thước: `Số từ × 4096 × 2 bytes` cho mỗi layer
- Ví dụ: 2000 từ → `2000 × 4096 × 2 = 16.38 MB` mỗi layer
- Với 28 layers → `16.38 MB × 28 = 458 MB`

#### b) Attention Matrices (Ma trận chú ý) - **QUAN TRỌNG NHẤT!**
- Đây là phần tốn bộ nhớ nhất!
- Model tính toán mức độ "chú ý" giữa mỗi cặp từ với nhau
- Kích thước: `(Số từ)² × 32 × 2 bytes` cho mỗi layer

**Ví dụ cụ thể với 2000 từ:**
```
Attention memory mỗi layer = 2000² × 32 × 2 bytes
                            = 4,000,000 × 32 × 2
                            = 256 MB
```

**Với 28 layers:**
```
Tổng attention memory = 256 MB × 28 = 7.17 GB
```

**Đây là lý do chính gây OOM!** Chỉ riêng attention matrices đã tốn 7 GB cho 2000 từ!

### 3. Bộ nhớ cho Cache (KV Cache) - Tùy chọn

**Đây là gì?**
- Khi model generate từng từ mới, nó có thể lưu lại kết quả tính toán trước đó để không phải tính lại
- Giống như "ghi chú" để không phải đọc lại từ đầu

**Kích thước:**
- Với 2000 từ: ~917 MB
- Chỉ có khi `use_cache=True`

**Lưu ý:** Có thể tắt để tiết kiệm bộ nhớ (nhưng sẽ chậm hơn một chút)

---

## Bảng so sánh: Bộ nhớ cần thiết theo độ dài prompt

**Giả sử sử dụng Qwen 7B với 4-bit quantization (model weights = 5 GB):**

| Độ dài prompt | Attention Memory | Tổng bộ nhớ | Ghi chú |
|---------------|------------------|-------------|---------|
| 100 từ        | 0.02 GB          | ~6 GB       | ✅ An toàn |
| 500 từ        | 0.45 GB          | ~6.5 GB     | ✅ An toàn |
| 1000 từ       | 1.8 GB           | ~8 GB       | ⚠️ Cần cẩn thận |
| 2000 từ       | 7.2 GB           | ~13 GB      | ❌ OOM trên GPU 12GB |
| 4000 từ       | 28.7 GB          | ~35 GB      | ❌ OOM trên hầu hết GPU |

**Nhận xét:**
- Với GPU 12-16 GB: Nên giới hạn prompt ≤ 1500 từ
- Với GPU 24 GB: Có thể xử lý prompt ≤ 2500 từ
- Với GPU 40+ GB (A100): Có thể xử lý prompt dài hơn nhiều

---

## Tại sao Attention lại tốn bộ nhớ nhiều đến vậy?

### Giải thích đơn giản

Khi bạn đọc câu: **"Con mèo ngồi trên tấm thảm"**

Model cần hiểu:
- Từ "mèo" liên quan đến "ngồi" như thế nào?
- Từ "ngồi" liên quan đến "thảm" như thế nào?
- Từ "mèo" liên quan đến "thảm" như thế nào?
- ... và tất cả các cặp từ khác

**Với 6 từ**, model cần tính: 6 × 6 = 36 mối quan hệ

**Với 2000 từ**, model cần tính: 2000 × 2000 = **4,000,000 mối quan hệ!**

Mỗi mối quan hệ cần lưu một số (attention score), và với 28 layers, số lượng này nhân lên rất nhiều!

### Hình ảnh minh họa (text)

```
Prompt ngắn (5 từ):
[1] [2] [3] [4] [5]
 ↓   ↓   ↓   ↓   ↓
Model chỉ cần tính 5×5 = 25 mối quan hệ
Bộ nhớ: Nhỏ

Prompt dài (2000 từ):
[1] [2] [3] ... [2000]
 ↓   ↓   ↓       ↓
Model cần tính 2000×2000 = 4,000,000 mối quan hệ!
Bộ nhớ: Rất lớn (7+ GB)
```

---

## Trường hợp thực tế: Lỗi OOM

### Error message thường gặp

```
OutOfMemoryError: CUDA out of memory. 
Tried to allocate 166.00 MiB. 
GPU 0 has a total capacity of 15.70 GiB 
of which 30.81 MiB is free.
```

### Phân tích lỗi

**Tình huống:**
- GPU có tổng cộng: **15.70 GB**
- Model đã dùng: **5.71 GB** (model weights + một phần activation)
- Còn lại: **30 MB** (rất ít!)
- Khi generate với prompt dài (~2000 từ), cần thêm: **7-8 GB**
- **Kết quả**: Không đủ bộ nhớ → Lỗi OOM

**Nguyên nhân phụ:**
- Có 2 process khác đang dùng GPU:
  - Process 3141301: 6.09 GB
  - Process 3144297: 9.58 GB
- Tổng: 15.67 GB đã được sử dụng
- Process hiện tại chỉ còn ~30 MB để làm việc

**Giải pháp:**
1. Đóng các process khác đang dùng GPU
2. Giảm độ dài prompt
3. Tắt KV cache
4. Giảm max_new_tokens

---

## Giải pháp chi tiết

### Giải pháp 1: Giới hạn độ dài Input (Truncation) ⭐ **KHUYẾN NGHỊ**

**Cách làm:**
```python
# Chỉ lấy 2000 từ đầu tiên của prompt
model_inputs = tokenizer(
    [text], 
    return_tensors="pt",
    truncation=True,        # Bật truncation
    max_length=2000,         # Giới hạn 2000 tokens
    padding=False
)
```

**Ưu điểm:**
- ✅ Giảm bộ nhớ đáng kể (từ 13 GB xuống ~8 GB với 2000 từ)
- ✅ Đơn giản, dễ áp dụng
- ✅ Hiệu quả cao

**Nhược điểm:**
- ⚠️ Mất thông tin nếu prompt quá dài
- ⚠️ Có thể ảnh hưởng chất lượng nếu thông tin quan trọng bị cắt

**Khi nào dùng:**
- Prompt rất dài (> 2000 từ)
- Không cần toàn bộ thông tin trong prompt
- Ưu tiên tránh OOM hơn là giữ nguyên prompt

---

### Giải pháp 2: Tắt KV Cache ⭐ **KHUYẾN NGHỊ**

**Cách làm:**
```python
model.generate(
    **model_inputs,
    use_cache=False,  # Tắt cache để tiết kiệm bộ nhớ
    max_new_tokens=256,
    ...
)
```

**Ưu điểm:**
- ✅ Tiết kiệm ~1 GB bộ nhớ (với prompt 2000 từ)
- ✅ Vẫn giữ được toàn bộ prompt (không mất thông tin)
- ✅ Dễ áp dụng

**Nhược điểm:**
- ⚠️ Chậm hơn một chút (phải tính lại attention mỗi bước)
- ⚠️ Không ảnh hưởng nhiều đến bộ nhớ attention matrices

**Khi nào dùng:**
- Cần giữ toàn bộ prompt
- Chấp nhận chậm hơn một chút
- Kết hợp với các giải pháp khác

---

### Giải pháp 3: Giảm số từ Generate

**Cách làm:**
```python
model.generate(
    **model_inputs,
    max_new_tokens=128,  # Giảm từ 512 xuống 128
    ...
)
```

**Ưu điểm:**
- ✅ Giảm bộ nhớ cho output sequence
- ✅ Generate nhanh hơn
- ✅ Output ngắn gọn hơn

**Nhược điểm:**
- ⚠️ Có thể không đủ dài cho một số task
- ⚠️ Không giải quyết được vấn đề chính (attention memory)

**Khi nào dùng:**
- Kết hợp với các giải pháp khác
- Task không cần output dài
- Ưu tiên tốc độ

---

### Giải pháp 4: Giải phóng GPU Memory

**Cách làm:**
```python
import torch
import gc

# Giải phóng bộ nhớ không dùng
torch.cuda.empty_cache()      # Xóa PyTorch cache
torch.cuda.synchronize()      # Đợi GPU hoàn thành
gc.collect()                  # Python garbage collection
```

**Ưu điểm:**
- ✅ Đơn giản
- ✅ Có thể giải phóng một phần bộ nhớ

**Nhược điểm:**
- ⚠️ Chỉ giải phóng được một phần nhỏ
- ⚠️ Không giải quyết được vấn đề cốt lõi (attention memory quá lớn)

**Khi nào dùng:**
- Trước khi generate với prompt dài
- Kết hợp với các giải pháp khác
- Sau khi xử lý nhiều requests

---

### Giải pháp 5: Sử dụng Model nhỏ hơn

**Các lựa chọn:**
- Qwen 2.5-3B (thay vì 7B) → Giảm ~50% bộ nhớ model
- Qwen 2.5-1.5B → Giảm ~75% bộ nhớ model

**Ưu điểm:**
- ✅ Bộ nhớ model thấp hơn đáng kể
- ✅ Có thể xử lý prompt dài hơn
- ✅ Vẫn đủ tốt cho nhiều task

**Nhược điểm:**
- ⚠️ Chất lượng có thể kém hơn 7B
- ⚠️ Vẫn gặp vấn đề với prompt rất dài (vì attention memory vẫn tăng theo bình phương)

**Khi nào dùng:**
- GPU rất nhỏ (< 12 GB)
- Chấp nhận giảm nhẹ chất lượng
- Cần xử lý prompt dài thường xuyên

---

### Giải pháp 6: Đóng các Process khác

**Cách làm:**
```bash
# Kiểm tra các process đang dùng GPU
nvidia-smi

# Kill process nếu cần (cẩn thận!)
kill -9 <PID>
```

**Ưu điểm:**
- ✅ Giải phóng bộ nhớ ngay lập tức
- ✅ Hiệu quả cao

**Nhược điểm:**
- ⚠️ Có thể ảnh hưởng đến các task khác
- ⚠️ Cần cẩn thận khi kill process

**Khi nào dùng:**
- Có process không cần thiết đang chạy
- Cần bộ nhớ ngay lập tức
- Đã thử các giải pháp khác nhưng vẫn thiếu bộ nhớ

---

## Best Practices (Thực hành tốt nhất)

### 1. Kiểm tra độ dài Input trước khi Generate

```python
# Tokenize và kiểm tra độ dài
model_inputs = tokenizer([text], return_tensors="pt")
input_length = model_inputs.input_ids.shape[1]

print(f"Input length: {input_length} tokens")

# Tự động điều chỉnh nếu quá dài
if input_length > 1500:
    print(f"⚠️ Input dài ({input_length} tokens), sẽ giảm max_new_tokens")
    max_new_tokens = min(max_new_tokens, 128)
    
    # Có thể truncate nếu cần
    if input_length > 2000:
        print(f"⚠️ Input quá dài, sẽ truncate xuống 2000 tokens")
        model_inputs = tokenizer(
            [text], 
            return_tensors="pt",
            truncation=True,
            max_length=2000
        )
```

### 2. Sử dụng Multiple Fallback Configs

```python
# Thử các config từ tối ưu nhất đến tiết kiệm nhất
generation_configs = [
    # Config 1: Tối ưu nhất
    {
        "max_new_tokens": 256,
        "use_cache": False,
        "temperature": 0.7,
    },
    # Config 2: Tiết kiệm hơn
    {
        "max_new_tokens": 128,
        "use_cache": False,
        "temperature": 0.7,
    },
    # Config 3: Tiết kiệm nhất
    {
        "max_new_tokens": 64,
        "use_cache": False,
        "do_sample": False,  # Greedy decoding
    }
]

# Thử từng config cho đến khi thành công
for i, config in enumerate(generation_configs):
    try:
        torch.cuda.empty_cache()
        output = model.generate(**model_inputs, **config)
        print(f"✅ Thành công với config {i+1}")
        break
    except torch.cuda.OutOfMemoryError:
        if i < len(generation_configs) - 1:
            print(f"⚠️ Config {i+1} thất bại, thử config {i+2}...")
        else:
            raise RuntimeError("Không đủ bộ nhớ ngay cả với config tiết kiệm nhất!")
```

### 3. Monitor GPU Memory

```python
def check_gpu_memory():
    """Kiểm tra và hiển thị bộ nhớ GPU"""
    if not torch.cuda.is_available():
        print("❌ Không có GPU")
        return
    
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    free = total - allocated
    
    print("=" * 50)
    print("THÔNG TIN GPU MEMORY")
    print("=" * 50)
    print(f"Total:     {total:.2f} GB")
    print(f"Allocated: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"Reserved:  {reserved:.2f} GB ({reserved/total*100:.1f}%)")
    print(f"Free:      {free:.2f} GB ({free/total*100:.1f}%)")
    
    # Cảnh báo
    if free < 1.0:
        print("\n⚠️ CẢNH BÁO: GPU memory còn rất ít!")
        print("   Có thể gặp lỗi OOM khi generate với prompt dài")
    elif free < 2.0:
        print("\n⚠️ Lưu ý: GPU memory còn ít")
        print("   Nên giải phóng memory trước khi generate")
    else:
        print("\n✅ GPU memory đủ để xử lý")
    print("=" * 50)

# Sử dụng
check_gpu_memory()
```

### 4. Hàm Generate an toàn (Safe Generate)

```python
def safe_generate(model, tokenizer, prompt, max_input_length=2000, max_new_tokens=256):
    """
    Generate an toàn với tự động xử lý prompt dài
    
    Args:
        model: Model đã load
        tokenizer: Tokenizer
        prompt: Prompt đầu vào
        max_input_length: Độ dài tối đa của input (tokens)
        max_new_tokens: Số tokens tối đa để generate
    """
    # 1. Clear cache trước
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. Tokenize với truncation
    model_inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding=False
    ).to(model.device)
    
    input_length = model_inputs.input_ids.shape[1]
    
    # 3. Cảnh báo nếu input bị truncate
    if input_length >= max_input_length:
        print(f"⚠️ Input quá dài, đã truncate xuống {max_input_length} tokens")
    
    # 4. Tự động giảm max_new_tokens nếu input dài
    effective_max_tokens = max_new_tokens
    if input_length > 1500:
        effective_max_tokens = min(max_new_tokens, 128)
        print(f"⚠️ Input dài ({input_length} tokens), giảm max_new_tokens xuống {effective_max_tokens}")
    
    # 5. Generate với config an toàn
    try:
        with torch.no_grad():
            output = model.generate(
                **model_inputs,
                max_new_tokens=effective_max_tokens,
                use_cache=False,  # Tắt cache để tiết kiệm
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 6. Decode output
        generated_text = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
        return generated_text
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ Out of Memory! Input length: {input_length} tokens")
        print("💡 Giải pháp:")
        print("   1. Giảm max_input_length")
        print("   2. Giảm max_new_tokens")
        print("   3. Đóng các process khác đang dùng GPU")
        print("   4. Sử dụng model nhỏ hơn")
        raise

# Sử dụng
answer = safe_generate(model, tokenizer, long_prompt)
```

---

## Tóm tắt và Kết luận

### Tại sao Input dài gây OOM?

1. **Attention mechanism** tính toán mối quan hệ giữa TẤT CẢ các cặp từ
2. **Bộ nhớ tăng theo bình phương** số từ (O(n²))
3. **GPU memory hạn chế** (thường 12-16 GB)
4. **Nhiều process** có thể chia sẻ GPU

### Giải pháp tốt nhất (Recommended)

**Kết hợp 3 giải pháp:**
1. ✅ **Truncation**: Giới hạn input ≤ 2000 tokens
2. ✅ **Tắt KV Cache**: `use_cache=False`
3. ✅ **Giảm max_new_tokens**: 128-256 thay vì 512

**Code mẫu:**
```python
# Tokenize với truncation
model_inputs = tokenizer(
    [prompt],
    return_tensors="pt",
    truncation=True,
    max_length=2000,
    padding=False
)

# Generate với config tiết kiệm
output = model.generate(
    **model_inputs,
    max_new_tokens=128,
    use_cache=False,
    temperature=0.7,
    do_sample=True,
)
```

### Checklist trước khi Generate

- [ ] Kiểm tra độ dài input (nên ≤ 2000 tokens)
- [ ] Clear GPU cache: `torch.cuda.empty_cache()`
- [ ] Kiểm tra GPU memory còn lại (nên ≥ 2 GB)
- [ ] Sử dụng `use_cache=False` nếu input dài
- [ ] Giảm `max_new_tokens` nếu input > 1500 tokens
- [ ] Có fallback configs nếu gặp OOM

---

## Tài liệu tham khảo

- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Transformers Generation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
- [Attention Mechanism (Original Paper)](https://arxiv.org/abs/1706.03762)
- [Qwen Model Documentation](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
