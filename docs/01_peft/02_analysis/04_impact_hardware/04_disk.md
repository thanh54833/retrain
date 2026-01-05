# Disk Storage và LLM

## 1. LLM tương tác với Disk như nào?

### Hiểu đơn giản

Disk Storage (Ổ cứng) giống như **kho lưu trữ lâu dài**:
- **RAM/VRAM**: Giống như bàn làm việc - nhanh nhưng nhỏ, mất dữ liệu khi tắt máy
- **Disk**: Giống như kho chứa - chậm hơn nhưng lớn, giữ dữ liệu vĩnh viễn

LLM cần Disk để:
- **Lưu trữ model files** (weights, configs, tokenizer)
- **Lưu trữ datasets** (training data, validation data)
- **Lưu checkpoints** (backup khi training)
- **Lưu logs và cache** (theo dõi, tối ưu)

**Không có Disk = Không có model để chạy!**

### 1.1. Tại sao cần Disk?

**Vấn đề nếu không có Disk:**

- **Không lưu được model**: Model phải ở đâu đó khi tắt máy
- **Mất dữ liệu**: RAM/VRAM mất dữ liệu khi tắt máy
- **Không có dataset**: Không thể training nếu không có data
- **Mất checkpoints**: Training bị gián đoạn → mất hết tiến độ

**Ví dụ:**
- Model 7B = 14 GB → Phải lưu trên disk
- Dataset = 100 GB → Phải lưu trên disk
- Checkpoints = 14 GB mỗi checkpoint → Cần nhiều disk space

**Kết luận:** Disk là bắt buộc để lưu trữ lâu dài!

## 2. Các loại Disk và ảnh hưởng đến LLM

### 2.1. HDD (Hard Disk Drive) - Ổ cứng cơ

**Đặc điểm:**
- **Tốc độ**: 100-200 MB/s (sequential read/write), 50-150 MB/s (random I/O)
- **Dung lượng**: 1-22 TB (lớn, rẻ)
- **Giá**: $15-30/TB (rất rẻ, 2024)
- **Độ bền**: Tốt, lâu dài, không giới hạn write cycles
- **Latency**: 5-15 ms (cao)

**Ảnh hưởng đến LLM:**
- **Loading model**: Chậm (mất vài phút để load 14 GB)
- **Training**: Chậm khi đọc dataset (I/O bottleneck)
- **Checkpoints**: Chậm khi save/load

**Phù hợp khi:**
- Lưu trữ lâu dài (archival)
- Dataset lớn, không cần tốc độ cao
- Budget hạn chế

**Ví dụ:**
- Load model 7B từ HDD: ~2-3 phút
- Load model 7B từ SSD: ~10-20 giây
- **Chênh lệch 6-18 lần!**

### 2.2. SSD (Solid State Drive) - Ổ cứng thể rắn

**Đặc điểm:**
- **Tốc độ**: 500-550 MB/s (SATA SSD), 2000-7000 MB/s (NVMe SSD)
- **Dung lượng**: 250 GB - 8 TB (SATA), 500 GB - 4 TB (NVMe)
- **Giá**: $40-100/TB (SATA), $60-200/TB (NVMe) - 2024
- **Độ bền**: Tốt, có giới hạn write cycles (TBW - Terabytes Written)
- **Latency**: 0.1-0.5 ms (rất thấp)

**Ảnh hưởng đến LLM:**
- **Loading model**: Nhanh (vài chục giây)
- **Training**: Nhanh khi đọc dataset
- **Checkpoints**: Nhanh khi save/load

**Phù hợp khi:**
- Production environment
- Cần tốc độ cao
- Active training/inference

**Ví dụ:**
- Load model 7B từ SSD: ~10-20 giây
- Load model 7B từ NVMe: ~5-10 giây
- **Nhanh hơn HDD 6-18 lần!**

### 2.3. NVMe SSD - Ổ cứng thể rắn tốc độ cao

**Đặc điểm:**
- **Tốc độ**: 2000-7000 MB/s (PCIe 3.0), 5000-12000 MB/s (PCIe 4.0), 10000-14000 MB/s (PCIe 5.0)
- **Dung lượng**: 500 GB - 4 TB (PCIe 3.0/4.0), 1-8 TB (PCIe 5.0)
- **Giá**: $60-150/TB (PCIe 3.0), $80-200/TB (PCIe 4.0), $100-250/TB (PCIe 5.0) - 2024
- **Độ bền**: Tốt, TBW cao hơn SATA SSD
- **Latency**: 0.05-0.2 ms (cực thấp)

**Ảnh hưởng đến LLM:**
- **Loading model**: Rất nhanh (vài giây)
- **Training**: Rất nhanh, không bottleneck
- **Checkpoints**: Rất nhanh

**Phù hợp khi:**
- High-performance production
- Frequent model loading
- Large-scale training

**Ví dụ:**
- Load model 7B từ NVMe: ~5-10 giây
- Load model 7B từ HDD: ~2-3 phút
- **Nhanh hơn HDD 12-36 lần!**

### 2.4. Bảng so sánh

| Loại Disk | Tốc độ đọc | Dung lượng | Giá/TB (2024) | Load Model 7B | Phù hợp |
|-----------|------------|------------|---------------|---------------|---------|
| HDD | 100-200 MB/s | 1-22 TB | $15-30 | 2-3 phút | Archival |
| SATA SSD | 500-550 MB/s | 250GB-8TB | $40-100 | 20-30 giây | General |
| NVMe PCIe 3.0 | 2000-3500 MB/s | 500GB-4TB | $60-150 | 5-10 giây | Production |
| NVMe PCIe 4.0 | 5000-7000 MB/s | 500GB-4TB | $80-200 | 3-5 giây | High-perf |
| NVMe PCIe 5.0 | 10000-14000 MB/s | 1-8 TB | $100-250 | 2-3 giây | Enterprise |

## 3. LLM sử dụng Disk như thế nào?

### 3.1. Lưu trữ Model Files

**Các file cần lưu:**

1. **Model Weights** (Trọng số model)
   - File lớn nhất: 7B model = 14 GB (FP16)
   - Format: `.bin`, `.safetensors`, `.pt`, `.pth`
   - Ví dụ: `llama2-7b.bin` = 14 GB

2. **Model Config** (Cấu hình)
   - File nhỏ: Vài KB đến vài MB
   - Format: `.json`, `.yaml`
   - Ví dụ: `config.json` = 1 KB

3. **Tokenizer Files** (Bộ từ điển)
   - File nhỏ: Vài MB đến vài chục MB
   - Format: `.json`, `.txt`, `.model`
   - Ví dụ: `tokenizer.json` = 5 MB

4. **Special Tokens** (Tokens đặc biệt)
   - File nhỏ: Vài KB
   - Format: `.json`
   - Ví dụ: `special_tokens_map.json` = 1 KB

**Tổng kích thước:**
- Model 7B: ~14 GB (chủ yếu là weights)
- Model 13B: ~26 GB
- Model 70B: ~140 GB

**Ví dụ cấu trúc:**
```
llama2-7b/
├── model.safetensors (14 GB)  # Model weights
├── config.json (1 KB)          # Config
├── tokenizer.json (5 MB)       # Tokenizer
└── special_tokens_map.json (1 KB)
```

### 3.2. Lưu trữ Datasets

**Các loại dataset:**

1. **Training Dataset**
   - Kích thước: Vài GB đến vài TB
   - Format: `.json`, `.jsonl`, `.parquet`, `.arrow`
   - Ví dụ: 1M samples × 1KB = 1 GB

2. **Validation Dataset**
   - Kích thước: Nhỏ hơn training (10-20%)
   - Format: Giống training
   - Ví dụ: 100K samples × 1KB = 100 MB

3. **Test Dataset**
   - Kích thước: Nhỏ nhất (5-10%)
   - Format: Giống training
   - Ví dụ: 50K samples × 1KB = 50 MB

**Ví dụ thực tế:**
- Dataset nhỏ: 1-10 GB
- Dataset trung bình: 10-100 GB
- Dataset lớn: 100 GB - 1 TB
- Dataset rất lớn: 1-10 TB

**Cấu trúc:**
```
datasets/
├── train.jsonl (10 GB)      # Training data
├── val.jsonl (1 GB)          # Validation data
└── test.jsonl (500 MB)        # Test data
```

### 3.3. Lưu trữ Checkpoints

**Checkpoint là gì?**

- **Backup model** tại một thời điểm trong training
- Cho phép resume training nếu bị gián đoạn
- Cho phép rollback nếu training không tốt

**Các loại checkpoint:**

1. **Full Checkpoint**
   - Lưu toàn bộ: Model + Optimizer states + Training state
   - Kích thước: 2-4× model size
   - Ví dụ: Model 7B → Checkpoint 28-56 GB

2. **Model-only Checkpoint**
   - Chỉ lưu model weights
   - Kích thước: 1× model size
   - Ví dụ: Model 7B → Checkpoint 14 GB

3. **LoRA Checkpoint**
   - Chỉ lưu LoRA weights (rất nhỏ)
   - Kích thước: Vài MB đến vài chục MB
   - Ví dụ: LoRA rank=8 → 4 MB

**Tần suất lưu:**
- Mỗi epoch: 1 checkpoint
- Mỗi N steps: 1 checkpoint
- Best model: Lưu khi có improvement

**Ví dụ:**
- Training 10 epochs → 10 checkpoints
- Model 7B, full checkpoint → 10 × 28 GB = 280 GB
- Cần nhiều disk space!

### 3.4. Lưu trữ Logs và Cache

**Logs (Nhật ký):**
- Training logs: Loss, accuracy, metrics
- Inference logs: Requests, responses, latency
- System logs: Errors, warnings
- Kích thước: Vài MB đến vài GB

**Cache (Bộ nhớ đệm):**
- Model cache: Cache model đã load
- Dataset cache: Cache dataset đã preprocess
- Token cache: Cache tokens đã tokenize
- Kích thước: Vài GB đến vài chục GB

**Ví dụ:**
```
logs/
├── training.log (100 MB)
├── inference.log (50 MB)
└── system.log (10 MB)

cache/
├── model_cache/ (14 GB)
├── dataset_cache/ (5 GB)
└── token_cache/ (1 GB)
```

## 4. Quá trình LLM tương tác với Disk

### 4.1. Loading Model (Tải model)

**Bước 1: Đọc từ Disk**
```
Disk → CPU RAM
- CPU đọc model file từ disk
- Tốc độ phụ thuộc vào loại disk:
  * HDD: 100-200 MB/s → 14 GB mất ~70-140 giây
  * SSD: 500-550 MB/s → 14 GB mất ~25-28 giây
  * NVMe: 2000-7000 MB/s → 14 GB mất ~2-7 giây
```

**Bước 2: Load vào RAM/VRAM**
```
CPU RAM → GPU VRAM (nếu có GPU)
- Copy model từ RAM sang VRAM
- Qua PCIe bus (16-32 GB/s)
- Mất vài giây
```

**Tổng thời gian:**
- HDD: ~2-3 phút
- SSD: ~30-40 giây
- NVMe: ~10-15 giây

**Ví dụ code:**
```python
# Đọc từ disk
model_path = "llama2-7b/model.safetensors"  # 14 GB trên disk
model = load_model(model_path)  # Đọc từ disk → RAM

# Load vào GPU
model = model.to("cuda")  # RAM → VRAM
```

### 4.2. Loading Dataset (Tải dataset)

**Bước 1: Đọc từ Disk**
```
Disk → CPU RAM
- CPU đọc dataset file từ disk
- Có thể đọc streaming (từng batch) hoặc load toàn bộ
```

**Bước 2: Preprocess (Nếu cần)**
```
CPU RAM: Tokenize, format, batch
- Xử lý dữ liệu
- Có thể lưu cache để lần sau nhanh hơn
```

**Bước 3: Load vào GPU (Training)**
```
CPU RAM → GPU VRAM
- Copy batch từ RAM sang VRAM
- Training xử lý batch này
```

**Ví dụ:**
```python
# Đọc từ disk
dataset = load_dataset("train.jsonl")  # Disk → RAM

# Preprocess
tokenized = tokenize(dataset)  # RAM

# Training
for batch in tokenized:
    batch = batch.to("cuda")  # RAM → VRAM
    train_step(batch)
```

### 4.3. Saving Checkpoint (Lưu checkpoint)

**Bước 1: Copy từ VRAM/RAM**
```
GPU VRAM → CPU RAM
- Copy model weights từ VRAM về RAM
- Copy optimizer states về RAM
```

**Bước 2: Ghi vào Disk**
```
CPU RAM → Disk
- Ghi model file vào disk
- Tốc độ phụ thuộc vào loại disk:
  * HDD: 100-200 MB/s
  * SSD: 500-550 MB/s
  * NVMe: 2000-7000 MB/s
```

**Ví dụ:**
```python
# Save checkpoint
checkpoint = {
    "model": model.state_dict(),  # VRAM → RAM
    "optimizer": optimizer.state_dict(),
    "epoch": epoch
}
torch.save(checkpoint, "checkpoint_epoch_5.pt")  # RAM → Disk
```

### 4.4. Luồng hoạt động tổng thể

**Inference:**
```
1. Disk (model file) → CPU RAM (load model)
2. CPU RAM → GPU VRAM (copy model)
3. Disk (input) → CPU RAM (load input) [optional]
4. CPU RAM → GPU VRAM (copy input)
5. GPU tính toán
6. GPU VRAM → CPU RAM (copy output)
7. CPU RAM → Disk (save output) [optional]
```

**Training:**
```
1. Disk (model) → CPU RAM → GPU VRAM (load model)
2. Disk (dataset) → CPU RAM (load dataset)
3. CPU RAM → GPU VRAM (copy batch)
4. GPU tính toán (forward + backward)
5. GPU VRAM → CPU RAM (copy checkpoint)
6. CPU RAM → Disk (save checkpoint)
7. Lặp lại từ bước 3
```

## 5. Tối ưu hóa Disk I/O cho LLM

### 5.1. Chọn loại Disk phù hợp

**Cho Production:**
- ✅ NVMe SSD: Tốc độ cao, phù hợp model loading thường xuyên
- ✅ SATA SSD: Cân bằng tốc độ và giá
- ❌ HDD: Chỉ cho archival, không phù hợp production

**Cho Development:**
- ✅ SATA SSD: Đủ nhanh, giá hợp lý
- ✅ NVMe SSD: Nếu có budget
- ⚠️ HDD: Chấp nhận được nếu không có lựa chọn

**Cho Archival:**
- ✅ HDD: Giá rẻ, dung lượng lớn
- ✅ Cloud storage: S3, GCS, Azure Blob

### 5.2. Dataset Caching

**Vấn đề:**
- Dataset lớn → đọc từ disk chậm
- Preprocessing tốn thời gian

**Giải pháp:**
- Cache dataset đã preprocess
- Lần đầu: Disk → Preprocess → Cache
- Lần sau: Cache → RAM (nhanh hơn nhiều!)

**Ví dụ:**
```python
# Lần đầu: Chậm
dataset = load_dataset("train.jsonl")  # Disk → RAM
tokenized = tokenize(dataset)  # Preprocess
save_cache(tokenized, "train_cache.arrow")  # Cache

# Lần sau: Nhanh
tokenized = load_cache("train_cache.arrow")  # Cache → RAM (nhanh!)
```

### 5.3. Streaming Dataset

**Vấn đề:**
- Dataset quá lớn → không load hết vào RAM
- Out of memory

**Giải pháp:**
- Streaming: Đọc từng batch từ disk
- Không load toàn bộ dataset vào RAM
- Tiết kiệm RAM

**Ví dụ:**
```python
# Không tốt: Load toàn bộ
dataset = load_dataset("train.jsonl")  # 100 GB → RAM (out of memory!)

# Tốt: Streaming
for batch in stream_dataset("train.jsonl"):  # Đọc từng batch
    process(batch)
```

### 5.4. Checkpoint Management

**Vấn đề:**
- Nhiều checkpoints → tốn disk space
- Checkpoints cũ không cần thiết

**Giải pháp:**
- Chỉ giữ N checkpoints gần nhất
- Xóa checkpoints cũ
- Compress checkpoints (nếu cần)

**Ví dụ:**
```python
# Chỉ giữ 3 checkpoints gần nhất
checkpoints = ["epoch_1.pt", "epoch_2.pt", "epoch_3.pt", "epoch_4.pt"]
keep_latest(checkpoints, n=3)  # Xóa epoch_1.pt
```

### 5.5. Parallel I/O

**Vấn đề:**
- I/O tuần tự → chậm
- GPU chờ data từ disk

**Giải pháp:**
- Parallel I/O: Đọc nhiều file cùng lúc
- Prefetch: Đọc batch tiếp theo khi đang train batch hiện tại
- Multi-threading: Nhiều threads đọc cùng lúc

**Ví dụ:**
```python
# Không tốt: Tuần tự
for batch in dataset:
    data = load_from_disk(batch)  # Chờ đọc
    train(data)

# Tốt: Parallel + Prefetch
for batch in prefetch_dataset(dataset, n_prefetch=2):
    train(batch)  # Đang train batch này, đã đọc batch tiếp theo
```

## 6. Disk Space Requirements

### 6.1. Model Storage

**Kích thước model:**
- Model 7B (FP16): ~14 GB
- Model 13B (FP16): ~26 GB
- Model 70B (FP16): ~140 GB

**Nhiều versions:**
- Base model: 14 GB
- Fine-tuned model: 14 GB
- LoRA adapter: 4 MB
- Quantized model (INT8): 7 GB

**Tổng:**
- 1 model: 14 GB
- 5 models: 70 GB
- 10 models: 140 GB

### 6.2. Dataset Storage

**Kích thước dataset:**
- Dataset nhỏ: 1-10 GB
- Dataset trung bình: 10-100 GB
- Dataset lớn: 100 GB - 1 TB
- Dataset rất lớn: 1-10 TB

**Nhiều datasets:**
- Training: 10 GB
- Validation: 1 GB
- Test: 500 MB
- Cache: 5 GB
- **Tổng: ~16.5 GB**

### 6.3. Checkpoint Storage

**Kích thước checkpoint:**
- Model 7B, full checkpoint: 28-56 GB
- Model 7B, model-only: 14 GB
- LoRA checkpoint: 4 MB

**Nhiều checkpoints:**
- 10 epochs, full checkpoint: 10 × 28 GB = 280 GB
- 10 epochs, model-only: 10 × 14 GB = 140 GB
- 10 epochs, LoRA: 10 × 4 MB = 40 MB

### 6.4. Tổng Disk Space Cần Thiết

**Cho Development:**
- Models: 50-100 GB
- Datasets: 10-50 GB
- Checkpoints: 50-200 GB
- Logs/Cache: 10-20 GB
- **Tổng: 120-370 GB**

**Cho Production:**
- Models: 200-500 GB (nhiều versions)
- Datasets: 100-500 GB
- Checkpoints: 500 GB - 2 TB
- Logs/Cache: 50-100 GB
- **Tổng: 850 GB - 3 TB**

**Cho Research:**
- Models: 500 GB - 2 TB (nhiều experiments)
- Datasets: 500 GB - 5 TB
- Checkpoints: 2-10 TB
- Logs/Cache: 100-500 GB
- **Tổng: 3-17 TB**

### 6.5. Bảng tổng hợp

| Use Case | Models | Datasets | Checkpoints | Logs/Cache | Tổng |
|----------|--------|----------|------------|------------|------|
| Development | 50-100 GB | 10-50 GB | 50-200 GB | 10-20 GB | 120-370 GB |
| Production | 200-500 GB | 100-500 GB | 500 GB-2 TB | 50-100 GB | 850 GB-3 TB |
| Research | 500 GB-2 TB | 500 GB-5 TB | 2-10 TB | 100-500 GB | 3-17 TB |

## 7. Best Practices

### 7.1. Chọn Disk phù hợp

**Cho Model Loading:**
- ✅ NVMe SSD: Nhanh nhất
- ✅ SATA SSD: Cân bằng
- ❌ HDD: Chậm, không phù hợp

**Cho Dataset Storage:**
- ✅ SATA SSD: Đủ nhanh
- ✅ HDD: Nếu dataset rất lớn, budget hạn chế
- ✅ Cloud storage: S3, GCS (scalable)

**Cho Checkpoints:**
- ✅ SATA SSD: Nhanh khi save/load
- ✅ HDD: Nếu có nhiều checkpoints, budget hạn chế
- ✅ Cloud storage: Backup lâu dài

### 7.2. Quản lý Disk Space

1. **Xóa checkpoints cũ**: Chỉ giữ N checkpoints gần nhất
2. **Compress**: Nén checkpoints nếu cần
3. **Archive**: Chuyển checkpoints cũ sang HDD hoặc cloud
4. **Monitor**: Theo dõi disk usage

**Ví dụ:**
```bash
# Xóa checkpoints cũ hơn 30 ngày
find checkpoints/ -name "*.pt" -mtime +30 -delete

# Compress checkpoints
gzip checkpoint_epoch_5.pt  # Giảm 50-70% kích thước
```

### 7.3. Tối ưu I/O

1. **Caching**: Cache dataset đã preprocess
2. **Streaming**: Đọc dataset theo batch, không load toàn bộ
3. **Parallel I/O**: Đọc nhiều file cùng lúc
4. **Prefetching**: Đọc batch tiếp theo khi đang train

### 7.4. Backup và Recovery

1. **Regular backups**: Backup models và checkpoints
2. **Version control**: Giữ nhiều versions
3. **Cloud backup**: Backup lên cloud (S3, GCS)
4. **Disaster recovery**: Có kế hoạch phục hồi

**Ví dụ:**
```bash
# Backup lên S3
aws s3 sync models/ s3://my-bucket/models/
aws s3 sync checkpoints/ s3://my-bucket/checkpoints/
```

## 8. Kết luận

### 8.1. Tóm tắt

**Disk Storage là bắt buộc cho LLM:**

1. **Lưu trữ Model Files**
   - Model weights: 14-140 GB
   - Config, tokenizer: Vài MB
   - Phải lưu trên disk để dùng lại

2. **Lưu trữ Datasets**
   - Training data: Vài GB đến vài TB
   - Validation/test data: Nhỏ hơn
   - Cần cho training

3. **Lưu trữ Checkpoints**
   - Full checkpoint: 2-4× model size
   - Model-only: 1× model size
   - LoRA: Vài MB
   - Cho phép resume training

4. **Lưu trữ Logs và Cache**
   - Logs: Vài MB đến vài GB
   - Cache: Vài GB đến vài chục GB
   - Hữu ích cho debugging và tối ưu

### 8.2. Lựa chọn Disk

**Cho Production:**
- ✅ NVMe SSD: Tốc độ cao nhất
- ✅ SATA SSD: Cân bằng tốc độ và giá
- ❌ HDD: Chỉ cho archival

**Cho Development:**
- ✅ SATA SSD: Đủ nhanh, giá hợp lý
- ✅ NVMe SSD: Nếu có budget

**Cho Archival:**
- ✅ HDD: Giá rẻ, dung lượng lớn
- ✅ Cloud storage: Scalable, reliable

### 8.3. Disk Space Requirements

- **Development**: 120-370 GB
- **Production**: 850 GB - 3 TB
- **Research**: 3-17 TB

### 8.4. Best Practices

1. **Chọn disk phù hợp**: NVMe cho production, SATA SSD cho development
2. **Quản lý disk space**: Xóa checkpoints cũ, compress, archive
3. **Tối ưu I/O**: Caching, streaming, parallel I/O, prefetching
4. **Backup**: Regular backups, version control, cloud backup

### 8.5. Tương lai

- **Disk nhanh hơn**: NVMe Gen 5 (14,000 MB/s)
- **Dung lượng lớn hơn**: 8TB+ NVMe SSD
- **Cloud storage**: Rẻ hơn, scalable hơn
- **Nhưng vẫn cần disk**: Local storage vẫn quan trọng

**Kết luận cuối cùng:** Disk Storage là thành phần không thể thiếu trong hệ thống LLM. Mặc dù chậm hơn RAM/VRAM, nhưng disk là nơi duy nhất lưu trữ lâu dài models, datasets, và checkpoints. Chọn disk phù hợp và tối ưu I/O là rất quan trọng!

---

**Ghi chú xác minh (2024):**
- Tốc độ disk đã được cập nhật theo thông số thực tế của các sản phẩm 2024
- Giá disk có thể dao động tùy thị trường và thời điểm
- PCIe 5.0 NVMe SSD đã có sẵn trên thị trường với tốc độ lên đến 14,000 MB/s
- Cloud storage (S3, GCS, Azure Blob) là lựa chọn tốt cho archival và backup
- Disk I/O có thể trở thành bottleneck trong training với dataset lớn nếu không tối ưu

