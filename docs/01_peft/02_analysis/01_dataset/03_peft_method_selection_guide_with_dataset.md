# Hướng Dẫn Chọn Phương Pháp PEFT Cho Dataset Rephrase

## Tổng Quan

Tài liệu này phân tích dataset Rephrase và đề xuất phương pháp PEFT tối ưu nhất dựa trên đặc điểm cụ thể của dataset này.

## Phân Tích Dataset

### Đặc Điểm Chính

| Đặc Điểm | Giá Trị | Ý Nghĩa |
|----------|---------|---------|
| **Số lượng samples** | 1,000 | Dataset nhỏ, cần ít tham số để tránh overfitting |
| **Task type** | Text-to-JSON Generation | Cần model generation tốt, output structured |
| **Language** | Tiếng Việt | Cần model hỗ trợ tiếng Việt |
| **Input length** | Avg 15.72 chars (1-74) | Input ngắn, đơn giản |
| **Output length** | Avg 25-120 chars | Output dài hơn, phức tạp (JSON) |
| **Class imbalance** | 96.1% in-scope, 3.9% out-of-scope | Mất cân bằng, cần xử lý cẩn thận |
| **Output format** | Structured JSON | Cần valid JSON, chính xác cao |

### Yêu Cầu Cụ Thể

1. **Text Generation**: Model cần generate text (JSON output)
2. **Structured Output**: Output phải là valid JSON với 5 trường
3. **Vietnamese Language**: Model phải hiểu và tạo text tiếng Việt
4. **Small Dataset**: 1K samples → Cần ít tham số để tránh overfitting
5. **Precision**: Cần độ chính xác cao cho JSON format

## Đánh Giá Các Phương Pháp PEFT

### 1. Prompt Tuning

**Đặc điểm**:
- Tham số trainable: ~0.001% (ít nhất)
- Tốc độ: Chậm (cần nhiều epochs)
- Độ phức tạp: Thấp

**Phù hợp với dataset này?**
- ✅ **Ưu điểm**: 
  - Ít tham số nhất, phù hợp dataset nhỏ
  - Ít nguy cơ overfitting
  - Đơn giản, dễ triển khai
- ❌ **Nhược điểm**: 
  - Chậm (cần 20-50 epochs)
  - Có thể không đủ mạnh cho structured JSON output phức tạp
  - Learning rate cao, có thể không ổn định

**Đánh giá**: ⚠️ **Có thể thử nhưng không tối ưu**

### 2. P-tuning

**Đặc điểm**:
- Tham số trainable: ~0.05%
- Tốc độ: Chậm (cần nhiều epochs)
- Độ phức tạp: Trung bình

**Phù hợp với dataset này?**
- ✅ **Ưu điểm**: 
  - Ít tham số, phù hợp dataset nhỏ
  - Linh hoạt trong việc đặt prompt
- ❌ **Nhược điểm**: 
  - Chậm (cần nhiều epochs)
  - Có thể không đủ mạnh cho JSON generation
  - Phức tạp hơn Prompt Tuning

**Đánh giá**: ⚠️ **Không khuyến nghị**

### 3. Prefix Tuning

**Đặc điểm**:
- Tham số trainable: ~0.18%
- Tốc độ: Chậm (cần nhiều epochs)
- Độ phức tạp: Cao

**Phù hợp với dataset này?**
- ✅ **Ưu điểm**: 
  - Hiệu suất cao
  - Tốt cho các tác vụ phức tạp
- ❌ **Nhược điểm**: 
  - Nhiều tham số hơn (0.18% có thể quá nhiều cho 1K samples)
  - Nguy cơ overfitting cao với dataset nhỏ
  - Chậm, cần nhiều epochs

**Đánh giá**: ❌ **Không phù hợp** (quá nhiều tham số cho dataset nhỏ)

### 4. LoRA

**Đặc điểm**:
- Tham số trainable: ~0.13-0.77% (tùy rank)
- Tốc độ: Nhanh
- Độ phức tạp: Thấp

**Phù hợp với dataset này?**
- ✅ **Ưu điểm**: 
  - Nhanh, hội tụ nhanh (5-10 epochs)
  - Có thể điều chỉnh rank (r=8-16 cho dataset nhỏ)
  - Ổn định, được hỗ trợ rộng rãi
  - Tốt cho text generation
  - Có thể kiểm soát số tham số qua rank
- ⚠️ **Lưu ý**: 
  - Cần chọn rank phù hợp (r=8-16 cho dataset nhỏ)
  - Có thể cần thử nghiệm với các rank khác nhau

**Đánh giá**: ✅ **Phù hợp nhất** - Lựa chọn tối ưu

### 5. LoHa

**Đặc điểm**:
- Tham số trainable: ~1.44%
- Tốc độ: Nhanh
- Độ phức tạp: Trung bình

**Phù hợp với dataset này?**
- ✅ **Ưu điểm**: 
  - Rank hiệu quả cao hơn (r²)
  - Nhanh
- ❌ **Nhược điểm**: 
  - Nhiều tham số hơn (1.44% có thể quá nhiều cho 1K samples)
  - Nguy cơ overfitting cao
  - Phức tạp hơn LoRA

**Đánh giá**: ❌ **Không phù hợp** (quá nhiều tham số)

### 6. LoKr

**Đặc điểm**:
- Tham số trainable: ~0.13%
- Tốc độ: Nhanh
- Độ phức tạp: Trung bình

**Phù hợp với dataset này?**
- ✅ **Ưu điểm**: 
  - Ít tham số (tương tự LoRA r=8)
  - Nhanh
- ❌ **Nhược điểm**: 
  - Phức tạp hơn LoRA
  - Ít tài liệu và ví dụ hơn
  - Không có lợi thế rõ ràng so với LoRA

**Đánh giá**: ⚠️ **Có thể thử nhưng LoRA tốt hơn**

### 7. AdaLoRA

**Đặc điểm**:
- Tham số trainable: ~0.59%
- Tốc độ: Nhanh
- Độ phức tạp: Cao

**Phù hợp với dataset này?**
- ✅ **Ưu điểm**: 
  - Phân bổ tham số thông minh
  - Hiệu suất cao
- ❌ **Nhược điểm**: 
  - Nhiều tham số hơn (0.59% có thể quá nhiều cho 1K samples)
  - Phức tạp, cần custom training loop
  - Nguy cơ overfitting với dataset nhỏ

**Đánh giá**: ❌ **Không phù hợp** (quá nhiều tham số và phức tạp)

### 8. IA3

**Đặc điểm**:
- Tham số trainable: ~0.02%
- Tốc độ: Rất nhanh
- Độ phức tạp: Thấp

**Phù hợp với dataset này?**
- ✅ **Ưu điểm**: 
  - Rất ít tham số (~0.02%), phù hợp dataset nhỏ
  - Rất nhanh, hội tụ nhanh
  - Ít nguy cơ overfitting
- ⚠️ **Lưu ý**: 
  - Tốt nhất cho Seq2Seq models
  - Có thể không đủ mạnh cho structured JSON output phức tạp
  - Ít tài liệu hơn LoRA

**Đánh giá**: ✅ **Phù hợp** - Lựa chọn thay thế tốt nếu dùng Seq2Seq model

## So Sánh Trực Tiếp

| Phương Pháp | Tham Số | Tốc Độ | Phù Hợp Dataset Nhỏ | Phù Hợp JSON Gen | Đánh Giá |
|-------------|---------|--------|---------------------|------------------|----------|
| **Prompt Tuning** | ~0.001% | Chậm | ✅ Rất tốt | ⚠️ Có thể | ⚠️ Có thể thử |
| **P-tuning** | ~0.05% | Chậm | ✅ Tốt | ⚠️ Có thể | ❌ Không khuyến nghị |
| **Prefix Tuning** | ~0.18% | Chậm | ❌ Quá nhiều | ✅ Tốt | ❌ Không phù hợp |
| **LoRA (r=8)** | ~0.13% | Nhanh | ✅ Rất tốt | ✅ Tốt | ✅ **Tối ưu** |
| **LoRA (r=16)** | ~0.26% | Nhanh | ✅ Tốt | ✅ Rất tốt | ✅ **Tối ưu** |
| **LoHa** | ~1.44% | Nhanh | ❌ Quá nhiều | ✅ Tốt | ❌ Không phù hợp |
| **LoKr** | ~0.13% | Nhanh | ✅ Tốt | ✅ Tốt | ⚠️ Có thể thử |
| **AdaLoRA** | ~0.59% | Nhanh | ❌ Quá nhiều | ✅ Tốt | ❌ Không phù hợp |
| **IA3** | ~0.02% | Rất nhanh | ✅ Rất tốt | ⚠️ Có thể | ✅ Thay thế tốt |

## Đề Xuất Giải Pháp

### Giải Pháp Tối Ưu: LoRA với r=8 hoặc r=16

**Lý do chọn LoRA**:

1. **Phù hợp dataset nhỏ**:
   - Với r=8: ~0.13% tham số (phù hợp 1K samples)
   - Với r=16: ~0.26% tham số (vẫn an toàn cho 1K samples)
   - Có thể điều chỉnh rank dễ dàng

2. **Nhanh và hiệu quả**:
   - Hội tụ nhanh (5-10 epochs)
   - Không cần nhiều epochs như prompt-based methods
   - Tiết kiệm thời gian và tài nguyên

3. **Tốt cho text generation**:
   - LoRA được thiết kế tốt cho generation tasks
   - Có thể handle structured output (JSON)
   - Được sử dụng rộng rãi cho các tác vụ tương tự

4. **Ổn định và được hỗ trợ**:
   - Nhiều tài liệu và ví dụ
   - Cộng đồng hỗ trợ tốt
   - Dễ debug và troubleshoot

5. **Linh hoạt**:
   - Có thể điều chỉnh rank dựa trên kết quả
   - Có thể thử nghiệm với các target_modules khác nhau
   - Dễ dàng fine-tune hyperparameters

### Cấu Hình Đề Xuất

#### Option 1: LoRA Conservative (r=8) - Khuyến nghị bắt đầu

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                          # Rank thấp cho dataset nhỏ
    lora_alpha=16,               # 2 * r
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
    lora_dropout=0.1,            # Dropout để tránh overfitting
    bias="none",                  # Không train bias
    task_type="CAUSAL_LM" hoặc "SEQ_2_SEQ_LM"  # Tùy model base
)
```

**Khi nào dùng**:
- Bắt đầu thử nghiệm
- Muốn ít tham số nhất có thể
- Nguy cơ overfitting cao

#### Option 2: LoRA Balanced (r=16) - Khuyến nghị chính

```python
lora_config = LoraConfig(
    r=16,                         # Rank cân bằng
    lora_alpha=32,                # 2 * r
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM" hoặc "SEQ_2_SEQ_LM"
)
```

**Khi nào dùng**:
- Lựa chọn chính, cân bằng tốt
- Cần hiệu suất tốt hơn r=8
- Dataset đủ để support r=16

### Giải Pháp Thay Thế: IA3 (nếu dùng Seq2Seq Model)

**Khi nào dùng IA3**:
- Nếu sử dụng Seq2Seq model (T5, mT5, mT0)
- Cần ít tham số nhất có thể
- Muốn hội tụ nhanh nhất

**Cấu hình**:
```python
from peft import IA3Config, get_peft_model

ia3_config = IA3Config(
    task_type="SEQ_2_SEQ_LM"
)
```

## Kế Hoạch Thực Hiện

### Phase 1: Baseline với LoRA r=8

1. **Chuẩn bị**:
   - Chia dataset: 80% train, 10% validation, 10% test
   - Chọn base model hỗ trợ tiếng Việt (PhoBERT, VinAI-BERT, hoặc multilingual model)
   - Format data cho training

2. **Training**:
   - LoRA r=8
   - Learning rate: 2e-4
   - Batch size: 8-16 (tùy GPU)
   - Epochs: 5-10
   - Monitor validation loss

3. **Đánh giá**:
   - JSON validity rate
   - Field completeness
   - Accuracy cho is_in_scope
   - Quality của generated text

### Phase 2: Optimization với LoRA r=16

1. **Nếu Phase 1 tốt nhưng cần cải thiện**:
   - Tăng rank lên r=16
   - Giữ nguyên các hyperparameters khác
   - So sánh kết quả

2. **Nếu Phase 1 overfitting**:
   - Giảm learning rate
   - Tăng dropout
   - Thêm regularization

### Phase 3: Fine-tuning

1. **Hyperparameter tuning**:
   - Learning rate: 1e-4 đến 5e-4
   - Dropout: 0.05 đến 0.2
   - Batch size: 4 đến 32

2. **Target modules**:
   - Thử chỉ `["q_proj", "v_proj"]` (ít tham số hơn)
   - Thử thêm MLP layers nếu cần

3. **Early stopping**:
   - Monitor validation loss
   - Stop nếu không cải thiện sau 3 epochs

## Lưu Ý Quan Trọng

### 1. Base Model Selection

**Yêu cầu**:
- Hỗ trợ tiếng Việt tốt
- Có khả năng generation tốt
- Phù hợp với structured output

**Đề xuất**:
- **PhoBERT**: Tốt cho tiếng Việt, nhưng là encoder-only (cần thêm decoder)
- **mT5/mT0**: Multilingual, Seq2Seq, tốt cho generation
- **LLaMA 2/3**: Nếu có, tốt cho generation, cần fine-tune cho tiếng Việt
- **GPT-2 Vietnamese**: Nếu có, tốt cho generation

### 2. JSON Format Validation

**Vấn đề**: Model có thể generate invalid JSON

**Giải pháp**:
- Post-processing để fix JSON
- JSON schema validation
- Constrained decoding nếu có thể
- Training với JSON examples rõ ràng

### 3. Class Imbalance

**Vấn đề**: 96.1% in-scope vs 3.9% out-of-scope

**Giải pháp**:
- Weighted loss function
- Oversampling out-of-scope samples
- Focal loss để focus vào hard examples
- Stratified sampling trong train/val/test split

### 4. Empty Fields Handling

**Vấn đề**: 
- In-scope: `reasoning` thường rỗng
- Out-of-scope: `keyword` và `message_banner` rỗng

**Giải pháp**:
- Training với explicit empty string examples
- Special token cho empty fields
- Post-processing để đảm bảo pattern đúng

### 5. Vietnamese Language

**Vấn đề**: Model cần hiểu và tạo text tiếng Việt tự nhiên

**Giải pháp**:
- Chọn model đã được pretrain trên tiếng Việt
- Fine-tune với Vietnamese dataset nếu cần
- Kiểm tra chất lượng text generation

## Metrics Đánh Giá

### 1. JSON Validity
- Tỷ lệ output là valid JSON
- Target: >95%

### 2. Field Completeness
- Tỷ lệ các trường được điền đầy đủ
- Target: >98%

### 3. is_in_scope Accuracy
- Độ chính xác phân loại in-scope/out-of-scope
- Target: >90% (cả hai classes)

### 4. Text Quality
- BLEU/ROUGE score cho keyword và messages
- Human evaluation cho naturalness

### 5. Consistency
- Keyword liên quan đến query
- Message tone và style nhất quán

## Kết Luận

**Giải pháp tối ưu cho dataset Rephrase**: **LoRA với r=8 hoặc r=16**

**Lý do**:
1. Phù hợp dataset nhỏ (1K samples)
2. Nhanh, hội tụ nhanh (5-10 epochs)
3. Tốt cho text generation và structured output
4. Ổn định, được hỗ trợ rộng rãi
5. Linh hoạt, dễ điều chỉnh

**Kế hoạch**:
1. Bắt đầu với **LoRA r=8** để baseline
2. Nâng cấp lên **r=16** nếu cần hiệu suất cao hơn
3. Fine-tune hyperparameters dựa trên kết quả
4. Xử lý class imbalance và JSON validation

**Giải pháp thay thế**: **IA3** nếu sử dụng Seq2Seq model và cần ít tham số nhất có thể.

## Tài Liệu Tham Khảo

- [Dataset Analysis](./01_rephrase.md)
- [PEFT Method Selection Guide](./02_peft_method_selection_guide.md)
- [LoRA Methods Guide](../01_research/04_peft_method_lora_method.md)
- [IA3 Guide](../01_research/05_peft_method_ia3.md)
- [Prompt-based Methods Guide](../01_research/03_peft_method_prompt_base.md)

