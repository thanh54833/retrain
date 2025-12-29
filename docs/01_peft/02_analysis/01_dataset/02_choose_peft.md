# Hướng Dẫn Chọn Phương Pháp PEFT Phù Hợp

## Tổng Quan

Việc chọn đúng phương pháp PEFT là rất quan trọng để đạt được hiệu suất tốt nhất với tài nguyên có sẵn. Tài liệu này phân tích ba nhóm phương pháp PEFT chính và hướng dẫn khi nào nên sử dụng phương pháp nào dựa trên các điều kiện và yêu cầu cụ thể.

## So Sánh Tổng Quan Các Phương Pháp

| Phương Pháp | Tham Số Trainable | Tốc Độ Huấn Luyện | Yêu Cầu Bộ Nhớ | Độ Phức Tạp | Hiệu Suất |
|-------------|-------------------|-------------------|----------------|-------------|-----------|
| **Prompt Tuning** | ~0.001% | Chậm (nhiều epochs) | Rất thấp | Thấp | Tốt |
| **P-tuning** | ~0.05% | Chậm (nhiều epochs) | Thấp | Trung bình | Tốt |
| **Prefix Tuning** | ~0.18% | Chậm (nhiều epochs) | Thấp | Cao | Rất tốt |
| **LoRA** | ~0.13-0.77% | Nhanh | Thấp | Thấp | Rất tốt |
| **LoHa** | ~1.44% | Nhanh | Trung bình | Trung bình | Rất tốt |
| **LoKr** | ~0.13% | Nhanh | Thấp | Trung bình | Tốt |
| **AdaLoRA** | ~0.59% | Nhanh | Trung bình | Cao | Rất tốt |
| **IA3** | ~0.02% | Rất nhanh | Rất thấp | Thấp | Tốt |

## Ma Trận Quyết Định

### 1. Dựa Trên Loại Model

#### Sequence-to-Sequence Models (T5, mT0, BART)
- ✅ **IA3**: Lựa chọn tốt nhất - ít tham số nhất, nhanh nhất
- ✅ **LoRA**: Lựa chọn tốt - phổ biến, ổn định
- ⚠️ **Prompt-based**: Có thể sử dụng nhưng không tối ưu

**Khuyến nghị**: Bắt đầu với **IA3**, nếu không đạt hiệu suất mong muốn thì chuyển sang **LoRA**.

#### Causal Language Models (GPT, LLaMA, Mistral)
- ✅ **LoRA**: Lựa chọn tốt nhất - được hỗ trợ rộng rãi, hiệu suất cao
- ✅ **Prompt-based**: Tốt cho generation tasks
- ⚠️ **IA3**: Có thể sử dụng nhưng ít tài liệu hơn

**Khuyến nghị**: Bắt đầu với **LoRA**, thử **Prompt Tuning** nếu cần ít tham số hơn.

#### Vision Models (ViT, CLIP)
- ✅ **LoRA**: Lựa chọn tốt nhất - phổ biến cho vision tasks
- ✅ **LoHa**: Tốt khi cần rank cao hơn
- ❌ **Prompt-based**: Không phù hợp
- ❌ **IA3**: Không được hỗ trợ tốt

**Khuyến nghị**: Sử dụng **LoRA** với `target_modules=["query", "value"]`.

#### Diffusion Models (Stable Diffusion)
- ✅ **LoRA**: Phổ biến nhất
- ✅ **LoHa**: Tốt cho controllable generation
- ✅ **LoKr**: Tốt cho vectorization
- ❌ **Prompt-based**: Không phù hợp
- ❌ **IA3**: Không được hỗ trợ

**Khuyến nghị**: Bắt đầu với **LoRA**, thử **LoHa** nếu cần rank cao hơn.

### 2. Dựa Trên Tài Nguyên Có Sẵn

#### Tài Nguyên Rất Hạn Chế (< 8GB GPU)
- ✅ **Prompt Tuning**: Ít tham số nhất (~0.001%)
- ✅ **IA3**: Rất ít tham số (~0.02%)
- ✅ **LoRA với r=8**: Ít tham số (~0.13%)

**Khuyến nghị**: Bắt đầu với **Prompt Tuning** hoặc **IA3**, nâng cấp lên **LoRA** nếu cần.

#### Tài Nguyên Hạn Chế (8-16GB GPU)
- ✅ **LoRA với r=16**: Cân bằng tốt
- ✅ **P-tuning**: Nếu phù hợp với task
- ✅ **IA3**: Cho Seq2Seq models

**Khuyến nghị**: Sử dụng **LoRA với r=16**, thử **IA3** cho Seq2Seq.

#### Tài Nguyên Đủ (16-24GB GPU)
- ✅ **LoRA với r=32-64**: Hiệu suất cao hơn
- ✅ **Prefix Tuning**: Cho các tác vụ phức tạp
- ✅ **LoHa**: Khi cần rank cao hơn

**Khuyến nghị**: Sử dụng **LoRA với r=32**, thử **Prefix Tuning** hoặc **LoHa** nếu cần.

#### Tài Nguyên Dồi Dào (> 24GB GPU)
- ✅ **LoRA với r=64+**: Hiệu suất tối đa
- ✅ **AdaLoRA**: Phân bổ tham số thông minh
- ✅ **Prefix Tuning**: Cho các tác vụ phức tạp nhất

**Khuyến nghị**: Sử dụng **LoRA với r=64** hoặc **AdaLoRA** cho tối ưu hóa.

### 3. Dựa Trên Loại Tác Vụ

#### Text Classification
- ✅ **LoRA**: Lựa chọn tốt nhất
- ✅ **Prompt Tuning**: Nếu cần ít tham số
- ⚠️ **IA3**: Có thể sử dụng

**Khuyến nghị**: **LoRA với r=16-32**.

#### Text Generation
- ✅ **Prompt Tuning**: Tốt cho generation
- ✅ **LoRA**: Lựa chọn phổ biến
- ✅ **Prefix Tuning**: Cho các tác vụ phức tạp

**Khuyến nghị**: Bắt đầu với **Prompt Tuning**, nâng cấp lên **LoRA** hoặc **Prefix Tuning** nếu cần.

#### Translation
- ✅ **IA3**: Lựa chọn tốt nhất cho Seq2Seq
- ✅ **LoRA**: Lựa chọn tốt
- ⚠️ **Prompt-based**: Có thể sử dụng

**Khuyến nghị**: **IA3** cho Seq2Seq models, **LoRA** cho các model khác.

#### Summarization
- ✅ **IA3**: Tốt cho Seq2Seq
- ✅ **LoRA**: Lựa chọn phổ biến
- ✅ **Prefix Tuning**: Cho các tác vụ phức tạp

**Khuyến nghị**: **IA3** cho Seq2Seq, **LoRA** cho các model khác.

#### Image Classification
- ✅ **LoRA**: Lựa chọn duy nhất phù hợp
- ✅ **LoHa**: Nếu cần rank cao hơn

**Khuyến nghị**: **LoRA với r=16-32**.

#### Controllable Generation (Text-to-Image)
- ✅ **LoHa**: Tốt cho rank cao
- ✅ **LoRA**: Lựa chọn phổ biến
- ✅ **OFT/BOFT**: Tốt cho bảo toàn subject

**Khuyến nghị**: **LoHa** hoặc **LoRA với r=32+**.

### 4. Dựa Trên Yêu Cầu Hiệu Suất

#### Cần Hiệu Suất Tối Đa
- ✅ **LoRA với r=64+**: Hiệu suất cao nhất
- ✅ **Prefix Tuning**: Cho các tác vụ phức tạp
- ✅ **AdaLoRA**: Phân bổ tham số thông minh

**Khuyến nghị**: **LoRA với r=64** hoặc **AdaLoRA**.

#### Cần Cân Bằng Hiệu Suất và Hiệu Quả
- ✅ **LoRA với r=16-32**: Cân bằng tốt
- ✅ **P-tuning**: Cho prompt-based
- ✅ **IA3**: Cho Seq2Seq

**Khuyến nghị**: **LoRA với r=16-32**.

#### Cần Hiệu Quả Tối Đa (Ít Tham Số)
- ✅ **Prompt Tuning**: Ít tham số nhất
- ✅ **IA3**: Rất ít tham số
- ✅ **LoRA với r=8**: Ít tham số

**Khuyến nghị**: **Prompt Tuning** hoặc **IA3**.

### 5. Dựa Trên Kích Thước Dataset

#### Dataset Nhỏ (< 1K mẫu)
- ✅ **Prompt Tuning**: Ít tham số, ít overfitting
- ✅ **LoRA với r=8**: Ít tham số
- ⚠️ **LoRA với r cao**: Có thể overfitting

**Khuyến nghị**: **Prompt Tuning** hoặc **LoRA với r=8**.

#### Dataset Vừa (1K-10K mẫu)
- ✅ **LoRA với r=16**: Cân bằng tốt
- ✅ **P-tuning**: Cho prompt-based
- ✅ **IA3**: Cho Seq2Seq

**Khuyến nghị**: **LoRA với r=16** hoặc **IA3**.

#### Dataset Lớn (> 10K mẫu)
- ✅ **LoRA với r=32-64**: Hiệu suất cao
- ✅ **Prefix Tuning**: Cho các tác vụ phức tạp
- ✅ **AdaLoRA**: Phân bổ tham số thông minh

**Khuyến nghị**: **LoRA với r=32-64** hoặc **AdaLoRA**.

### 6. Dựa Trên Thời Gian Huấn Luyện

#### Cần Huấn Luyện Nhanh
- ✅ **IA3**: Nhanh nhất
- ✅ **LoRA**: Nhanh
- ❌ **Prompt-based**: Chậm (cần nhiều epochs)

**Khuyến nghị**: **IA3** hoặc **LoRA**.

#### Có Thời Gian Huấn Luyện Đủ
- ✅ **LoRA**: Cân bằng tốt
- ✅ **Prompt-based**: Có thể sử dụng
- ✅ **AdaLoRA**: Cần thời gian để phân bổ

**Khuyến nghị**: **LoRA** hoặc **Prompt-based**.

#### Có Thời Gian Huấn Luyện Dài
- ✅ **LoRA với r cao**: Hiệu suất tối đa
- ✅ **Prefix Tuning**: Cho các tác vụ phức tạp
- ✅ **AdaLoRA**: Tối ưu hóa phân bổ

**Khuyến nghị**: **LoRA với r=64+** hoặc **AdaLoRA**.

### 7. Dựa Trên Kinh Nghiệm và Hỗ Trợ

#### Mới Bắt Đầu với PEFT
- ✅ **LoRA**: Đơn giản nhất, nhiều tài liệu
- ✅ **Prompt Tuning**: Đơn giản
- ❌ **AdaLoRA**: Phức tạp, cần custom training loop

**Khuyến nghị**: Bắt đầu với **LoRA**.

#### Có Kinh Nghiệm
- ✅ **Tất cả phương pháp**: Có thể thử nghiệm
- ✅ **AdaLoRA**: Cho tối ưu hóa
- ✅ **LoHa/LoKr**: Cho các use case đặc biệt

**Khuyến nghị**: Thử nghiệm các phương pháp khác nhau.

#### Cần Hỗ Trợ Cộng Đồng
- ✅ **LoRA**: Hỗ trợ tốt nhất
- ✅ **Prompt Tuning**: Hỗ trợ tốt
- ⚠️ **IA3**: Ít hỗ trợ hơn
- ⚠️ **LoHa/LoKr**: Ít hỗ trợ hơn

**Khuyến nghị**: **LoRA** hoặc **Prompt Tuning**.

## Workflow Quyết Định

### Bước 1: Xác Định Loại Model
```
Seq2Seq Model?
├─ Yes → Xem xét IA3 hoặc LoRA
└─ No → Xem xét LoRA hoặc Prompt-based
```

### Bước 2: Đánh Giá Tài Nguyên
```
GPU Memory < 8GB?
├─ Yes → Prompt Tuning hoặc IA3
└─ No → LoRA với r phù hợp
```

### Bước 3: Xác Định Yêu Cầu Hiệu Suất
```
Cần hiệu suất tối đa?
├─ Yes → LoRA r=64+ hoặc AdaLoRA
└─ No → LoRA r=16-32 hoặc Prompt Tuning
```

### Bước 4: Xem Xét Dataset
```
Dataset < 1K?
├─ Yes → Prompt Tuning hoặc LoRA r=8
└─ No → LoRA r=16-32
```

### Bước 5: Quyết Định Cuối Cùng
Dựa trên tất cả các yếu tố trên, chọn phương pháp phù hợp nhất.

## Ví Dụ Cụ Thể

### Ví Dụ 1: Fine-tune LLaMA 7B cho Chatbot
- **Model**: Causal LM (LLaMA)
- **Tài nguyên**: 16GB GPU
- **Tác vụ**: Text generation
- **Dataset**: 5K mẫu
- **Yêu cầu**: Hiệu suất tốt

**Khuyến nghị**: **LoRA với r=32, target_modules=["q_proj", "v_proj"]**

### Ví Dụ 2: Fine-tune T5 cho Translation
- **Model**: Seq2Seq (T5)
- **Tài nguyên**: 8GB GPU
- **Tác vụ**: Translation
- **Dataset**: 10K mẫu
- **Yêu cầu**: Hiệu quả tối đa

**Khuyến nghị**: **IA3** (ít tham số nhất, nhanh nhất cho Seq2Seq)

### Ví Dụ 3: Fine-tune ViT cho Image Classification
- **Model**: Vision Transformer
- **Tài nguyên**: 12GB GPU
- **Tác vụ**: Image classification
- **Dataset**: 20K mẫu
- **Yêu cầu**: Hiệu suất tốt

**Khuyến nghị**: **LoRA với r=16, target_modules=["query", "value"]**

### Ví Dụ 4: Fine-tune GPT-2 cho Text Classification
- **Model**: Causal LM (GPT-2)
- **Tài nguyên**: 6GB GPU
- **Tác vụ**: Text classification
- **Dataset**: 2K mẫu
- **Yêu cầu**: Ít tham số

**Khuyến nghị**: **Prompt Tuning** (ít tham số nhất, phù hợp cho classification)

### Ví Dụ 5: Fine-tune Stable Diffusion
- **Model**: Diffusion Model
- **Tài nguyên**: 24GB GPU
- **Tác vụ**: Text-to-image
- **Dataset**: 50K mẫu
- **Yêu cầu**: Hiệu suất cao

**Khuyến nghị**: **LoRA với r=64** hoặc **LoHa** (cho rank cao hơn)

## Bảng Tóm Tắt Nhanh

| Điều Kiện | Phương Pháp Đề Xuất | Rank/Config |
|-----------|---------------------|-------------|
| **Seq2Seq + ít tài nguyên** | IA3 | - |
| **Seq2Seq + đủ tài nguyên** | LoRA | r=16-32 |
| **Causal LM + ít tài nguyên** | Prompt Tuning | - |
| **Causal LM + đủ tài nguyên** | LoRA | r=16-32 |
| **Vision Model** | LoRA | r=16-32 |
| **Diffusion Model** | LoRA/LoHa | r=32-64 |
| **Dataset nhỏ** | Prompt Tuning | - |
| **Dataset lớn** | LoRA | r=32-64 |
| **Cần hiệu suất tối đa** | LoRA/AdaLoRA | r=64+ |
| **Cần hiệu quả tối đa** | Prompt Tuning/IA3 | - |
| **Mới bắt đầu** | LoRA | r=16 |

## Best Practices

### 1. Bắt Đầu Đơn Giản
- Luôn bắt đầu với **LoRA r=16** nếu không chắc chắn
- Đây là điểm khởi đầu tốt cho hầu hết các trường hợp

### 2. Thử Nghiệm Tăng Dần
- Bắt đầu với rank thấp (r=8-16)
- Tăng dần nếu cần hiệu suất cao hơn
- Giảm nếu overfitting

### 3. Monitor Metrics
- Theo dõi cả training và validation loss
- Điều chỉnh rank dựa trên hiệu suất
- Sử dụng early stopping nếu cần

### 4. Kết Hợp Khi Cần
- Có thể kết hợp nhiều phương pháp
- Ví dụ: LoRA + Prompt Tuning cho một số tác vụ

### 5. Tối Ưu Hóa Dần
- Bắt đầu với phương pháp đơn giản
- Nâng cấp lên phương pháp phức tạp hơn nếu cần
- Đánh giá trade-off giữa hiệu suất và chi phí

## Kết Luận

Việc chọn phương pháp PEFT phù hợp phụ thuộc vào nhiều yếu tố:

1. **Loại model**: Seq2Seq → IA3, Causal LM → LoRA, Vision → LoRA
2. **Tài nguyên**: Ít → Prompt Tuning/IA3, Đủ → LoRA
3. **Yêu cầu hiệu suất**: Cao → LoRA r cao, Thấp → Prompt Tuning
4. **Kích thước dataset**: Nhỏ → Prompt Tuning, Lớn → LoRA
5. **Kinh nghiệm**: Mới → LoRA, Có kinh nghiệm → Thử nghiệm

**Khuyến nghị chung**: 
- Bắt đầu với **LoRA r=16** cho hầu hết các trường hợp
- Sử dụng **IA3** cho Seq2Seq models khi cần ít tham số
- Sử dụng **Prompt Tuning** khi tài nguyên rất hạn chế
- Nâng cấp lên **LoRA r cao** hoặc **AdaLoRA** khi cần hiệu suất tối đa

Nhớ rằng không có phương pháp nào là "tốt nhất" cho mọi trường hợp. Việc lựa chọn phụ thuộc vào tình huống cụ thể của bạn.

## Tài Liệu Tham Khảo

- [Prompt-based Methods Guide](../01_research/03_peft_method_prompt_base.md)
- [LoRA Methods Guide](../01_research/04_peft_method_lora_method.md)
- [IA3 Guide](../01_research/05_peft_method_ia3.md)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)

