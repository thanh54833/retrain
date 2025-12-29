# SFT vs DFO vs PEFT vs GRPO: Chọn Chiến Lược Fine-Tuning Đúng Cho LLM

## Tổng Quan

Khi làm việc với Large Language Models (LLM), việc chọn đúng chiến lược fine-tuning là rất quan trọng. Mỗi phương pháp có ưu và nhược điểm riêng, phù hợp với các use case khác nhau. Tài liệu này so sánh bốn phương pháp phổ biến: **SFT**, **DFO**, **PEFT**, và **GRPO**.

## 1. Supervised Fine-Tuning (SFT)

### Định Nghĩa

**Supervised Fine-Tuning (SFT)** là phương pháp fine-tuning truyền thống, cập nhật tất cả các tham số của model trên một tập dữ liệu có nhãn cụ thể cho tác vụ.

### Cách Hoạt Động

1. Sử dụng một LLM đã được pretrained
2. Huấn luyện model trên tập dữ liệu cụ thể cho tác vụ (ví dụ: "BIM sheet correction → corrected sheet name")
3. Cập nhật **tất cả các tham số** trong mạng neural

### Ưu Điểm

- ✅ **Tốt nhất cho chuyển giao kiến thức chuyên môn**: Model hoàn toàn thích nghi với domain cụ thể
- ✅ **Hiệu suất cao**: Đạt kết quả tốt nhất trên các tác vụ phức tạp
- ✅ **Hoạt động tốt với dữ liệu lớn**: Phù hợp khi có hàng chục nghìn mẫu
- ✅ **Tối ưu toàn diện**: Toàn bộ model được tối ưu cho tác vụ cụ thể

### Nhược Điểm

- ❌ **Tốn kém**: Yêu cầu nhiều tài nguyên GPU và bộ nhớ
- ❌ **Cần nhiều dữ liệu**: Thường cần hàng chục nghìn mẫu dữ liệu chất lượng cao
- ❌ **Catastrophic Forgetting**: Nguy cơ mất đi kỹ năng của model gốc
- ❌ **Không linh hoạt**: Một model cho một tác vụ, khó tái sử dụng

### Khi Nào Sử Dụng

- Khi có đủ tài nguyên tính toán (GPU mạnh)
- Khi có tập dữ liệu lớn và chất lượng cao
- Khi cần hiệu suất tối đa cho một tác vụ cụ thể
- Khi domain adaptation là ưu tiên hàng đầu

### Ví Dụ Use Case

```python
# Ví dụ: Fine-tuning cho tác vụ sửa lỗi tên sheet BIM
# Input: "Sheet-01-Architectural-Plan"
# Output: "Sheet-01-Architectural-Plan-Corrected"
```

## 2. Direct Preference Optimization (DFO)

### Định Nghĩa

**Direct Preference Optimization (DFO)** là phương pháp tối ưu hóa trực tiếp các tham số của model để tăng xác suất của phản hồi được ưu tiên, không cần model phần thưởng riêng biệt.

### Cách Hoạt Động

1. Thu thập các cặp phản hồi: một phản hồi được ưu tiên, một phản hồi kém ưu tiên hơn
2. Thay vì huấn luyện một model phần thưởng riêng biệt (như trong RLHF), DFO tối ưu hóa trực tiếp các tham số của model
3. Model học để tăng xác suất của phản hồi được ưu tiên và giảm xác suất của phản hồi kém ưu tiên

### Ưu Điểm

- ✅ **Đơn giản hơn RLHF**: Không cần model phần thưởng riêng biệt
- ✅ **Phù hợp cho sở thích chủ quan**: Điều chỉnh giọng điệu, phong cách, an toàn
- ✅ **Huấn luyện ổn định**: Ít biến động hơn so với RLHF
- ✅ **Hiệu quả**: Tối ưu hóa trực tiếp, không qua bước trung gian

### Nhược Điểm

- ❌ **Cần dữ liệu có nhãn ưu tiên**: Phải có các cặp phản hồi được đánh giá
- ❌ **Không cải thiện kiến thức thực tế**: Không tốt như SFT cho domain adaptation
- ❌ **Phụ thuộc vào chất lượng dữ liệu ưu tiên**: Kết quả phụ thuộc vào cách đánh giá ưu tiên
- ❌ **Khó thu thập dữ liệu**: Cần con người đánh giá và so sánh phản hồi

### Khi Nào Sử Dụng

- Khi cần điều chỉnh theo sở thích chủ quan (giọng điệu, phong cách)
- Khi muốn cải thiện an toàn và hữu ích của model
- Khi có dữ liệu ưu tiên từ người dùng hoặc chuyên gia
- Khi muốn tránh độ phức tạp của RLHF

### Ví Dụ Use Case

```python
# Ví dụ: Cải thiện phản hồi lịch sự
# Preferred: "Tôi hiểu bạn đang gặp khó khăn. Hãy để tôi giúp bạn."
# Rejected: "Đó là vấn đề của bạn, tự giải quyết đi."
```

## 3. Parameter-Efficient Fine-Tuning (PEFT)

### Định Nghĩa

**Parameter-Efficient Fine-Tuning (PEFT)** là phương pháp chỉ huấn luyện một phần nhỏ các tham số (thường <1%) trong khi giữ nguyên model gốc.

### Cách Hoạt Động

1. Giữ nguyên model LLM gốc (frozen)
2. Thêm các tham số nhẹ (adapter layers) vào model
3. Chỉ huấn luyện các adapter này (thường <1% trọng số của model)
4. Có thể hoán đổi nhiều adapter cho các domain/tác vụ khác nhau

### Ưu Điểm

- ✅ **Tiết kiệm tài nguyên**: Chỉ huấn luyện một phần nhỏ của model
- ✅ **Linh hoạt**: Dễ dàng chuyển đổi giữa các tác vụ hoặc domain khác nhau
- ✅ **Giảm Catastrophic Forgetting**: Model gốc không bị thay đổi
- ✅ **Yêu cầu phần cứng thấp**: Có thể chạy trên GPU consumer-grade
- ✅ **Nhiều adapter**: Có thể có nhiều adapter cho nhiều tác vụ

### Nhược Điểm

- ❌ **Hiệu suất có thể thấp hơn SFT**: Trên các tác vụ phức tạp
- ❌ **Cần quản lý nhiều adapter**: Phức tạp hơn khi có nhiều tác vụ
- ❌ **Phụ thuộc vào kiến trúc**: Một số model có thể không hỗ trợ tốt
- ❌ **Cần chọn target modules**: Phải biết module nào để áp dụng adapter

### Khi Nào Sử Dụng

- Khi tài nguyên tính toán hạn chế
- Khi cần nhiều adapter cho nhiều tác vụ khác nhau
- Khi muốn giữ nguyên model gốc
- Khi model quá lớn (>7B parameters)
- Khi cần prototype và thử nghiệm nhanh

### Ví Dụ Use Case

```python
# Ví dụ: Nhiều adapter cho nhiều domain
# adapter_medical.npz - cho tác vụ y tế
# adapter_legal.npz - cho tác vụ pháp lý
# adapter_finance.npz - cho tác vụ tài chính
```

## 4. Group Relative Policy Optimization (GRPO)

### Định Nghĩa

**Group Relative Policy Optimization (GRPO)** là phương pháp so sánh các nhóm phản hồi để xác định phản hồi tốt nhất và tối ưu hóa chính sách của model dựa trên các so sánh này.

### Cách Hoạt Động

1. Thu thập các nhóm phản hồi (nhiều hơn 2 phản hồi)
2. So sánh các nhóm phản hồi để xác định phản hồi tốt nhất
3. Tối ưu hóa chính sách của model dựa trên các so sánh nhóm này
4. Model học từ việc so sánh tương đối giữa các nhóm

### Ưu Điểm

- ✅ **Cải thiện khả năng suy luận**: Tốt cho các tác vụ yêu cầu đánh giá tương đối
- ✅ **Không cần model phần thưởng**: Tối ưu hóa trực tiếp như DFO
- ✅ **Phù hợp cho đánh giá nhóm**: Khi có nhiều lựa chọn để so sánh
- ✅ **Cải thiện ra quyết định**: Model học cách đánh giá và chọn lựa tốt hơn

### Nhược Điểm

- ❌ **Cần dữ liệu nhóm phản hồi**: Phức tạp hơn trong việc thu thập dữ liệu
- ❌ **Thiết kế phức tạp**: Phức tạp hơn DFO trong việc thiết kế và triển khai
- ❌ **Yêu cầu nhiều phản hồi**: Cần nhiều hơn 2 phản hồi để so sánh
- ❌ **Ít tài liệu và ví dụ**: Phương pháp mới hơn, ít tài liệu hơn

### Khi Nào Sử Dụng

- Khi cần cải thiện khả năng suy luận và ra quyết định
- Khi có nhiều lựa chọn phản hồi để so sánh
- Khi tác vụ yêu cầu đánh giá tương đối giữa các phản hồi
- Khi muốn model học cách chọn lựa tốt nhất từ nhiều phương án

### Ví Dụ Use Case

```python
# Ví dụ: So sánh nhiều phương án giải quyết vấn đề
# Group 1: [Solution A, Solution B, Solution C]
# Model học để xác định Solution nào tốt nhất dựa trên context
```

## Bảng So Sánh Tổng Quan

| Tiêu Chí | SFT | DFO | PEFT | GRPO |
|----------|-----|-----|------|------|
| **Số tham số trainable** | 100% | 100% | <1% | 100% |
| **Yêu cầu tài nguyên** | Rất cao | Cao | Thấp | Cao |
| **Yêu cầu dữ liệu** | Rất nhiều (10k+) | Cặp ưu tiên | Vừa phải | Nhóm phản hồi |
| **Hiệu suất** | Tốt nhất | Tốt (sở thích) | Gần SFT | Tốt (suy luận) |
| **Linh hoạt** | Thấp | Thấp | Rất cao | Thấp |
| **Catastrophic Forgetting** | Cao | Cao | Thấp | Cao |
| **Độ phức tạp** | Trung bình | Trung bình | Thấp | Cao |
| **Tốc độ huấn luyện** | Chậm | Chậm | Nhanh | Chậm |
| **Chi phí** | Rất cao | Cao | Thấp | Cao |

## Ma Trận Quyết Định

### Chọn SFT Nếu:

- ✅ Có đủ tài nguyên GPU mạnh
- ✅ Có tập dữ liệu lớn (>10k mẫu)
- ✅ Cần hiệu suất tối đa cho một tác vụ cụ thể
- ✅ Domain adaptation là ưu tiên
- ✅ Không quan tâm đến catastrophic forgetting

### Chọn DFO Nếu:

- ✅ Cần điều chỉnh theo sở thích chủ quan (giọng điệu, phong cách)
- ✅ Có dữ liệu ưu tiên từ người dùng
- ✅ Muốn cải thiện an toàn và hữu ích
- ✅ Muốn tránh độ phức tạp của RLHF
- ✅ Không cần cải thiện kiến thức thực tế nhiều

### Chọn PEFT Nếu:

- ✅ Tài nguyên tính toán hạn chế
- ✅ Cần nhiều adapter cho nhiều tác vụ
- ✅ Muốn giữ nguyên model gốc
- ✅ Model quá lớn (>7B parameters)
- ✅ Cần prototype và thử nghiệm nhanh

### Chọn GRPO Nếu:

- ✅ Cần cải thiện khả năng suy luận và ra quyết định
- ✅ Có nhiều lựa chọn phản hồi để so sánh
- ✅ Tác vụ yêu cầu đánh giá tương đối
- ✅ Muốn model học cách chọn lựa tốt nhất
- ✅ Có đủ tài nguyên và dữ liệu nhóm

## Kết Hợp Các Phương Pháp

Trong thực tế, bạn có thể kết hợp nhiều phương pháp:

### Workflow Đề Xuất

1. **Bắt đầu với PEFT (LoRA)**:
   - Thử nghiệm nhanh và rẻ
   - Đánh giá hiệu suất ban đầu
   - Nếu đạt yêu cầu → dừng lại

2. **Nâng cấp lên SFT nếu cần**:
   - Khi PEFT không đạt hiệu suất mong muốn
   - Khi có đủ tài nguyên và dữ liệu

3. **Áp dụng DFO sau SFT**:
   - Điều chỉnh sở thích và phong cách
   - Cải thiện an toàn và hữu ích

4. **Sử dụng GRPO cho tác vụ đặc biệt**:
   - Khi cần cải thiện suy luận
   - Khi có nhiều phương án để so sánh

### Ví Dụ Kết Hợp

```python
# 1. Fine-tune với PEFT (LoRA)
lora_model = apply_lora(base_model, lora_config)
train_lora(lora_model, task_data)

# 2. Nếu cần, nâng cấp lên SFT
sft_model = fine_tune_full(base_model, large_dataset)

# 3. Điều chỉnh sở thích với DFO
dpo_model = apply_dpo(sft_model, preference_pairs)

# 4. Cải thiện suy luận với GRPO (nếu cần)
grpo_model = apply_grpo(dpo_model, response_groups)
```

## Best Practices

### Cho SFT:
- Sử dụng learning rate thấp (1e-5 đến 5e-5)
- Sử dụng gradient checkpointing để tiết kiệm bộ nhớ
- Monitor cả training và validation loss
- Sử dụng early stopping để tránh overfitting
- Backup model gốc trước khi fine-tune

### Cho DFO:
- Đảm bảo chất lượng dữ liệu ưu tiên
- Sử dụng nhiều người đánh giá để tránh bias
- Cân bằng giữa các loại ưu tiên khác nhau
- Validate với người dùng thực tế

### Cho PEFT:
- Bắt đầu với `r=16` hoặc `r=32` cho LoRA
- Thử nghiệm với các target modules khác nhau
- Sử dụng QLoRA nếu model quá lớn
- Lưu trữ và quản lý nhiều adapter một cách có tổ chức

### Cho GRPO:
- Thiết kế cẩn thận cách nhóm phản hồi
- Đảm bảo tính nhất quán trong đánh giá nhóm
- Validate với nhiều nhóm phản hồi khác nhau
- Monitor sự cải thiện trong khả năng suy luận

## Kết Luận

Việc lựa chọn phương pháp fine-tuning phù hợp phụ thuộc vào:

1. **Tài nguyên có sẵn**: GPU, bộ nhớ, thời gian
2. **Yêu cầu hiệu suất**: Cần hiệu suất tối đa hay chấp nhận trade-off
3. **Loại dữ liệu**: Có dữ liệu gì (labeled, preference pairs, groups)
4. **Mục tiêu**: Domain adaptation, sở thích, suy luận, hay đa tác vụ
5. **Độ phức tạp**: Sẵn sàng chấp nhận độ phức tạp đến mức nào

**Khuyến nghị chung**:
- Bắt đầu với **PEFT** để thử nghiệm nhanh và rẻ
- Nâng cấp lên **SFT** nếu cần hiệu suất cao hơn
- Áp dụng **DFO** để điều chỉnh sở thích và phong cách
- Sử dụng **GRPO** cho các tác vụ đặc biệt yêu cầu suy luận

Nhớ rằng không có phương pháp nào là "tốt nhất" cho mọi trường hợp. Việc lựa chọn phụ thuộc vào tình huống cụ thể của bạn.

## Tài Liệu Tham Khảo

- [SFT vs DFO vs PEFT vs GRPO: Choosing the Right Fine-Tuning Strategy for LLMs](https://bobrupakroy.medium.com/sft-vs-dfo-vs-peft-vs-grpo-choosing-the-right-fine-tuning-strategy-for-llms-3a7671c9347a)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)

