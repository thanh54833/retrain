# Các Phương Pháp LoRA

## Giới Thiệu

Một cách phổ biến để huấn luyện hiệu quả các model lớn là chèn (thường là trong các khối attention) các ma trận có thể huấn luyện nhỏ hơn, đây là phân rã hạng thấp (low-rank decomposition) của ma trận trọng số delta cần học trong quá trình fine-tuning. Ma trận trọng số gốc của model pretrained được đóng băng và chỉ các ma trận nhỏ hơn được cập nhật trong quá trình huấn luyện. Điều này giảm số lượng tham số có thể huấn luyện, giảm sử dụng bộ nhớ và thời gian huấn luyện, điều này có thể rất tốn kém đối với các model lớn.

Có nhiều cách khác nhau để biểu diễn ma trận trọng số dưới dạng phân rã hạng thấp, nhưng [Low-Rank Adaptation (LoRA)](../conceptual_guides/adapter#low-rank-adaptation-lora) là phương pháp phổ biến nhất. Thư viện PEFT hỗ trợ một số biến thể LoRA khác, chẳng hạn như [Low-Rank Hadamard Product (LoHa)](../conceptual_guides/adapter#low-rank-hadamard-product-loha), [Low-Rank Kronecker Product (LoKr)](../conceptual_guides/adapter#low-rank-kronecker-product-lokr), và [Adaptive Low-Rank Adaptation (AdaLoRA)](../conceptual_guides/adapter#adaptive-low-rank-adaptation-adalora).

Ngoài ra, PEFT hỗ trợ phương pháp [X-LoRA](../conceptual_guides/adapter#mixture-of-lora-experts-x-lora) Mixture of LoRA Experts.

Hướng dẫn này sẽ chỉ cho bạn cách nhanh chóng huấn luyện một model phân loại hình ảnh - với phương pháp phân rã hạng thấp - để xác định loại thực phẩm được hiển thị trong hình ảnh.

> [!TIP]
> Một số quen thuộc với quy trình chung của việc huấn luyện model phân loại hình ảnh sẽ rất hữu ích và cho phép bạn tập trung vào các phương pháp phân rã hạng thấp. Nếu bạn mới bắt đầu, chúng tôi khuyên bạn nên xem hướng dẫn [Image classification](https://huggingface.co/docs/transformers/tasks/image_classification) trước từ tài liệu Transformers. Khi bạn đã sẵn sàng, hãy quay lại và xem cách dễ dàng để tích hợp PEFT vào quá trình huấn luyện của bạn!

Trước khi bắt đầu, hãy đảm bảo bạn đã cài đặt tất cả các thư viện cần thiết.

```bash
pip install -q peft transformers datasets
```

## Dataset

Trong hướng dẫn này, bạn sẽ sử dụng dataset [Food-101](https://huggingface.co/datasets/food101) chứa hình ảnh của 101 loại thực phẩm (hãy xem [dataset viewer](https://huggingface.co/datasets/food101/viewer/default/train) để hiểu rõ hơn về dataset).

Tải dataset bằng hàm [load_dataset](https://huggingface.co/docs/datasets/v4.4.1/en/package_reference/loading_methods#datasets.load_dataset).

```python
from datasets import load_dataset

ds = load_dataset("food101")
```

Mỗi loại thực phẩm được gắn nhãn bằng một số nguyên, vì vậy để dễ hiểu hơn về những số nguyên này đại diện cho gì, bạn sẽ tạo từ điển `label2id` và `id2label` để ánh xạ số nguyên với nhãn lớp của nó.

```python
labels = ds["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

id2label[2]
# "baklava"
```

Tải image processor để resize và chuẩn hóa đúng các giá trị pixel của hình ảnh huấn luyện và đánh giá.

```python
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
```

Bạn cũng có thể sử dụng image processor để chuẩn bị một số hàm chuyển đổi cho data augmentation và scaling pixel.

```python
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

def preprocess_train(example_batch):
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch
```

Định nghĩa dataset huấn luyện và validation, và sử dụng hàm [set_transform](https://huggingface.co/docs/datasets/v4.4.1/en/package_reference/main_classes#datasets.Dataset.set_transform) để áp dụng các phép biến đổi on-the-fly.

```python
train_ds = ds["train"]
val_ds = ds["validation"]

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)
```

Cuối cùng, bạn sẽ cần một data collator để tạo batch dữ liệu huấn luyện và đánh giá, và chuyển đổi nhãn thành các đối tượng `torch.tensor`.

```python
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
```

## Model

Bây giờ hãy tải một model pretrained để sử dụng làm model gốc. Hướng dẫn này sử dụng model [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k), nhưng bạn có thể sử dụng bất kỳ model phân loại hình ảnh nào bạn muốn. Truyền các từ điển `label2id` và `id2label` vào model để nó biết cách ánh xạ nhãn số nguyên với nhãn lớp của chúng, và bạn có thể tùy chọn truyền tham số `ignore_mismatched_sizes=True` nếu bạn đang fine-tuning một checkpoint đã được fine-tuning.

```python
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
```

### Cấu Hình PEFT và Model

Mỗi phương pháp PEFT yêu cầu một cấu hình chứa tất cả các tham số chỉ định cách phương pháp PEFT nên được áp dụng. Sau khi cấu hình được thiết lập, truyền nó vào hàm [get_peft_model()](/docs/peft/v0.18.0/en/package_reference/peft_model#peft.get_peft_model) cùng với model gốc để tạo một [PeftModel](/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel) có thể huấn luyện.

> [!TIP]
> Gọi phương thức [print_trainable_parameters()](/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel.print_trainable_parameters) để so sánh số lượng tham số của [PeftModel](/docs/peft/v0.18.0/en/package_reference/peft_model#peft.PeftModel) so với số lượng tham số trong model gốc!

### LoRA (Low-Rank Adaptation)

[LoRA](../conceptual_guides/adapter#low-rank-adaptation-lora) phân rã ma trận cập nhật trọng số thành *hai* ma trận nhỏ hơn. Kích thước của các ma trận hạng thấp này được xác định bởi *rank* hoặc `r` của nó. Rank cao hơn có nghĩa là model có nhiều tham số hơn để huấn luyện, nhưng cũng có nghĩa là model có khả năng học tập cao hơn. Bạn cũng sẽ muốn chỉ định `target_modules` xác định nơi các ma trận nhỏ hơn được chèn vào. Đối với hướng dẫn này, bạn sẽ nhắm vào các ma trận *query* và *value* của các khối attention. Các tham số quan trọng khác cần đặt là `lora_alpha` (hệ số scaling), `bias` (có phải `none`, `all` hay chỉ các tham số bias LoRA nên được huấn luyện), và `modules_to_save` (các module ngoài các lớp LoRA để được huấn luyện và lưu). Tất cả các tham số này - và nhiều hơn nữa - được tìm thấy trong [LoraConfig](/docs/peft/v0.18.0/en/package_reference/lora#peft.LoraConfig).

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# "trainable params: 667,493 || all params: 86,543,818 || trainable%: 0.7712775047664294"
```

### LoHa (Low-Rank Hadamard Product)

[LoHa](../conceptual_guides/adapter#low-rank-hadamard-product-loha) phân rã ma trận cập nhật trọng số thành *bốn* ma trận nhỏ hơn và mỗi cặp ma trận nhỏ hơn được kết hợp với tích Hadamard. Điều này cho phép ma trận cập nhật trọng số giữ nguyên số lượng tham số có thể huấn luyện khi so sánh với LoRA, nhưng với rank cao hơn (`r^2` cho LoHA khi so sánh với `2*r` cho LoRA). Kích thước của các ma trận nhỏ hơn được xác định bởi *rank* hoặc `r` của nó. Bạn cũng sẽ muốn chỉ định `target_modules` xác định nơi các ma trận nhỏ hơn được chèn vào. Đối với hướng dẫn này, bạn sẽ nhắm vào các ma trận *query* và *value* của các khối attention. Các tham số quan trọng khác cần đặt là `alpha` (hệ số scaling), và `modules_to_save` (các module ngoài các lớp LoHa để được huấn luyện và lưu). Tất cả các tham số này - và nhiều hơn nữa - được tìm thấy trong [LoHaConfig](/docs/peft/v0.18.0/en/package_reference/loha#peft.LoHaConfig).

```python
from peft import LoHaConfig, get_peft_model

config = LoHaConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# "trainable params: 1,257,317 || all params: 87,133,642 || trainable%: 1.4429753779831676"
```

### LoKr (Low-Rank Kronecker Product)

[LoKr](../conceptual_guides/adapter#low-rank-kronecker-product-lokr) biểu diễn ma trận cập nhật trọng số dưới dạng phân rã của tích Kronecker, tạo ra một ma trận khối có thể bảo toàn rank của ma trận trọng số gốc. Kích thước của các ma trận nhỏ hơn được xác định bởi *rank* hoặc `r` của nó. Bạn cũng sẽ muốn chỉ định `target_modules` xác định nơi các ma trận nhỏ hơn được chèn vào. Đối với hướng dẫn này, bạn sẽ nhắm vào các ma trận *query* và *value* của các khối attention. Các tham số quan trọng khác cần đặt là `alpha` (hệ số scaling), và `modules_to_save` (các module ngoài các lớp LoKr để được huấn luyện và lưu). Tất cả các tham số này - và nhiều hơn nữa - được tìm thấy trong [LoKrConfig](/docs/peft/v0.18.0/en/package_reference/lokr#peft.LoKrConfig).

```python
from peft import LoKrConfig, get_peft_model

config = LoKrConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# "trainable params: 116,069 || all params: 87,172,042 || trainable%: 0.13314934162033282"
```

### AdaLoRA (Adaptive Low-Rank Adaptation)

[AdaLoRA](../conceptual_guides/adapter#adaptive-low-rank-adaptation-adalora) quản lý hiệu quả ngân sách tham số LoRA bằng cách gán nhiều tham số hơn cho các ma trận trọng số quan trọng và cắt tỉa những ma trận ít quan trọng hơn. Ngược lại, LoRA phân phối đều tham số trên tất cả các module. Bạn có thể kiểm soát *rank* trung bình mong muốn hoặc `r` của các ma trận, và các module nào để áp dụng AdaLoRA với `target_modules`. Các tham số quan trọng khác cần đặt là `lora_alpha` (hệ số scaling), và `modules_to_save` (các module ngoài các lớp AdaLoRA để được huấn luyện và lưu). Tất cả các tham số này - và nhiều hơn nữa - được tìm thấy trong [AdaLoraConfig](/docs/peft/v0.18.0/en/package_reference/adalora#peft.AdaLoraConfig).

```python
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    r=8,
    init_r=12,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# "trainable params: 520,325 || all params: 87,614,722 || trainable%: 0.5938785036606062"
```

## Huấn Luyện

Đối với huấn luyện, hãy sử dụng lớp [Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#transformers.Trainer) từ Transformers. `Trainer` chứa một vòng lặp huấn luyện PyTorch, và khi bạn đã sẵn sàng, gọi [train](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#transformers.Trainer.train) để bắt đầu huấn luyện. Để tùy chỉnh quá trình huấn luyện, cấu hình các siêu tham số huấn luyện trong lớp [TrainingArguments](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#transformers.TrainingArguments). Với các phương pháp giống LoRA, bạn có thể sử dụng batch size và learning rate cao hơn.

> [!WARNING]
> AdaLoRA có phương thức [update_and_allocate()](/docs/peft/v0.18.0/en/package_reference/adalora#peft.AdaLoraModel.update_and_allocate) nên được gọi ở mỗi bước huấn luyện để cập nhật ngân sách tham số và mask, nếu không bước adaptation sẽ không được thực hiện. Điều này yêu cầu viết một vòng lặp huấn luyện tùy chỉnh hoặc subclassing [Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#transformers.Trainer) để tích hợp phương thức này. Ví dụ, hãy xem [custom training loop](https://github.com/huggingface/peft/blob/912ad41e96e03652cabf47522cd876076f7a0c4f/examples/conditional_generation/peft_adalora_seq2seq.py#L120) này.

```python
from transformers import TrainingArguments, Trainer

account = "stevhliu"
peft_model_id = f"{account}/google/vit-base-patch16-224-in21k-lora"
batch_size = 128

args = TrainingArguments(
    peft_model_id,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    label_names=["labels"],
)
```

Bắt đầu huấn luyện với [train](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#transformers.Trainer.train).

```python
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=image_processor,
    data_collator=collate_fn,
)
trainer.train()
```

## Chia Sẻ Model Của Bạn

Sau khi huấn luyện hoàn tất, bạn có thể tải model của mình lên Hub bằng phương thức [push_to_hub](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.push_to_hub). Bạn sẽ cần đăng nhập vào tài khoản Hugging Face của mình trước và nhập token khi được nhắc.

```python
from huggingface_hub import notebook_login

notebook_login()
```

Gọi [push_to_hub](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.push_to_hub) để lưu model của bạn vào repository.

```python
model.push_to_hub(peft_model_id)
```

## Inference

Hãy tải model từ Hub và kiểm tra nó trên một hình ảnh thực phẩm.

```python
from peft import PeftConfig, PeftModel
from transformers import AutoImageProcessor
from PIL import Image
import requests

config = PeftConfig.from_pretrained("stevhliu/vit-base-patch16-224-in21k-lora")
model = AutoModelForImageClassification.from_pretrained(
    config.base_model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
model = PeftModel.from_pretrained(model, "stevhliu/vit-base-patch16-224-in21k-lora")

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
image
```

Chuyển đổi hình ảnh sang RGB và trả về các tensor PyTorch cơ bản.

```python
encoding = image_processor(image.convert("RGB"), return_tensors="pt")
```

Bây giờ chạy model và trả về lớp dự đoán!

```python
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
# "Predicted class: beignets"
```

## So Sánh Các Phương Pháp LoRA

### Bảng So Sánh

| Phương Pháp | Số Ma Trận | Rank Hiệu Quả | Tham Số Trainable | Độ Phức Tạp |
|------------|-----------|---------------|-------------------|-------------|
| **LoRA** | 2 | r | Thấp nhất | Đơn giản nhất |
| **LoHa** | 4 | r² | Trung bình | Trung bình |
| **LoKr** | 2 (Kronecker) | r | Thấp nhất | Phức tạp |
| **AdaLoRA** | 2 (Adaptive) | r (dynamic) | Trung bình | Phức tạp nhất |

### Khi Nào Sử Dụng Phương Pháp Nào?

#### LoRA
- **Khi nào**: Hầu hết các trường hợp sử dụng, đặc biệt là khi bắt đầu
- **Ưu điểm**: Đơn giản, ổn định, được hỗ trợ rộng rãi
- **Nhược điểm**: Phân phối tham số đều, không tối ưu cho tất cả các lớp

#### LoHa
- **Khi nào**: Khi cần rank cao hơn với cùng số lượng tham số
- **Ưu điểm**: Rank hiệu quả cao hơn (r² so với 2r)
- **Nhược điểm**: Phức tạp hơn LoRA, có thể không cần thiết cho nhiều tác vụ

#### LoKr
- **Khi nào**: Khi cần bảo toàn rank của ma trận gốc
- **Ưu điểm**: Ít tham số nhất, bảo toàn cấu trúc
- **Nhược điểm**: Phức tạp hơn, ít được sử dụng

#### AdaLoRA
- **Khi nào**: Khi có ngân sách tham số cố định và muốn tối ưu phân phối
- **Ưu điểm**: Phân phối tham số thông minh, hiệu quả cao
- **Nhược điểm**: Phức tạp nhất, yêu cầu vòng lặp huấn luyện tùy chỉnh

## Ví Dụ Hoàn Chỉnh

Dưới đây là một ví dụ hoàn chỉnh sử dụng LoRA cho image classification:

```python
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import torch

# 1. Tải dataset
ds = load_dataset("food101")

# 2. Tạo label mappings
labels = ds["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

# 3. Tải image processor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# 4. Chuẩn bị transforms
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

def preprocess_train(example_batch):
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_val(example_batch):
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# 5. Áp dụng transforms
train_ds = ds["train"]
val_ds = ds["validation"]
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# 6. Data collator
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# 7. Tải model gốc
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# 8. Cấu hình LoRA
config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)

# 9. Áp dụng LoRA
model = get_peft_model(model, config)
model.print_trainable_parameters()

# 10. Cấu hình training
args = TrainingArguments(
    "vit-lora-food101",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=128,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=128,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    label_names=["labels"],
)

# 11. Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=image_processor,
    data_collator=collate_fn,
)

# 12. Huấn luyện
trainer.train()

# 13. Lưu model
model.save_pretrained("vit-lora-food101")
```

## Tài Liệu Tham Khảo

- [Hugging Face PEFT LoRA-based Methods Guide](https://huggingface.co/docs/peft/en/task_guides/lora_based_methods)
- [LoRA Paper: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [LoHa Paper: Low-Rank Hadamard Product](https://arxiv.org/abs/2308.11695)
- [LoKr Paper: Low-Rank Kronecker Product](https://arxiv.org/abs/2309.14859)
- [AdaLoRA Paper: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)

