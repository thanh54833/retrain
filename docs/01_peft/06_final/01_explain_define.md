

## Overfitting

Với 100 samples, model có thể học thuộc lòng (memorize) thay vì học pattern. Điều này xảy ra khi:

- **Dataset quá nhỏ**: Với chỉ 100 mẫu, model có đủ capacity để ghi nhớ từng mẫu cụ thể thay vì học các pattern tổng quát
- **Không có khả năng tổng quát hóa**: Model có thể đạt accuracy cao trên training set nhưng performance kém trên validation/test set
- **Dấu hiệu nhận biết**: 
  - Training loss giảm nhanh và về gần 0
  - Validation loss không giảm hoặc tăng dần
  - Model không thể xử lý các input tương tự nhưng khác một chút so với training data

**Giải pháp**:
- Tăng số lượng samples trong dataset
- Sử dụng data augmentation
- Áp dụng regularization techniques (dropout, weight decay)
- Early stopping dựa trên validation loss
- Sử dụng cross-validation để đánh giá tốt hơn