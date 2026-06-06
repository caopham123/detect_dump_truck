# Hướng dẫn Đánh giá Mô hình YOLO sau Huấn luyện

Tài liệu này hướng dẫn cách đọc và phân tích các biểu đồ, số liệu được sinh ra trong thư mục `runs/detect/` hoặc `runs/pose/` sau khi huấn luyện mô hình YOLO bằng thư viện Ultralytics.

---

## 1. Phân tích sự hội tụ (File `results.png` & `results.csv`)

File **`results.png`** cung cấp cái nhìn tổng quan nhất về toàn bộ quá trình training thông qua các biểu đồ Loss (mất mát) và Metrics (độ đo) qua từng epoch.

### Tiêu chí đánh giá

* **Đường Loss (Train & Val):** Nhìn vào các biểu đồ như `box_loss`, `cls_loss`, `dfl_loss` (đối với detect) hoặc `pose_loss` (đối với pose).
  * **Tốt:** Cả hai đường Train loss và Val loss đều giảm dần đều và đi vào ổn định (nằm ngang) ở những epoch cuối.
  * **Overfitting (Quá khớp):** Train loss tiếp tục giảm mạnh, nhưng Val loss lại có dấu hiệu chững lại và **tăng vọt lên**. Nghĩa là model đang "học vẹt" tập train và dự đoán sai trên tập validation.
  * **Underfitting (Chưa khớp):** Loss giảm rất chậm hoặc chưa đạt đến độ ổn định, loss còn rất cao khi kết thúc training.
* **Đường Metrics (mAP, Precision, Recall):** Các đường này phải có xu hướng tăng đều đặn và hội tụ ở một mức cao nhất định ở các epoch cuối.

---

## 2. Đánh giá chất lượng bằng Metrics

Các file hình ảnh như `BoxPR_curve.png`, `BoxF1_curve.png`, `BoxP_curve.png`, `BoxR_curve.png` thể hiện hiệu suất phân loại và nhận diện.

### Các khái niệm cốt lõi

1. **mAP50 (Mean Average Precision @ IoU=0.5):**
    * *Ý nghĩa:* Độ chính xác trung bình khi Bounding Box dự đoán trùng khớp với thực tế ít nhất 50%.
    * *Đánh giá:* Chỉ số quan trọng nhất cho biết model có nhận diện được sự tồn tại của vật thể không. Càng cao (gần 1.0) càng tốt.
2. **mAP50-95:**
    * *Ý nghĩa:* Trung bình của mAP với các ngưỡng IoU tăng dần từ 0.5 đến 0.95.
    * *Đánh giá:* Khắt khe hơn mAP50. Đo lường xem Bounding Box hay Keypoint của model bám có sát và chuẩn xác với vật thể hay không.
3. **Precision (Độ chuẩn xác) - File `BoxP_curve.png`:**
    * *Ý nghĩa:* Trong số các vật thể model nhận diện ra, có bao nhiêu % thực sự là đúng.
    * *Dấu hiệu xấu:* Nếu Precision thấp, hệ thống đang bắt sai nhiều (False Positive - Nhầm cảnh nền thành xe tải).
4. **Recall (Độ phủ) - File `BoxR_curve.png`:**
    * *Ý nghĩa:* Trong tổng số vật thể thực tế tồn tại trong ảnh, model tìm được bao nhiêu %.
    * *Dấu hiệu xấu:* Nếu Recall thấp, hệ thống đang bỏ sót nhiều (False Negative - Có xe tải nhưng model không thấy).
5. **F1-Score - File `BoxF1_curve.png`:**
    * *Ý nghĩa:* Điểm số cân bằng giữa Precision và Recall.
    * *Mẹo:* Bạn có thể dựa vào đỉnh của đường cong F1 để chọn ngưỡng `confidence score` tối ưu khi lập trình ứng dụng thực tế.

---

## 3. Phân tích chi tiết lỗi qua Ma trận nhầm lẫn

File **`confusion_matrix.png`** (và bản normalized) chỉ ra rõ ràng việc model đang bối rối ở điểm nào.

### Cách đọc

* **Đường chéo chính (Tốt):** Các ô chạy chéo từ góc trên-trái xuống dưới-phải thể hiện số lượng dự đoán đúng. Các ô này cần có màu đậm nhất.
* **Lỗi phân loại sai Class:** Các ô nằm ngoài đường chéo chính. (VD: Xe rác bị nhận diện nhầm thành Xe bồn).
* **Cột Background (Lỗi Precision):** Mô hình khoanh vùng sai một khu vực cảnh nền và cho rằng đó là vật thể. (Nhận báo động giả).
* **Hàng Background (Lỗi Recall):** Vật thể sờ sờ trong ảnh nhưng mô hình lại coi đó là cảnh nền (bỏ qua vật thể).

---

## 4. Kiểm tra trực quan bằng mắt thường (Visual Inspection)

Số liệu chỉ mang tính thống kê. Đừng bỏ qua việc mở các file ảnh để quan sát:

* **`train_batch*.jpg`:** Kiểm tra xem dữ liệu đưa vào train đã được xử lý đúng chưa (kiểm tra Data Augmentation như mosaic, lật ảnh xem có làm méo mó vật thể không).
* **`val_batch*_labels.jpg` (Thực tế) vs `val_batch*_pred.jpg` (Dự đoán):**
  * So sánh 2 ảnh này xem model dự đoán có sát thực tế không.
  * Đặc biệt chú ý đến các trường hợp khó (góc khuất, xe bị che chắn một phần, xe ở xa).

---

## 5. Chọn Trọng số (Weights) để triển khai

Trong thư mục `weights/`, bạn sẽ thấy:

* **`last.pt`:** Trọng số của Epoch cuối cùng chạy xong. Dùng khi bạn bị rớt mạng/mất điện và muốn `resume` training từ đoạn đang dang dở.
* **`best.pt`:** Trọng số của Epoch đạt kết quả tốt nhất (có điểm Fitness/mAP cao nhất) trên tập Validation. **Luôn sử dụng file này** để đem đi Test hoặc chạy ứng dụng thực tế (Inference).

---

## 6. Hướng xử lý sau đánh giá

| Hiện trạng Model | Vấn đề tiềm ẩn | Giải pháp khắc phục |
| :--- | :--- | :--- |
| **Model đạt kết quả cao (mAP > 0.8)** | Hoạt động tốt trên tập Validation. | Đem file `best.pt` đi Test trên Video thực tế chưa từng xuất hiện để đánh giá độ tin cậy. |
| **Bị Overfitting (Val Loss tăng vọt)** | Model quá to, train quá lâu, hoặc ít dữ liệu. | Giảm số lượng Epoch; Dùng kiến trúc YOLO nhỏ hơn (ví dụ dùng YOLOv8n thay vì YOLOv8m); Bổ sung thêm data. |
| **Recall thấp (Hay bỏ sót xe)** | Thiếu mẫu xe ở xa, xe bị che khuất. | Thêm hình ảnh xe bị che hoặc các góc độ lạ vào tập Train. Xem xét hạ ngưỡng `conf` khi inference. |
| **Precision thấp (Bắt sai bóng nắng, rác...)** | Nhầm lẫn background thành xe. | Thêm ảnh "âm tính" (ảnh không chứa chiếc xe tải nào) vào tập Train để model học cách phân biệt cảnh nền. |
| **Bounding box/Keypoint lệch** | Gán nhãn sai, ảnh đầu vào quá mờ/nhỏ. | Kiểm tra lại tool Label. Thử tăng tham số `imgsz` khi train (VD: 640 lên 1280) để tăng độ phân giải ảnh nạp vào. |
