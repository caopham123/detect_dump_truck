# KẾ HOẠCH TRIỂN KHAI GIAI ĐOẠN PROOF OF CONCEPT (PoC)
**Mục tiêu:** Chứng minh tính khả thi của giải pháp AI dùng YOLO-Pose để đo góc nghiêng xe ben. Viết chương trình chạy được nghiệm thu trên video thực tế.
**Thời gian dự kiến:** 2 - 3 tuần.

---

## BƯỚC 1: THU THẬP DỮ LIỆU (DATA COLLECTION) - (3-4 Ngày)
Mục tiêu: Có được tập dữ liệu thô phản ánh đúng thực tế lắp đặt.
1. **Thiết lập góc quay giả lập:** 
   - Dùng điện thoại gắn tripod hoặc flycam quay mô phỏng đúng góc độ: Góc ngang sườn/chéo **40°-60°**, độ cao **3m-4.5m**.
2. **Kịch bản quay video:**
   - Xe chạy qua với thùng hạ an toàn (nhiều loại xe khác nhau: rơ-moóc, 3 chân, 4 chân).
   - Xe di chuyển chầm chậm nhưng thùng **đang nâng** (kịch bản vi phạm).
   - Quay ở các điều kiện ánh sáng khác nhau nếu có thể (Sáng/Chiều).
3. **Trích xuất ảnh (Frame Extraction):**
   - Viết script Python/FFmpeg cắt video thành ảnh tĩnh (khoảng 3-5 FPS). Lọc bỏ các ảnh rác không có xe.
   - **Target:** Thu thập khoảng **500 - 800 ảnh** chất lượng để làm tập PoC.

## BƯỚC 2: GÁN NHÃN DỮ LIỆU (DATA ANNOTATION) - (4-5 Ngày)
Mục tiêu: Đóng gói dữ liệu định dạng chuẩn của YOLO-Pose.
1. **Lựa chọn Tool:** Sử dụng **CVAT** (local) hoặc **Roboflow** (trực tuyến, khuyên dùng cho PoC vì hỗ trợ export nhanh).
2. **Định nghĩa Lớp (Class & Skeleton):**
   - Bounding Box Class: `dump_truck`.
   - Keypoints (3 điểm):
     * V0: `Cabin` (Điểm dưới cùng của đầu ca-bin).
     * V1: `Hinge` (Trục bản lề/điểm xoay thùng).
     * V2: `Tail` (Mép dưới đuôi thùng).
3. **Thực thi gán nhãn:**
   - Vẽ box bao quanh xe. Cắm 3 điểm tọạ độ. 
   - *Lưu ý quan trọng:* Đối với điểm `Hinge` bị bánh xe che lấp, người gán nhãn cần tự ước lượng vị trí cơ học để cắm điểm (YOLOv8-Pose học được tính đối xứng ẩn này).
4. **Export:** Xuất dữ liệu sang định dạng `YOLOv8 Pose`. Phân chia Train/Val theo tỷ lệ 80/20.

## BƯỚC 3: HUẤN LUYỆN MÔ HÌNH (MODEL TRAINING) - (2 Ngày)
Mục tiêu: Train ra trọng số (weights) `best.pt`.
1. **Môi trường:** Thuê GPU trên Google Colab Pro hoặc dùng máy tính có sẵn GPU rời (RTX 3060 trở lên).
2. **Cấu hình YOLOv8:**
   - Tải pretrained model: `yolov8n-pose.pt` (bản Nano, tối ưu tốc độ) hoặc `yolov8s-pose.pt`.
   - Sửa file YAML khai báo số lượng keypoints = 3.
3. **Training:**
   - Train với epochs ~ 100-150. Quan sát biểu đồ loss của box và pose để tránh Overfitting.

## BƯỚC 4: VIẾT CHƯƠNG TRÌNH DEMO (DEMO APPLICATION) - (4 Ngày)
Mục tiêu: Viết script Python đọc Video $\rightarrow$ Detect $\rightarrow$ Tính góc $\rightarrow$ Hiển thị Alert.
1. **Luồng xử lý (Inference):**
   - Load video bằng `cv2.VideoCapture()`.
   - Đẩy từng frame qua mô hình `model.predict()`.
2. **Thuật toán cốt lõi (Angle Calculation):**
   - Lấy toạ độ $(x, y)$ của 3 điểm trả về từ YOLO.
   - Viết hàm toán học tính góc tạo bởi 2 vector: $V_{hinge \rightarrow tail}$ và $V_{hinge \rightarrow cabin}$.
3. **Logic Kích hoạt:**
   - Định nghĩa trước một ngưỡng `DUMP_ANGLE_THRESHOLD = 15.0` (độ).
   - Nếu Góc tính toán > Ngưỡng $\Rightarrow$ đổi màu Bounding box sang ĐỎ, in chữ "CẢNH BÁO: CHƯA HẠ THÙNG".
4. **Vẽ UI (Visualization):**
   - Dùng OpenCV vẽ đường thẳng nối 3 điểm (Skeleton lines).
   - Ghi thông số đo góc bằng text lên đầu khung hình.

## BƯỚC 5: KIỂM THỬ VÀ NGHIỆM THU (EVALUATION) - (2 Ngày)
Mục tiêu: Báo cáo kết quả PoC.
1. **Chạy Test:** Chạy file Python trên các đoạn video **chưa từng được train**.
2. **Xuất Video Demo:** Lưu kết quả dạng video `.mp4` để chuẩn bị trình bày.
3. **Đánh giá KPI:**
   - Tốc độ xử lý (FPS) đạt bao nhiêu?
   - Tính toán tỷ lệ False Positive (báo sai) trên video.
   - MAE (Độ lệch góc): Bằng mắt thường so với góc hiển thị.
4. **Tài liệu Báo cáo:** Chốt kiến trúc và lập dự toán phần cứng cho giai đoạn Pilot.
