# TÀI LIỆU PHÂN TÍCH YÊU CẦU & GIẢI PHÁP KỸ THUẬT (SPEC)
**Dự án:** Hệ thống AI phát hiện xe ben, xe tải chưa hạ hết thùng trong quá trình di chuyển.
**Phiên bản:** 2.0 (Hợp nhất ưu điểm thuật toán định lượng Pose Estimation và Kiến trúc Edge AI)

---

## 1. MỤC TIÊU BÀI TOÁN (OBJECTIVE)
Phát hiện và cảnh báo tự động, thời gian thực (độ trễ gần như bằng 0) các trường hợp xe tải ben, xe đầu kéo đang di chuyển rời khỏi hiện trường nhưng thùng xe (bed/box) chưa được hạ xuống vị trí an toàn.
Mục đích cốt lõi là ngăn chặn tai nạn đứt cáp điện, sập cầu vượt hoặc lật xe.

## 2. PHÂN TÍCH YÊU CẦU (REQUIREMENTS)

### 2.1. Yêu cầu chức năng
1. **Phát hiện đối tượng:** Nhận diện được toàn bộ chiếc xe và các điểm khớp nối quan trọng (Keypoints).
2. **Tính toán góc nghiêng:** Phải đo lường được định lượng góc nâng của thùng xe (Dump Angle) tính bằng độ.
3. **Quản lý khu vực (ROI):** Chỉ phân tích và cảnh báo khi xe di chuyển vào khu vực lối ra/vào (cổng trạm). Không cảnh báo khi xe đang nằm ở bãi đỗ hoặc khu vực đổ vật liệu.
4. **Cảnh báo tức thời:** Gửi tín hiệu điều khiển trực tiếp tới Barrier (rào chắn) hoặc còi/đèn báo động.

### 2.2. Yêu cầu phi chức năng
1. **Độ trễ (Latency):** < 500ms kể từ khi phát hiện vi phạm để kịp thời đóng rào chắn. Tối thiểu 15-25 FPS.
2. **Độ chính xác:** Cực kỳ khắt khe với False Negative (bỏ lọt). Có cơ chế loại bỏ nhiễu/kẹt điểm (Occlusion) khi góc quay bị che lấp.
3. **Tính bền bỉ:** Hoạt động tốt ban đêm (IR Camera) và môi trường khói bụi công trường.

---

## 3. THUẬT TOÁN & GIẢI PHÁP AI (AI CORE ALGORITHM)

Sử dụng thuật toán **Pose Estimation (Keypoint Detection)** thay vì Object Detection (Bounding Box) thông thường để đạt độ chính xác toán học cao nhất.

* **Mô hình đề xuất:** YOLOv8-Pose (phiên bản nano/small tối ưu TensorRT).
* **Định nghĩa Keypoints:** Mô hình sẽ được huấn luyện để luôn tìm kiếm 3 điểm mấu chốt trên xe:
  1. `Cabin`: Điểm gốc của đầu kéo/khoang lái.
  2. `Hinge` (Trục bản lề): Điểm xoay mà thùng ben nâng lên hạ xuống.
  3. `Tail` (Đuôi thùng): Điểm xa nhất của mép dưới thùng ben.
* **Toán học (Góc nghiêng):** Hệ thống sẽ tính toán góc $\alpha$ tạo bởi 2 vector: Vector `(Hinge -> Tail)` và Vector `(Hinge -> Cabin)`.
* **Ngưỡng an toàn (Threshold):** Nếu góc $\alpha$ vượt qua một ngưỡng cố định (VD: > 10 độ hoặc 15 độ), xe bị đánh dấu là "Unsafe" (Chưa hạ thùng).

---

## 4. LOGIC KÍCH HOẠT CẢNH BÁO (TRIGGER LOGIC)

Để loại bỏ hoàn toàn các trường hợp báo động giả khi xe đang đỗ và đổ đất, hệ thống kết hợp Tracking và ROI:

1. **Vùng quan tâm (ROI - Region of Interest):** Vẽ một đa giác ảo (Virtual Polygon) ngay tại vị trí luồng xe rời khỏi cổng.
2. **Theo dõi (Object Tracking):** Sử dụng thuật toán `ByteTrack` hoặc `BoT-SORT` để gán ID duy nhất cho mỗi chiếc xe xuất hiện trong khung hình.
3. **Logic kiểm tra:**
   - Hệ thống CHỈ bắt đầu tính toán góc nghiêng khi bounding box của xe đi vào vùng ROI.
   - Trích xuất góc nghiêng trung bình trong $N$ frames liên tiếp (VD: trung bình của 5 frames) để tránh sai số chớp nhoáng (outliers).
   - Nếu kết quả trung bình > Ngưỡng quy định $\Rightarrow$ Phát lệnh Cảnh báo.

---

## 5. KIẾN TRÚC TRIỂN KHAI (STANDALONE EDGE DEPLOYMENT)

Nhằm đảm bảo yêu cầu khắt khe về **độ trễ (Real-time Alerting)**, hệ thống áp dụng kiến trúc **Xử lý tại biên (Edge AI Computing)** thay vì đẩy dữ liệu lên Server.

* **Thiết bị:** NVIDIA Jetson Orin Nano / Orin NX lắp trực tiếp tại trạm/cổng công trường.
* **Tối ưu hóa (Optimization):** Mô hình YOLO-Pose được export sang định dạng **TensorRT (FP16/INT8)** để tận dụng tối đa GPU phần cứng, đảm bảo tốc độ > 30 FPS.
* **Luồng chạy (Pipeline):**
  RTSP Camera $\rightarrow$ GStreamer Decoder $\rightarrow$ YOLO-Pose TensorRT $\rightarrow$ Tracking $\rightarrow$ ROI & Angle Engine $\rightarrow$ GPIO/Relay (Còi/Đèn/Barrier).
* *Lưu ý:* Việc đồng bộ dữ liệu (Log vi phạm, hình ảnh snapshot cảnh báo) về Cloud Server trung tâm sẽ được thực hiện bất đồng bộ (Async Queue) dưới background, hoàn toàn không làm chậm luồng cảnh báo thực tế.

---

## 6. QUY CHUẨN LẮP ĐẶT CAMERA (HARDWARE SETUP)

Quy chuẩn lắp đặt đóng vai trò quyết định để mô hình Pose nhận diện được 3 điểm khớp nối:

1. **Góc quan sát (Camera Viewpoint):** Góc nghiêng ngang sườn xe (Side-profile) chếch từ **40° đến 60°**. Góc xiên này giúp AI nhìn rõ cả chiều dài thân xe và có không gian để nội suy các điểm bị che khuất (Occlusion).
2. **Khung hình (FOV):** Điều chỉnh tiêu cự sao cho chiếc xe tải khi đi vào vùng ROI chiếm khoảng **40% đến 60%** diện tích khung hình camera.
3. **Vị trí độ cao:** Lắp camera trên cột cao **3m - 4.5m** hướng chéo xuống.
4. **Loại Camera:** Độ phân giải 2MP - 4MP. Chống ngược sáng WDR 120dB. Tầm xa hồng ngoại ban đêm (IR) rõ nét.

---

## 7. KẾ HOẠCH TRIỂN KHAI (IMPLEMENTATION PHASES)

* **Giai đoạn 1: Data & Modeling (3 tuần)**
  - Thu thập hình ảnh từ góc camera chuẩn.
  - Gán nhãn (Labeling) Bounding Box toàn xe + 3 Keypoints.
  - Train YOLOv8-Pose và kiểm chứng MAE (Mean Absolute Error) của góc nghiêng trên tập test.
* **Giai đoạn 2: Software Pipeline (2 tuần)**
  - Code luồng RTSP, ByteTrack, và thuật toán xử lý ROI.
  - Convert model sang TensorRT.
* **Giai đoạn 3: Edge Pilot (2 tuần)**
  - Lắp Jetson chạy thử nghiệm tại 1 cổng trạm. Chỉ lưu log, ghi nhận False Positive / False Negative.
* **Giai đoạn 4: Production & Alerting (1 tuần)**
  - Đấu nối rơ-le phần cứng điều khiển còi/đèn chớp và barrier.
  - Bàn giao vận hành.
