# Tài liệu Đặc tả Nghiệp vụ: Tính năng Đếm Số Lượng Xe Đổ Nguyên Liệu

## 1. Mô tả yêu cầu (Requirement Description)

* **Bối cảnh:** Hệ thống hiện tại đã ứng dụng AI (YOLO) để nhận diện được trạng thái thùng xe (nâng/hạ) và theo dõi xe (Tracking). Tuy nhiên, để đáp ứng nhu cầu quản lý vận hành khép kín, cần chuyển đổi từ dữ liệu "nhận diện trạng thái" sang dữ liệu "thống kê sự kiện".
* **Mục tiêu:** Xây dựng module tự động đếm số lượt xe đổ nguyên liệu thành công dựa trên luồng video camera giám sát bãi đổ theo thời gian thực.
* **Ý nghĩa & Giá trị:**
  * **Đối soát số liệu:** Khớp nối dữ liệu với trạm cân đầu vào. Ngăn chặn các rủi ro thất thoát (ví dụ: xe qua cân nhưng không đổ, đổ trộm, hoặc quay vòng xe khống).
  * **Báo cáo tự động:** Cung cấp số liệu thống kê năng suất theo ca/ngày làm việc một cách minh bạch mà không cần nhân sự túc trực đếm thủ công.

---

## 2. Yêu cầu nghiệp vụ (Business Requirements)

Để hệ thống hoạt động chính xác trong môi trường công nghiệp phức tạp, một "lượt đổ nguyên liệu" cần thỏa mãn các điều kiện khắt khe sau:

* **2.1. Không gian (Spatial):** Hành động đổ nguyên liệu phải xảy ra bên trong khu vực bãi đổ được quy định, gọi là **Dump ROI (Region of Interest)**. Nâng thùng ngoài khu vực này (ví dụ: bãi bảo trì, đường di chuyển) không được tính.
* **2.2. Hành vi (Behavioral):** Xe phải có sự chuyển đổi trạng thái rõ ràng từ trạng thái `truck_lowered` (hạ thùng) sang trạng thái `truck_raised` (nâng thùng).
* **2.3. Thời gian (Temporal):** Trạng thái `truck_raised` phải được duy trì liên tục và ổn định trong một khoảng thời gian tối thiểu (ví dụ: > 5 giây) để xác nhận đây là hành động đổ thực tế, loại trừ nhiễu do model nhận diện nhầm chớp nhoáng.
* **2.4. Tính duy nhất (Anti-duplicate):** Một chiếc xe (định danh bằng Track ID) đi vào bãi và thực hiện quá trình đổ (dù có thể nhích tới nhích lui hoặc nâng hạ thùng 2-3 lần để rũ sạch đất) chỉ được tính là **01 lượt đếm** trong một phiên xuất hiện tại bãi.
* **2.5. Lưu vết minh chứng (Proof of Dump):** Mỗi khi hệ thống đếm +1, bắt buộc phải lưu lại thông tin: Thời gian (Timestamp), Track ID, và 01 hình ảnh (Snapshot) cắt từ camera ngay tại khoảnh khắc xe đang nâng thùng.

---

## 3. Hướng giải quyết (Solution & Implementation Strategy)

Dựa trên kiến trúc mã nguồn hiện tại, tính năng này sẽ được phát triển qua các bước sau:

### 3.1. Thiết lập Vùng giám sát (ROI Configuration)

* Sử dụng tọa độ đa giác (Polygon) để vẽ và định nghĩa vùng bãi đổ hợp lệ trên khung hình camera.
* **Logic lọc:** Chỉ những xe (Track ID) có tọa độ tâm (center) hoặc cạnh dưới (bottom bounding box) nằm gọn bên trong Polygon này mới được đưa vào luồng kiểm tra logic đếm.

### 3.2. Quản lý trạng thái theo vòng đời xe (Track State Machine)

Mở rộng class `TrackerVoting` để cấp phát một cấu trúc dữ liệu (Dictionary) theo dõi trạng thái cho từng Track ID đang tồn tại trong ROI:

* `is_dumping` (Boolean - Mặc định: `False`): Xe có đang trong quá trình đổ hay không.
* `has_been_counted` (Boolean - Mặc định: `False`): Xe này đã được cộng vào bộ đếm chưa.

**Luồng chạy (Workflow):**

1. **Trigger:** Khi hệ thống voting xác nhận Track ID đạt trạng thái `truck_raised` ổn định -> Đặt `is_dumping = True`. Cắt và lưu tạm ảnh snapshot.
2. **Đếm:** Khi xe hạ thùng xuống (chuyển về `truck_lowered`) hoặc biến mất khỏi khung hình/rời khỏi ROI, VÀ xe đó đang có cờ `is_dumping == True` -> Hệ thống sẽ:
    * Tăng biến đếm: `Total_Dumps += 1`
    * Gắn cờ: `has_been_counted = True` (Để chặn việc đếm lại nếu xe nâng thùng lần 2).
    * Lưu sự kiện vào log/database cùng ảnh snapshot.

### 3.3. Xử lý ngoại lệ: Mất dấu (Occlusion / ID Switch)

Trong thực tế bãi đổ, xe thường bị che khuất bởi khói bụi hoặc xe khác. Nếu Tracker cấp một ID mới cho chiếc xe đang đổ:

* **Giải pháp 1:** Tăng tham số bộ nhớ `track_buffer` của Tracker (như ByteTrack/BoT-SORT) lên cao hơn để Tracker "kiên nhẫn" chờ đợi xe xuất hiện lại sau làn bụi.
* **Giải pháp 2 (Bổ tiết IOU Matching):** Xây dựng logic nối ID. Nếu một Track ID mới xuất hiện ngay tại vị trí (tọa độ IOU trùng khớp cao) mà một Track ID cũ vừa biến mất cách đó vài giây, đồng thời cả hai đều mang trạng thái `truck_raised`, hệ thống sẽ gộp chu kỳ đếm của 2 ID này làm một, tránh bị đếm lặp (double-count).

### 3.4. Giao diện và Tích hợp (UI & Integration)

* **Hiển thị:** In chỉ số đếm (Ví dụ: `Dumps: 45`) và vẽ viền xanh bao quanh vùng ROI trực tiếp lên video stream bằng OpenCV (trong `inference.py`).
* **Lưu trữ sự kiện:** Viết hàm Event Logger để xuất dữ liệu định dạng JSON hoặc lưu vào SQLite/PostgreSQL, thuận tiện cho việc gọi API đồng bộ lên phần mềm quản lý tổng của nhà máy.
