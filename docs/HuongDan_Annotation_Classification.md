# HƯỚNG DẪN GÁN NHÃN DỮ LIỆU (DATA ANNOTATION) CHO PHƯƠNG PHÁP IMAGE CLASSIFICATION

Tài liệu này hướng dẫn chi tiết cách chuyển đổi tư duy gán nhãn từ **Pose Estimation** (3 điểm Keypoints) sang **Image Classification** (Phân loại hình ảnh) để nhận diện trạng thái thùng xe ben (Đã hạ an toàn vs. Chưa hạ/Đang nâng).

---

## 1. SO SÁNH PHƯƠNG PHÁP & THIẾT KẾ KIẾN TRÚC

Trước khi bắt tay vào gán nhãn, chúng ta cần lựa chọn thiết kế kiến trúc phù hợp. Có **3 hướng tiếp cận chính** khi chuyển sang Image Classification:

| Hướng tiếp cận | Cách hoạt động | Cách gán nhãn | Ưu điểm | Nhược điểm | Đánh giá |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cách 1: Phân loại toàn khung hình** *(Whole-Frame Classification)* | Đưa cả ảnh từ camera vào mô hình để phân loại: `an_toan` hoặc `nguy_hiem`. | Chia ảnh raw vào 2 thư mục tương ứng với 2 nhãn. | Cực kỳ nhanh, không cần vẽ khung hay cắm điểm. | Dễ bị nhiễu do nền, thời tiết, hoặc khi có nhiều xe khác nhau xuất hiện cùng lúc. Không hỗ trợ Tracking/ROI. | **Không khuyên dùng** cho thực tế. |
| **Cách 2: Phân loại hai giai đoạn (Hai mô hình)** *(2-Stage: Detection + Crop Classification)* | **Bước 1:** Dùng YOLOv8 thông thường để Detect xe ben (ra Box).<br>**Bước 2:** Cắt (Crop) vùng ảnh chứa xe ben đó và đẩy vào mô hình YOLOv8-Cls để phân loại `ha_thung` hoặc `chua_ha_thung`. | **1.** Vẽ Box bao quanh xe để train Detector.<br>**2.** Cắt ảnh xe, phân loại vào 2 thư mục để train Classifier. | Độ chính xác cao, loại bỏ hoàn toàn nhiễu nền. Hỗ trợ Tracking từng xe và chỉ cảnh báo trong vùng ROI. | Cần chạy 2 mô hình nối tiếp (tốn tài nguyên hơn một chút nhưng Jetson Orin hoàn toàn đáp ứng tốt). | **Khuyên dùng số 1** cho hệ thống lớn, ổn định cao. |
| **Cách 3: Object Detection đa lớp (Một mô hình)** *(Single-Stage Multi-Class)* | Dùng duy nhất 1 mô hình YOLOv8 Object Detection để phát hiện xe, nhưng phân chia trực tiếp thành 2 Class xe ben khác nhau. | Vẽ Box bao quanh xe và gán 1 trong 2 nhãn:<br>+ `lowered_truck` (xe đã hạ thùng)<br>+ `raised_truck` (xe chưa hạ thùng) | Đơn giản trong triển khai (chỉ 1 pipeline). Có toạ độ để dùng ByteTrack/ROI. Tốc độ suy luận cực nhanh. | Mô hình phải học cả đặc trưng hình dáng xe lẫn trạng thái thùng cùng lúc, cần lượng ảnh lớn hơn. | **Khuyên dùng số 2** (Giải pháp cân bằng tốt nhất cho PoC). |

---

## 2. QUY TRÌNH GÁN NHÃN CHO TỪNG PHƯƠNG PHÁP

### PHƯƠNG PHÁP A: GÁN NHÃN CHO KIẾN TRÚC 2 GIAI ĐOẠN (DETECTION + CROP CLASSIFICATION) - *KHUYÊN DÙNG SỐ 1*

Đây là phương pháp chuẩn công nghiệp. Để gán nhãn cho phần **Image Classification (Classifier)**, ta thực hiện các bước sau:

#### Bước 1: Trích xuất & Cắt ảnh (Crop)
1. Sử dụng một mô hình Object Detection có sẵn (ví dụ YOLOv8n pretrained trên COCO hoặc mô hình tự train sơ bộ) để tự động phát hiện xe ben trong tập dữ liệu video của bạn.
2. Viết script cắt (crop) các Bounding Box chứa xe ben ra thành các ảnh nhỏ riêng biệt.
3. *Lưu ý:* Chỉ giữ lại những ảnh cắt rõ nét, không bị che khuất quá 50% và có góc nhìn bên sườn (side-profile) tối ưu.

#### Bước 2: Tổ chức thư mục dữ liệu chuẩn YOLO-Classification
YOLOv8-Classify yêu cầu dữ liệu được tổ chức dưới dạng cấu trúc thư mục cụ thể chứ không cần file `.txt` chứa tọa độ. Bạn tạo cấu trúc như sau:

```text
dataset_classifier/
├── train/
│   ├── lowered/           # Chứa các ảnh xe ben đã hạ thùng hoàn toàn
│   │   ├── truck_001.jpg
│   │   └── truck_002.jpg
│   └── raised/            # Chứa các ảnh xe ben thùng chưa hạ/đang nâng
│       ├── truck_003.jpg
│       └── truck_004.jpg
└── val/
    ├── lowered/
    │   ├── truck_101.jpg
    │   └── truck_102.jpg
    └── raised/
        ├── truck_103.jpg
        └── truck_104.jpg
```

#### Bước 3: Gán nhãn thủ công (Phân loại ảnh vào thư mục)
* Bạn duyệt qua thư mục ảnh đã crop và kéo-thả chúng vào đúng thư mục `lowered` hoặc `raised`.
* **Quy chuẩn quyết định nhãn:**
  * **`lowered` (An toàn):** Thùng xe nằm khít hoàn toàn lên khung gầm. Không có khe hở góc nghiêng.
  * **`raised` (Nguy hiểm):** Thùng xe có bất kỳ xu hướng nâng lên nào (dù chỉ hé mở góc 3°-5°). Trạng thái đang nâng, nâng một nửa hay nâng hết cỡ đều xếp chung vào nhóm này.

---

### PHƯƠNG PHÁP B: GÁN NHÃN CHO OBJECT DETECTION ĐA LỚP - *KHUYÊN DÙNG SỐ 2 (TỐT NHẤT CHO PoC)*

Nếu muốn tối giản pipeline (chỉ chạy 1 mô hình duy nhất nhưng vẫn lấy được Box để làm Tracking và ROI), hãy chuyển bài toán sang **Object Detection đa lớp**.

#### Bước 1: Thiết lập lớp gán nhãn (Classes)
Thay vì gán nhãn `dump_truck` chung chung, bạn định nghĩa 2 lớp:
1. `lowered_truck` (ID: 0) - Xe ben đã hạ thùng an toàn.
2. `raised_truck` (ID: 1) - Xe ben chưa hạ thùng/đang nâng.

#### Bước 2: Thực hiện vẽ Bounding Box
Sử dụng các công cụ như **CVAT** hoặc **Roboflow**:
1. Vẽ một khung hình chữ nhật (Bounding Box) khít xung quanh toàn bộ chiếc xe (bao gồm cả đầu xe và phần thùng xe đang nâng lên).
2. Chọn nhãn thích hợp:
   * Nếu thùng xe đang áp sát vào thân xe $\rightarrow$ Chọn nhãn `lowered_truck`.
   * Nếu thùng xe có khe hở hoặc đang dựng đứng $\rightarrow$ Chọn nhãn `raised_truck`.

```text
Ảnh mẫu minh họa gán nhãn:
+------------------------------------+
|  [raised_truck]                    |
|  +------------------+              |
|  |     / \          |              |
|  |    /   \ thùng   |              |
|  |   /_____\        |              |
|  |   [Cabin] [Hinge]|              |
|  +------------------+              |
+------------------------------------+
```

#### Bước 3: Export định dạng dữ liệu
Xuất tập dữ liệu ra định dạng **YOLOv8** (chứa các file `.txt` có định dạng `<class_id> <x_center> <y_center> <width> <height>`).

---

## 3. TIÊU CHUẨN VÀ LƯU Ý QUAN TRỌNG KHI GÁN NHÃN

Để mô hình phân loại đạt độ chính xác cao nhất (không bỏ lọt lỗi và ít báo động giả), quy trình gán nhãn phải tuân thủ nghiêm ngặt các quy tắc sau:

### 3.1. Phân định ranh giới vùng chuyển tiếp (Transition State)
* **Vấn đề:** Khi thùng xe bắt đầu nâng lên một chút (góc từ $1^\circ$ đến $5^\circ$), mắt thường đôi khi khó phân biệt rõ ràng.
* **Quy tắc gán nhãn:** 
  * Hãy coi bất kỳ góc nâng nào lớn hơn $3^\circ$ (bắt đầu hở khớp nối) là **`raised`** (chưa hạ thùng).
  * Đối với các ảnh xe đang trong quá trình nâng/hạ dở dang (đứng yên hoặc di chuyển chậm), luôn gán nhãn là **`raised`** để đảm bảo tính an toàn tối đa (Ưu tiên giảm False Negative - bỏ lọt lỗi).

### 3.2. Cân bằng dữ liệu (Dataset Balance)
* Trong thực tế, số lượng xe ben đã hạ thùng (an toàn) đi qua trạm kiểm soát sẽ chiếm **95% - 99%**. Nếu bạn thu thập dữ liệu tự nhiên, mô hình sẽ bị lệch nặng (class imbalance) và xu hướng luôn dự đoán là `lowered`.
* **Giải pháp:**
  * Chủ động quay thêm video các kịch bản xe nâng thùng chạy qua (kịch bản diễn).
  * Đảm bảo tỷ lệ số lượng ảnh trong tập huấn luyện (Train Set) tối thiểu đạt tỉ lệ **60% lowered / 40% raised** hoặc lý tưởng là **50% / 50%**.
  * Sử dụng các kỹ thuật tăng cường dữ liệu (Data Augmentation) như: lật ảnh (flip), thay đổi độ sáng (brightness), thêm nhiễu (noise) đối với class `raised` để làm phong phú dữ liệu.

### 3.3. Xử lý các góc khuất và vật cản (Occlusion)
* Nếu xe ben bị che khuất một phần bởi các xe khác hoặc barrier:
  * **Đối với 2-Stage Crop:** Chỉ đưa vào tập train Classifier các ảnh crop nhìn thấy rõ ít nhất **70% thân xe** và **phải thấy rõ khớp nối giữa thùng và cabin/gầm**.
  * **Đối với Multi-Class Detection:** Vẽ box bao quanh phần xe nhìn thấy được và gán nhãn dựa trên phần thùng xe quan sát được. Nếu hoàn toàn không nhìn thấy phần thùng (bị che lấp hoàn toàn bởi xe container khác đi song song) $\rightarrow$ **Không gán nhãn** (bỏ qua ảnh đó để tránh làm nhiễu mô hình).

### 3.4. Đa dạng hóa các chủng loại xe ben
Mô hình Classification sẽ học các đặc trưng hình ảnh trực tiếp. Do đó, bạn cần đảm bảo tập dữ liệu có sự xuất hiện đa dạng của:
* Xe ben 3 chân, 4 chân cỡ lớn.
* Xe tải nhỏ (xe ben tự chế, xe chở vật liệu nhỏ).
* Các loại xe đầu kéo kéo theo rơ-moóc ben (loại này có thùng rất dài).
* Các màu sắc cabin khác nhau (Xanh lá, Xanh dương, Trắng, Vàng).

---

## 4. HƯỚNG DẪN THAO TÁC TRÊN ROBOFLOW (KHUYÊN DÙNG CHO PoC)

Roboflow là công cụ nhanh nhất giúp bạn thử nghiệm phương pháp này.

### Cách A: Gán nhãn dạng Multi-Class Detection (Cách khuyên dùng số 2)
1. **Tạo Project:** Chọn Project Type là **Object Detection**.
2. **Upload dữ liệu:** Tải toàn bộ ảnh raw lên Roboflow.
3. **Annotate:** 
   * Vẽ bounding box ôm khít xe ben.
   * Gõ nhãn: `lowered_truck` hoặc `raised_truck`.
4. **Generate Dataset:** Thêm các bước Augmentation (Brightness, Blur, Horizontal Flip).
5. **Export:** Chọn định dạng **YOLOv8** và tải link về máy để huấn luyện.

### Cách B: Gán nhãn dạng Classification thuần túy (Dành cho mô hình Classifier ở bước 2 của Cách 1/Cách 2)
1. **Tạo Project:** Chọn Project Type là **Single-Label Classification** (Mỗi ảnh chỉ có 1 nhãn duy nhất).
2. **Upload dữ liệu:** Tải trực tiếp các thư mục ảnh đã được crop từ trước lên. Roboflow sẽ tự động nhận diện tên thư mục con (`lowered`, `raised`) thành các Class tương ứng.
3. **Review:** Kiểm tra lại xem có ảnh nào bị xếp sai nhãn không.
4. **Export:** Chọn định dạng **Folder Structure** hoặc **YOLOv8 Classify** để tải về huấn luyện mô hình phân loại.

---

## KẾT LUẬN & ĐỀ XUẤT CHO DỰ ÁN CỦA BẠN

Để triển khai PoC nhanh nhất nhưng vẫn giữ được logic ROI và Tracking (vốn đã thiết kế ở bản Spec V2.0):
👉 **Chúng tôi đề xuất bạn nên đi theo Phương pháp B (Object Detection đa lớp: `lowered_truck` và `raised_truck`)**.

* **Lý do:** Bạn chỉ cần dùng đúng 1 mô hình YOLOv8 Object Detection (ví dụ `yolov8n.pt` rất nhẹ). Bạn vẫn lấy được Bounding Box để truyền vào thuật toán Tracking (`ByteTrack`), vẫn vẽ được vùng `ROI` tại cổng trạm, mà không cần mất công chạy thêm một mô hình Classification phụ. Việc gán nhãn cũng vô cùng đơn giản trên Roboflow/CVAT.
