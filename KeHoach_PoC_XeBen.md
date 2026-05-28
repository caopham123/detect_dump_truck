# KẾ HOẠCH TRIỂN KHAI PROOF OF CONCEPT (PoC)

## Hệ Thống AI Phát Hiện Xe Tải Nâng Thùng

| Thông tin | Giá trị |
|---|---|
| **Dự án** | Phát hiện xe tải nâng thùng (Dump Truck Detection) |
| **Kỹ thuật AI** | YOLO Object Detection (Classification) |
| **Phiên bản tài liệu** | 3.0 — Chuyển sang YOLO Detection với 2 Class |
| **Ngày cập nhật** | 2026-05-27 |
| **Tổng thời gian PoC** | ~3 tuần (15 ngày làm việc) |

---

## 1. MỤC TIÊU & PHẠM VI PoC

### 1.1. Mục tiêu tổng thể

Chứng minh tính khả thi của việc sử dụng **YOLO Detection** để phân loại trạng thái thùng xe tải (nâng / hạ) theo thời gian thực từ luồng camera IP/CCTV.

### 1.2. Tiêu chí nghiệm thu PoC

| KPI | Ngưỡng tối thiểu | Mục tiêu |
|---|---|---|
| mAP@50 trên tập Validation | ≥ 70% | ≥ 85% |
| Precision (truck_raised) | ≥ 75% | ≥ 90% |
| Recall (truck_raised) | ≥ 70% | ≥ 85% |
| Tốc độ xử lý (Inference FPS) | ≥ 15 FPS | ≥ 25 FPS |
| False Positive trên video demo | ≤ 15% | ≤ 5% |

### 1.3. Định nghĩa 2 nhãn (Label)

| Label ID | Tên nhãn | Mô tả | Trạng thái |
|---|---|---|---|
| `0` | `truck_raised` | Xe tải đang **nâng thùng** (thùng không ở vị trí nằm ngang, góc nghiêng đáng kể) | ⚠️ **VI PHẠM** |
| `1` | `truck_lowered` | Xe tải đã **hạ thùng** an toàn (thùng nằm ngang, song song mặt đường) | ✅ **AN TOÀN** |

> **Lưu ý annotation:** Chỉ gán nhãn `truck_raised` khi thùng rõ ràng tách rời khỏi khung xe (góc nâng trực quan). Trường hợp không nhìn thấy đủ sườn xe → bỏ qua ảnh đó.

---

## 2. KIẾN TRÚC GIẢI PHÁP

```
Camera IP (RTSP)
      │
      ▼
┌──────────────────────────────────┐
│  Frame Grabber (OpenCV)          │
│  cv2.VideoCapture(RTSP_URL)      │
└──────────────┬───────────────────┘
               │  Frame (BGR)
               ▼
┌──────────────────────────────────┐
│  YOLO Detection Model            │
│  YOLOv8n / YOLOv8s              │
│  Class 0: truck_raised           │
│  Class 1: truck_lowered          │
└──────────────┬───────────────────┘
               │  [bbox, class_id, confidence]
               ▼
┌──────────────────────────────────┐
│  Post-Processing Logic           │
│  - Lọc conf ≥ CONF_THRESHOLD     │
│  - Kiểm tra ROI (vùng cổng ra)   │
│  - N-frame majority vote (N=5)   │
└──────────────┬───────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
  truck_raised    truck_lowered
  → Cảnh báo ĐỎ  → Bình thường XANH
  → Log vi phạm  → Tiếp tục theo dõi
```

---

## 3. KẾ HOẠCH THỰC THI CHI TIẾT

### BƯỚC 1 — THU THẬP DỮ LIỆU (Data Collection)

**Thời gian:** 3 - 4 ngày làm việc

#### 1.1. Nguồn dữ liệu

- [ ] **Camera hiện trường:** Trích xuất frame từ luồng RTSP `rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}:554/Chanel_Number`
- [ ] **Video quay bổ sung:** Dùng điện thoại / flycam mô phỏng góc lắp đặt thực tế
- [ ] **Ảnh tham khảo online:** Thu thập từ Google Images, YouTube (xe ben Việt Nam)

#### 1.2. Yêu cầu góc quay & điều kiện

| Yêu cầu | Chi tiết |
|---|---|
| Góc camera | Bên hông xe, chếch 40°–60° so với trục xe |
| Độ cao lắp camera | 3m – 4.5m |
| Loại xe cần có | Ben 3 chân, 4 chân, xe tải thùng lật, xe rơ-moóc |
| Điều kiện ánh sáng | Ban ngày (bắt buộc), Ban đêm / đèn đường (có thể thêm sau) |
| Trạng thái thùng | ~50% truck_raised + ~50% truck_lowered (balance) |

#### 1.3. Script thu thập tự động

Script `training/auto_capture_trucks.py` đã có sẵn, hỗ trợ:

- Kết nối RTSP stream và lưu frame theo interval
- Lọc frame trùng lặp theo image hash

#### 1.4. Mục tiêu số lượng ảnh

| Nhãn | Tối thiểu | Mục tiêu |
|---|---|---|
| `truck_raised` (class 0) | 300 ảnh | 600 ảnh |
| `truck_lowered` (class 1) | 300 ảnh | 600 ảnh |
| **Tổng** | **600 ảnh** | **1,200 ảnh** |

---

### BƯỚC 2 — GÁN NHÃN DỮ LIỆU (Data Annotation)

**Thời gian:** 3 - 4 ngày làm việc

#### 2.1. Tool gán nhãn

- **Khuyến nghị:** [LabelImg](https://github.com/HumanSignal/labelImg) (đã cài sẵn, format YOLO)
- Hướng dẫn chi tiết: xem file `HuongDan_Annotation_Classification.md`

#### 2.2. Quy tắc vẽ Bounding Box

```
✅ ĐÚNG:
  - Box bao sát toàn bộ thân xe (từ đầu cabin đến đuôi thùng)
  - Có thể crop sát mép ảnh nếu xe bị cắt
  - Gán class theo trạng thái thùng HIỆN TẠI trong ảnh

❌ SAI:
  - Box quá nhỏ (chỉ bao phần thùng)
  - Box quá rộng (bao cả nền đường)
  - Gán nhãn khi không nhìn rõ trạng thái thùng
  - Gán nhãn truck_raised cho xe nghiêng đường hoặc đổ vật liệu đứng yên
```

#### 2.3. Các trường hợp biên (Edge Cases)

| Trường hợp | Xử lý |
|---|---|
| Thùng nâng nhưng xe bị che khuất > 50% | Bỏ qua (skip) |
| Nhiều xe trong 1 ảnh | Vẽ box riêng cho từng xe |
| Xe đang đổ vật liệu (đứng im) | Có thể gán `truck_raised` nếu thùng rõ ràng nâng |
| Xe mui bạt / xe container | Bỏ qua (không phải xe ben) |

#### 2.4. Cấu trúc dataset YOLO

```
training/dataset/
├── images/
│   ├── train/          ← 80% ảnh train
│   └── val/            ← 20% ảnh val
├── labels/
│   ├── train/          ← .txt tương ứng
│   └── val/
└── data.yaml           ← cấu hình dataset
```

**Nội dung `data.yaml`:**

```yaml
path: training/dataset
train: images/train
val: images/val

nc: 2
names:
  0: truck_raised
  1: truck_lowered
```

#### 2.5. Phân chia dataset

```bash
# Script tự động phân chia train/val (80/20)
cd training
python split_dataset.py
```

---

### BƯỚC 3 — HUẤN LUYỆN MÔ HÌNH (Model Training)

**Thời gian:** 2 - 3 ngày làm việc

#### 3.1. Môi trường & yêu cầu

| Thành phần | Yêu cầu |
|---|---|
| GPU | NVIDIA RTX 3060 trở lên (VRAM ≥ 6GB) |
| Framework | Ultralytics YOLOv8 (đã cài trong `requirements.txt`) |
| Python | ≥ 3.10 |
| CUDA | ≥ 11.8 |

#### 3.2. Chọn backbone model

| Model | Tham số | Tốc độ | Độ chính xác | Khuyến nghị |
|---|---|---|---|---|
| `yolov8n.pt` | 3.2M | ⚡⚡⚡ Nhanh nhất | ⭐⭐⭐ | PoC nhanh, thiết bị yếu |
| `yolov8s.pt` | 11.2M | ⚡⚡ Nhanh | ⭐⭐⭐⭐ | **✅ Khuyến nghị cho PoC** |
| `yolov8m.pt` | 25.9M | ⚡ Vừa | ⭐⭐⭐⭐⭐ | Production |

#### 3.3. Cấu hình huấn luyện (Jupyter Notebook)

Mở file `training/train_yolo_detect.ipynb`:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # Load pretrained backbone

results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    workers=2,              # Tối ưu hóa tải dữ liệu (1, 2, 4)
    batch=8,                # Điều chỉnh theo VRAM GPU (8, 16, 32)
    patience=30,            # Early stopping
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    flipud=0.0,             # Không flip dọc (xe không lộn ngược)
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    project="runs/detect",
    name="poc_v1",
    save=True,
    plots=True,
)
```

#### 3.4. Theo dõi quá trình training

- Biểu đồ `box_loss`, `cls_loss` phải **giảm đều** qua các epoch
- `mAP50` trên validation phải **tăng** và ổn định
- Nếu `train_loss` tiếp tục giảm nhưng `val_loss` tăng → **Overfitting** → giảm epoch, tăng augmentation

#### 3.5. Kết quả lưu tại

```
training/runs/detect/poc_v1/
├── weights/
│   ├── best.pt       ← Trọng số tốt nhất (dùng cho inference)
│   └── last.pt
├── results.csv
├── confusion_matrix.png
└── results.png
```

---

### BƯỚC 4 — ĐÁNH GIÁ MÔ HÌNH (Model Evaluation)

**Thời gian:** 1 - 2 ngày làm việc

#### 4.1. Đánh giá trên tập Validation

```python
from ultralytics import YOLO

model = YOLO("training/runs/detect/poc_v1/weights/best.pt")
metrics = model.val(data="training/dataset/data.yaml")

print(f"mAP50:           {metrics.box.map50:.3f}")
print(f"mAP50-95:        {metrics.box.map:.3f}")
print(f"Precision:       {metrics.box.mp:.3f}")
print(f"Recall:          {metrics.box.mr:.3f}")
```

#### 4.2. Kiểm tra Confusion Matrix

Xem file `runs/detect/poc_v1/confusion_matrix_normalized.png`:

- **Mục tiêu:** Đường chéo chính (True Positive) phải đạt ≥ 70% mỗi class
- **Cảnh báo:** Nếu `truck_raised` bị predict thành `truck_lowered` nhiều → mô hình nguy hiểm (bỏ sót vi phạm)

#### 4.3. Phân tích nhầm lẫn theo class

| Trường hợp cần kiểm tra | Hành động |
|---|---|
| `truck_raised` → predict `truck_lowered` (False Negative) | **Nghiêm trọng** → Thu thêm ảnh raised, relabel |
| `truck_lowered` → predict `truck_raised` (False Positive) | Cần giảm → Tăng ảnh lowered đa dạng |
| `background` → predict bất kỳ class | Thêm ảnh negative (không có xe) |

#### 4.4. Test trên ảnh đơn

```bash
# Chạy script test có sẵn
cd training
python test_yolo_detect.py
```

---

### BƯỚC 5 — VIẾT CHƯƠNG TRÌNH DEMO REALTIME

**Thời gian:** 2 - 3 ngày làm việc

#### 5.1. Luồng xử lý chính

```python
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_PATH    = os.getenv("MODEL_PATH")
RTSP_URL      = os.getenv("RTSP_STREAM_URL")
CONF_THRESH   = float(os.getenv("CONF_THRESHOLD", 0.5))

model = cv2.VideoCapture(RTSP_URL)  # hoặc path video .mp4
cap   = cv2.VideoCapture(RTSP_URL)

CLASS_COLORS = {
    0: (0, 0, 255),    # truck_raised  → ĐỎ (BGR)
    1: (0, 200, 0),    # truck_lowered → XANH (BGR)
}
CLASS_LABELS = {
    0: "⚠ CANH BAO: THUNG CHUA HA",
    1: "AN TOAN: THUNG DA HA",
}

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  results = model.predict(frame, conf=CONF_THRESH, verbose=False)

  for box in results[0].boxes:
    cls   = int(box.cls[0])
    conf  = float(box.conf[0])
    xyxy  = box.xyxy[0].tolist()
    x1,y1,x2,y2 = map(int, xyxy)

    color = CLASS_COLORS[cls]
    label = f"{CLASS_LABELS[cls]} ({conf:.0%})"

    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
    cv2.putText(frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

  cv2.imshow("Dump Truck Detection - PoC", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
```

#### 5.2. Logic N-frame Majority Vote (chống báo nhầm)

```python
from collections import deque, Counter

VOTE_WINDOW = 5  # Số frame lấy đa số phiếu
history = deque(maxlen=VOTE_WINDOW)

# Trong vòng lặp:
history.append(detected_class)
if len(history) == VOTE_WINDOW:
    majority_class = Counter(history).most_common(1)[0][0]
    if majority_class == 0:
        trigger_alert()  # Chỉ cảnh báo khi đa số N frames = truck_raised
```

---

### BƯỚC 6 — KIỂM THỬ & NGHIỆM THU (Testing & Acceptance)

**Thời gian:** 2 ngày làm việc

#### 6.1. Bộ video kiểm thử

Chuẩn bị ít nhất **3 đoạn video chưa từng dùng trong training:**

| Video | Nội dung | Kỳ vọng |
|---|---|---|
| `test_01_raised.mp4` | Xe tải nâng thùng đi qua cổng | Phát hiện và cảnh báo |
| `test_02_lowered.mp4` | Xe tải hạ thùng bình thường | Không cảnh báo |
| `test_03_mixed.mp4` | Nhiều xe, cả 2 trạng thái xen kẽ | Phân loại đúng từng xe |

#### 6.2. Bảng ghi kết quả kiểm thử

| # | Video | Tổng xe | Detect đúng | False Positive | False Negative | Ghi chú |
|---|---|---|---|---|---|---|
| 1 | test_01_raised | | | | | |
| 2 | test_02_lowered | | | | | |
| 3 | test_03_mixed | | | | | |
| **Tổng** | | | | | | |

#### 6.3. Xuất video demo

```bash
# Lưu video kết quả để trình bày
python demo_realtime.py --source test_videos/test_01_raised.mp4 --save-video
```

---

## 4. CẤU TRÚC THƯ MỤC DỰ ÁN

```
detect_dump_truck/
├── .env                          ← Cấu hình RTSP, model path, threshold
├── requirements.txt              ← Thư viện Python
├── KeHoach_PoC_XeBen.md          ← Tài liệu này
├── HuongDan_Annotation_Classification.md
├── EVALUATE_MODEL.md
├── Spec_PhatHienXeBenChuaHaThung.md
│
├── training/
│   ├── dataset/                  ← Dữ liệu đã annotate (YOLO format)
│   │   ├── images/train/
│   │   ├── images/val/
│   │   ├── labels/train/
│   │   ├── labels/val/
│   │   └── data.yaml
│   ├── dataset_raw/              ← Ảnh thô chưa annotate
│   ├── runs/detect/              ← Kết quả training các phiên bản
│   │   └── poc_v7.1/weights/best.pt  ← Model hiện tại
│   ├── auto_capture_trucks.py    ← Thu thập ảnh từ RTSP
│   ├── sort_dataset_item.py      ← Sắp xếp/lọc dataset
│   ├── split_dataset.py          ← Phân chia train/val
│   ├── test_yolo_detect.py       ← Test inference trên ảnh
│   └── train_yolo_detect.ipynb   ← Notebook training
│
└── model_loader/                 ← Module load model (production)
```

---

## 5. RỦI RO & PHƯƠNG ÁN XỬ LÝ

| Rủi ro | Mức độ | Phương án |
|---|---|---|
| Thiếu ảnh `truck_raised` (class imbalance) | 🔴 Cao | Augmentation mạnh hơn cho class 0; Thu thêm ảnh |
| Model overfitting (train tốt, val kém) | 🔴 Cao | Giảm epoch, tăng augmentation, thêm dropout |
| Ánh sáng xấu / xe bụi → miss detection | 🟡 Trung bình | Thu thêm ảnh điều kiện tương tự; test ban đêm |
| RTSP stream lag / ngắt kết nối | 🟡 Trung bình | Thêm retry logic, buffer frame |
| Xe bị che khuất (occlusion) | 🟡 Trung bình | N-frame voting, không cảnh báo khi confidence thấp |
| Camera góc không chuẩn → sai class | 🔴 Cao | Chuẩn hóa góc lắp đặt trước khi thu thập data |

---

## 6. LỊCH TRÌNH (GANTT ĐƠN GIẢN)

```
Tuần 1        Tuần 2        Tuần 3
│             │             │
├─[B1] Thu thập & Auto-capture (3-4 ngày)
│     ├─[B2] Gán nhãn LabelImg (3-4 ngày)
│             ├─[B3] Training & Tuning (2-3 ngày)
│             │     ├─[B4] Evaluate & Fix (1-2 ngày)
│             │           ├─[B5] Demo Realtime (2-3 ngày)
│             │                 └─[B6] Kiểm thử nghiệm thu (2 ngày)
```

---

## 7. CHECKLIST NGHIỆM THU PoC

### Dữ liệu

- [ ] Dataset ≥ 600 ảnh, cân bằng 2 class
- [ ] 100% ảnh đã annotate, review qua `EVALUATE_MODEL.md`
- [ ] Tập validation độc lập với training

### Mô hình

- [ ] mAP@50 ≥ 70% trên validation set
- [ ] Confusion matrix: True Positive ≥ 70% mỗi class
- [ ] File `best.pt` đã được cập nhật vào `.env`

### Chương trình demo

- [ ] Chạy được trên video file `.mp4`
- [ ] Chạy được trên luồng RTSP camera thực tế
- [ ] Hiển thị bbox màu ĐỎ khi `truck_raised`, XANH khi `truck_lowered`
- [ ] FPS hiển thị ≥ 15 FPS trên máy test

### Nghiệm thu

- [ ] Video demo đã xuất file `.mp4`
- [ ] Bảng kết quả kiểm thử đã điền đầy đủ
- [ ] Báo cáo số liệu FPS, Precision, Recall, mAP
- [ ] Đề xuất bước tiếp theo (Pilot / cải tiến model)

---

## 8. BƯỚC TIẾP THEO SAU PoC

Nếu PoC đạt KPI, chuyển sang **Giai đoạn Pilot:**

1. **Nâng cấp model:** Dùng `yolov8m.pt` hoặc `yolov8l.pt`, tăng dataset lên 3,000+ ảnh
2. **Thêm Object Tracking:** Tích hợp `ByteTrack` để theo dõi ID từng xe
3. **Tích hợp ROI:** Chỉ kích hoạt cảnh báo khi xe đi vào vùng cổng ra
4. **Edge Deployment:** Export model sang TensorRT trên Jetson Orin Nano
5. **Kết nối phần cứng:** Relay GPIO điều khiển còi / đèn / barrier

---

*Tài liệu được tạo và cập nhật bởi Team AI — Longson 2026.*
