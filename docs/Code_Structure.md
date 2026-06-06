truck_detection_project/
│
├── config/
│   └── settings.py            # Cấu hình RTSP URLs, thông số model, ROI, SKIP_FRAME, ngưỡng Voting
│
├── models/
│   └── yolov8_truck.pt        # File weights YOLOv8 đã train (.pt)
│
├── src/
│   ├── __init__.py
│   ├── camera_stream.py       # Thread đọc luồng RTSP để tránh lag hình
│   ├── image_processor.py     # Logic vẽ ROI, BBox, text lên frame
│   ├── tracker_voting.py      # Logic tracking, lưu trữ lịch sử bằng defaultdict và bầu cử (voting)
│   └── pipeline.py            # Luồng xử lý chính cho từng Camera (Xử lý skip frame, gọi YOLO)
│
├── logs/
│   └── system.log             # Lưu log các trường hợp xe chưa nâng thùng được phát hiện
│
├── requirements.txt           # Thư viện cần thiết (ultralytics, opencv-python, v.v.)
└── main.py                    # Entry point - Khởi chạy toàn bộ hệ thống camera