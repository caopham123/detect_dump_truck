from ultralytics import YOLO


def test_poc_v1():
  model = YOLO('/training/runs/pose/26-05-18-truck-pose/poc_v1/weights/best.pt')
  results = model('/training/dataset/test/pic1.jpg')
  # results = model.predict(source='dataset/test/pic1.jpg', save=True,conf=0.5)

def test_video_tracking(model_path, video_src):
  model = YOLO(model_path)
  results = model.track(video_src, save=True,conf=0.5, persist=True, save_crop=True, show=True)

  # for r in results:
  #   if r.keypoints is not None:
  #     kpts = r.keypoints.xyn.cpu().numpy() # Normalize keypoints (0-1)
  #     ids = r.boxes.id # Get track ID
      
  #     if ids is not None:
  #       for i, track_id in enumerate(ids):
  #         print(f"----- Xe ID: {int(track_id)}")
  #         if i < len(kpts):
  #           v0 = kpts[i][0]
  #           v1 = kpts[i][1]
  #           v2 = kpts[i][2]
  #           print(f"======V0 (Cabin): {v0}")
  #           print(f"======V1 (Hinge): {v1}")
  #           print(f"======V2 (Tail): {v2}")

if __name__ == '__main__':
  test_poc_v1()
  # test_video_tracking('runs/pose/Truck_Pose_Project/poc_v1-3/weights/best.pt',
  #  'dataset/test/video02.mp4')