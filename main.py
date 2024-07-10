import os
import time
import cv2
import torch
import base64
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
from ultralytics import YOLO    
from tracker import Tracker

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.device = 'cpu'

    def predict(self, frame, use_gpu=False, classes=None):
        device = 0 if use_gpu and torch.cuda.is_available() else 'cpu'
        return self.model(frame, device=device, classes=classes)

class VideoProcessor:
    def __init__(self, model, detection_threshold, bbox_color, enable_gpu):
        self.model = model
        self.detection_threshold = detection_threshold
        self.bbox_color = bbox_color
        self.enable_gpu = enable_gpu
        self.tracker = Tracker()
        self.in_ids = set()
        self.out_ids = set()
        self.in_line = ((175, 500), (550, 500))
        self.out_line = ((750, 500), (1090, 500))
        self.line_length = 20
        

    def process_frame(self, frame):

        results = self.model.predict(frame, use_gpu=self.enable_gpu, classes=[2,5,7])
        detections = [
            [int(x1), int(y1), int(x2), int(y2), score]
            for result in results
            for x1, y1, x2, y2, score, class_id in result.boxes.data.tolist()
            if score > self.detection_threshold
        ]

        if detections:
            self.tracker.update(frame, detections)

        for track in self.tracker.tracks or []:
            
            self.draw_bounding_box(frame, track.bbox)
            self.draw_id(frame, track.bbox, track.track_id)
            center = ((track.bbox[0] + track.bbox[2]) // 2, (track.bbox[1] + track.bbox[3]) // 2)
            self.in_line_color = self.check_line_crossing(center, self.in_line, self.in_ids, track.track_id)
            

            self.out_line_color = self.check_line_crossing(center, self.out_line, self.out_ids, track.track_id)
            

        cv2.line(frame, self.in_line[0], self.in_line[1], (0,255,0), 2)
        cv2.line(frame, self.out_line[0], self.out_line[1], (0,0,255), 2)

        return frame

    def draw_bounding_box(self, frame, bbox):
        line_length = 7
        bbox_color = (255,255,255)
        x1, y1, x2, y2 = map(int, bbox)
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        cv2.line(frame,(x1,y1),(x1+line_length,y1),bbox_color,2)
        cv2.line(frame, (x1, y1), (x1, y1+ line_length), bbox_color, 2)

        cv2.line(frame, (x2, y1), (x2 - line_length, y1), bbox_color, 2)
        cv2.line(frame, (x2, y1), (x2, y1 + line_length), bbox_color, 2)

        cv2.line(frame, (x1, y2), (x1 + line_length, y2), bbox_color, 2)
        cv2.line(frame, (x1, y2), (x1 , y2 - line_length), bbox_color, 2)

        cv2.line(frame, (x2, y2), (x2-line_length, y2), bbox_color, 2)
        cv2.line(frame, (x2, y2), (x2, y2-line_length), bbox_color, 2)

    def draw_id(self, frame, bbox, track_id,):
        x1, y1, _, _ = map(int, bbox)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 0, 0.5, self.bbox_color, 2)

    def check_line_crossing(self, center, line, ids, track_id):
        if center[0] >= line[0][0] and center[1] >= line[0][1]-5 and center[0] <= line[1][0] and center[1] <= line[1][1]+5:
            ids.add(track_id)
        return (255, 255, 255) if track_id in ids else (0, 255, 0)

class StreamlitApp:
    def __init__(self, video_processor):
        self.video_processor = video_processor
        self.setup_ui()

    def setup_ui(self):
        st.title("Vehicle Tracking")
        st.sidebar.title("Settings")
        app_mode = st.sidebar.selectbox("Choose the app mode", ["About App", "Count Vehicles From Video"])

        if app_mode == "About App":
            st.markdown("## This app is used to track vehicles in a video using **YOLOv8** & **DeepSORT**")
            st.image("img/of-2.gif")
        else:
            self.run_video_processing()

    def run_video_processing(self):
        detection_threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.01)
        enable_gpu = st.sidebar.checkbox("Enable GPU", False) if torch.cuda.is_available() else False

        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4"])
        if video_file_buffer:
            self.video_processor.detection_threshold = detection_threshold
            self.video_processor.enable_gpu = enable_gpu

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("## In Count: ")
                self.in_markdown = st.markdown(f"## {len(self.video_processor.in_ids)}")
            with col2:
                st.markdown("## Out Count: ")
                self.out_markdown = st.markdown(f"## {len(self.video_processor.out_ids)}")
            with col3:
                st.markdown("## FPS: ")
                fps_markdown = st.markdown(f"## 0")
            stframe = st.empty()

            tffile = tempfile.NamedTemporaryFile(delete=False)
            tffile.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tffile.name)

            st.sidebar.text("Input Video")
            st.sidebar.video(tffile.name)

            fps = 0
            frame_count = 0
            start_time = time.time()

            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break
                
                frame = self.video_processor.process_frame(frame)
                stframe.image(frame, channels="BGR", use_column_width=True)
                frame_count += 1
                elapsed_time = time.time() - start_time

                if elapsed_time > 1:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()

                fps_markdown.markdown(f"## {fps:.2f}")
                self.in_markdown.markdown(f"## {len(self.video_processor.in_ids)}")
                self.out_markdown.markdown(f"## {len(self.video_processor.out_ids)}")



# Initialize and run the app
model = YOLOModel("model/yolov8s.pt")
video_processor = VideoProcessor(model, detection_threshold=0.5, bbox_color=(255, 0, 0), enable_gpu=False)
app = StreamlitApp(video_processor)
