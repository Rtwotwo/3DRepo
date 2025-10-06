import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import mediapipe as mp
import threading
import time


# Initialize the MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class GestureDataCollection:
    """create tkinter interface for gesture data collection"""
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Data Collection")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        # 摄像头状态
        self.cap = None
        self.is_running = False
        self.is_recording = False
        # 创建数据目录
        self.image_dir = "data/gesture_images"
        self.video_dir = "data/gesture_videos"
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # MediaPipe Hands 初始化（稍后启动）
        self.hands = None
        # GUI 组件
        self.setup_ui()
        # 启动摄像头
        self.start_camera()

    def setup_ui(self):
        title_label = tk.Label(self.root, text="Gesture Data Collection", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        # 视频显示区域
        self.video_label = tk.Label(self.root)
        self.video_label.pack(padx=10, pady=10)

        # 按钮框架
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        # 拍照按钮
        self.capture_btn = tk.Button(
            button_frame, text="📸 Capture Image", font=("Arial", 12),
            bg="lightgreen", width=15, height=2, command=self.capture_image
        )
        self.capture_btn.grid(row=0, column=0, padx=10)

        # 录制视频按钮
        self.record_btn = tk.Button(
            button_frame, text="🎥 Record 2s Video", font=("Arial", 12),
            bg="lightcoral", width=15, height=2, command=self.start_recording
        )
        self.record_btn.grid(row=0, column=1, padx=10)

        # 退出按钮
        self.quit_btn = tk.Button(
            button_frame, text="⏹️ Quit", font=("Arial", 12),
            bg="lightgray", width=10, height=2, command=self.close_app
        )
        self.quit_btn.grid(row=0, column=2, padx=10)

        # 状态标签
        self.status_label = tk.Label(self.root, text="Status: Ready", font=("Arial", 12), fg="blue")
        self.status_label.pack(pady=5)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.show_error(self.root, "Error", "Cannot open camera!")
            self.root.quit()
            return

        # 启动 MediaPipe
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.is_running = True
        self.update_frame()

    def update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # 镜像
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 手势检测
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # 转为 PIL 图像显示在 Tkinter 中
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # 如果正在录像
            if self.is_recording and hasattr(self, 'out'):
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                self.out.write(bgr_frame)
        self.video_label.after(10, self.update_frame)
    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                # 可选：只在检测到手时保存
                filename = os.path.join(self.image_dir, f"gesture_{int(time.time())}.jpg")
                cv2.imwrite(filename, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                self.status_label.config(text=f"📸 Image saved: {filename}", fg="green")
            else:
                self.status_label.config(text="⚠️ No hand detected!", fg="orange")
        else:
            self.status_label.config(text="❌ Failed to capture image", fg="red")
    def start_recording(self):
        if self.is_recording:
            return  # 防止重复点击
        self.is_recording = True
        self.status_label.config(text="🔴 Recording 2 seconds...", fg="red")
        # 新线程执行录制，避免阻塞 GUI
        threading.Thread(target=self.record_video_thread, daemon=True).start()
    def record_video_thread(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = os.path.join(self.video_dir, f"gesture_video_{int(time.time())}.avi")
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
        start_time = time.time()
        while time.time() - start_time < 2.0 and self.is_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            time.sleep(1/30)  # 约 30 FPS
        out.release()
        self.is_recording = False
        self.root.after(0, lambda: self.status_label.config(text=f"🎥 Video saved: {filename}", fg="green"))
    def close_app(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureDataCollection(root)
    root.mainloop()