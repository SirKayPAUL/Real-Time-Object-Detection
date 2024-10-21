import os
import cv2
import numpy as np
import logging
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import random
import winsound  # For sound notification
import time  # To calculate time between frames for speed detection


stop_flag = False  # Global flag to stop processin


# Parameters for distance calculation and speed estimation
KNOWN_WIDTH = 1.8  # Average width of the vehicle in meters
FOCAL_LENGTH = 700  # Focal length of the camera
SPEED_LIMIT = 80  # Speed limit in km/h

# Configure logging
logging.basicConfig(filename='object_detection.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load YOLO model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load object classes (COCO dataset)
with open("CocoDataset/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Generating random colors for each class/object
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# List of harmful objects
harmful_objects = ['pistol', 'rifle', 'knife', 'gun', 'firearm']

# Initialize variables for speed detection
previous_vehicle_positions = {}
previous_frame_time = None

# Helper function to calculate speed
def calculate_speed(prev_position, current_position, time_elapsed):
    if time_elapsed <= 0:
        return 0
    distance_traveled = np.linalg.norm(np.array(current_position) - np.array(prev_position))  # Pixel distance
    meters_traveled = (KNOWN_WIDTH * FOCAL_LENGTH) / distance_traveled  # Convert pixel distance to meters
    speed_m_per_s = meters_traveled / time_elapsed  # Speed in meters per second
    speed_km_per_h = speed_m_per_s * 3.6  # Convert to km/h
    return speed_km_per_h

# Frame processing logic for object detection
def process_frame(frame):
    global previous_vehicle_positions, previous_frame_time
    height, width, _ = frame.shape
    current_frame_time = time.time()  # Timestamp for the current frame
    confidence_threshold = 0.5
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids, confidences, boxes = [], [], []
    crowd_size = 0
    harmful_object_detected = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                detected_class = classes[class_id]
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = tuple(map(int, colors[class_ids[i]].astype(int)))
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {distance:.2f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Detect harmful objects
            if label in harmful_objects:
                harmful_object_detected = True
                logging.warning(f"Harmful object detected: {label} at ({x}, {y})!")
                print(f"Alert! Harmful object detected: {label} at ({x}, {y})")
                cv2.putText(frame, f"ALERT: {label.upper()} DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                winsound.Beep(1000, 3000)  # Play beep for 3 seconds

                        # Check crowd size
            if label == "person":
                crowd_size += 1

            #Beep when one person is detected
            if crowd_size == 1:
                winsound.Beep(1000, 500)  # Short beep for 500 milliseconds

            #Detect vehicles and calculate the speed
            if label == "car" or label == "truck" or label == "bus" or label == "Lorry":
                current_position = (x + w // 2, y + h // 2)  # Vehicle center position
                if previous_frame_time is not None and label in previous_vehicle_positions:
                    time_elapsed = current_frame_time - previous_frame_time
                    prev_position = previous_vehicle_positions[label]
                    speed = calculate_speed(prev_position, current_position, time_elapsed)

                    # Display speed
                    cv2.putText(frame, f"Speed: {speed:.2f} km/h", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Check if vehicle exceeds speed limit
                    if speed > SPEED_LIMIT:
                        logging.warning(f"Speeding detected: {speed:.2f} km/h for {label} at ({x}, {y})!")
                        print(f"Speeding detected: {speed:.2f} km/h for {label} at ({x}, {y})")
                        winsound.Beep(2000, 2000)  # Warning beep for 2 seconds

                #Update the previous vehicle position and time
                previous_vehicle_positions[label] = current_position

    #Update the time for the next frame
    previous_frame_time = current_frame_time

    if crowd_size > 5:
        logging.warning(f"Large crowd detected! Crowd size: {crowd_size}")
        print(f"Warning: Large crowd detected! Crowd size: {crowd_size}")
        winsound.Beep(1500, 4000)  # Long beep for 4 seconds

    return frame

def stop_processing():
    global stop_flag
    stop_flag = True  # Set the flag to True to stop the processing
    print("Stop button pressed! Processing will stop.")


def process_image(image_path, output_dir, max_width, max_height):
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Error: Could not read image {image_path}.")
        print(f"Error: Could not read image {image_path}.")
        return
    img = resize_image_if_needed(img, max_width, max_height)
    result_img = process_frame(img)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, result_img)
    cv2.imshow("Image", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Resize image if too large
def resize_image_if_needed(image, max_width, max_height):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def process_video(output_dir):
    global stop_flag
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open video device.")
        print("Error: Could not open video device.")
        return
    # adjust_camera_settings(cap)
    frame_count = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"output_{timestamp}.avi"
    video_writer = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_frame = process_frame(frame)
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(os.path.join(output_dir, video_filename), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            video_writer.write(result_frame)
            frame_count += 1
            cv2.imshow("Video", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

def process_uploaded_video(video_path, output_dir):
    global stop_flag
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}.")
        print(f"Error: Could not open video file {video_path}.")
        return
    frame_count = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"output_{timestamp}.avi"
    video_writer = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret or stop_flag:
                break
            result_frame = process_frame(frame)
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(os.path.join(output_dir, video_filename), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            video_writer.write(result_frame)
            frame_count += 1
            cv2.imshow("Video", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

def select_image():
    global stop_flag
    stop_flag = False
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if image_path:
        process_image(image_path, "output_images", max_width=800, max_height=800)

def select_video():
    global stop_flag
    stop_flag = False
    video_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("Video Files", "*.mp4 *.avi")])
    if video_path:
        process_uploaded_video(video_path, "output_videos")

def start_camera():
    global stop_flag
    stop_flag = False
    process_video("output_videos")
    

def start_process():
    mode = mode_var.get()
    output_dir = output_dir_var.get()
    max_width = int(max_width_var.get())
    max_height = int(max_height_var.get())
    image_path = None
    video_path = None
    if mode == "image":
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not image_path:
            messagebox.showwarning("No Image Selected", "Please select an image to process.")
            return
    elif mode == "video_file":
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        if not video_path:
            messagebox.showwarning("No Video Selected", "Please select a video to process.")
            return
    if mode == "image":
        process_image(image_path, output_dir, max_width, max_height)
    elif mode == "video":
        process_video(output_dir)
    elif mode == "video_file":
        process_uploaded_video(video_path, output_dir)

# tkinter GUI
root = tk.Tk()
root.title("Real-Time Object Detection")

# variables
mode_var = tk.StringVar(value="image")
output_dir_var = tk.StringVar(value="Output")
max_width_var = tk.StringVar(value="1280")
max_height_var = tk.StringVar(value="720")

# create GUI elements
mode_label = tk.Label(root, text="Mode:")
mode_label.grid(row=0, column=0, padx=5, pady=5)
mode_frame = tk.Frame(root)
mode_frame.grid(row=0, column=1, padx=5, pady=5)
image_mode_rb = tk.Radiobutton(mode_frame, text="Image", variable=mode_var, value="image")
video_mode_rb = tk.Radiobutton(mode_frame, text="Video (Camera)", variable=mode_var, value="video")
video_file_mode_rb = tk.Radiobutton(mode_frame, text="Video Upload", variable=mode_var, value="video_file")
image_mode_rb.pack(side="left")
video_mode_rb.pack(side="left")
video_file_mode_rb.pack(side="left")

output_dir_label = tk.Label(root, text="Output Directory:")
output_dir_label.grid(row=1, column=0, padx=5, pady=5)
output_dir_entry = tk.Entry(root, textvariable=output_dir_var, width=40)
output_dir_entry.grid(row=1, column=1, padx=5, pady=5)
output_dir_button = tk.Button(root, text="Browse", command=lambda: output_dir_var.set(filedialog.askdirectory()))
output_dir_button.grid(row=1, column=2, padx=5, pady=5)

max_width_label = tk.Label(root, text="Max Width:")
max_width_label.grid(row=2, column=0, padx=5, pady=5)
max_width_entry = tk.Entry(root, textvariable=max_width_var, width=10)
max_width_entry.grid(row=2, column=1, padx=5, pady=5)

max_height_label = tk.Label(root, text="Max Height:")
max_height_label.grid(row=3, column=0, padx=5, pady=5)
max_height_entry = tk.Entry(root, textvariable=max_height_var, width=10)
max_height_entry.grid(row=3, column=1, padx=5, pady=5)

# Buttons
start_button = tk.Button(root, text="Start", command=start_process)
start_button.grid(row=4, column=0, padx=5, pady=10)

stop_button = tk.Button(root, text="Stop", command=root.quit)  # Stop button to exit the app
stop_button.grid(row=4, column=1, padx=5, pady=10)

quit_button = tk.Button(root, text="Quit", command=root.quit)
quit_button.grid(row=4, column=2, padx=5, pady=10)

# Start the Tkinter loop
root.mainloop()
