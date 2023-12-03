import PySimpleGUI as sg
import cv2
import math
from ultralytics import YOLO
import torch
import pygame
from datetime import datetime, timedelta
import time
import threading

sg.theme('DefaultNoMoreNagging')

def get_gpu_info():
    # Check if GPU is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()

        # Print information for each GPU
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

    else:
        print("No GPU available.")

# Set the device based on availability
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print GPU information
get_gpu_info()

# Initialize Pygame for audio playback
pygame.mixer.init()

# Set the sound file path (replace 'path/to/sound/file.wav' with your sound file)
sound_file_path = 'dontdoit.mp3'

# Set the duration threshold for triggering the sound (in seconds)
sound_trigger_duration = 3.0
sound_delay = 1.0  # Delay for the sound in seconds
last_detection_time = None

def get_available_webcams():
    available_webcams = []
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                webcam_name = f"Webcam {i}"
                _, frame = cap.read()
                if frame is not None:
                    height, width, _ = frame.shape
                    webcam_name = f"Webcam {i} ({width}x{height})"
                available_webcams.append((i, webcam_name))
                cap.release()
        except Exception as e:
            print(f"Error in main loop: {e}")
            pass

    return available_webcams

def play_sound_with_delay():
    # Play the sound
    pygame.mixer.music.load(sound_file_path)
    pygame.mixer.music.play()

    # Add delay for the sound
    time.sleep(sound_delay)

webcam_list = get_available_webcams()
run_model = False

# Create a new window for displaying the image
image_layout = [[sg.Image(filename='', key='image_display')]]
image_window = sg.Window('Image Display', image_layout, finalize=True)

layout = [
            [sg.Text('Detection tester', size=(30, 1), justification='center', font=("Helvetica", 25), relief=sg.RELIEF_RIDGE, pad=((3, 3), 3))],
            [sg.Text('Simple GUI for testing YOLO model by phawitpp', text_color='black', justification='center', font=("Helvetica", 12), relief=sg.RELIEF_RIDGE, pad=((3, 3), 3))],
            [sg.Text('Select available webcam'), sg.Combo(values=webcam_list, key='webcam')],
            [sg.Button('Run', button_color=('white', 'springgreen4')), sg.Button('Stop', button_color=('white', 'firebrick3')), sg.Button('Close')],
            [sg.Text(text_color='black', key='out')],
            [sg.Text('Confidence Threshold:'), sg.Combo(values=[i / 100 for i in range(0, 101)], default_value=0.7, key='confidence_select')],
            [sg.Image(filename='', key='image')]
            ]
sg.theme('Default')
window = sg.Window('No Drinks', layout, text_justification='r', auto_size_text=True, default_element_size=(40, 1), grab_anywhere=False, resizable=True, finalize=True)

while True:
    event, values = window.read(timeout=1)

    if event == sg.WINDOW_CLOSED or event == 'Close':
        break

    if event == 'Run':
        model = YOLO('best.pt')
        model = model.to(device)
        class_list = model.names
        webcamnumber = values['webcam'][0]
        video = cv2.VideoCapture(webcamnumber, cv2.CAP_DSHOW)
        run_model = True

    if event in ('Stop', sg.WIN_CLOSED, 'Close'):
        if run_model:
            run_model = False
            video.release()
            window['image'].update(filename='')

    if run_model:
        ret, frame = video.read()
        if ret:
            # Move the input frame to the selected device
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

            results = model(frame_tensor, stream=True)

            # Inside the loop where you process the results
            confidence_threshold = float(values['confidence_select'])

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    # Check if confidence level exceeds the threshold
                    if confidence >= confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        confidence = math.ceil((confidence * 100)) / 100
                        cls = int(box.cls[0])
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2
                        cv2.putText(frame, f"{class_list[cls]} Confidence: {confidence}", org, font, fontScale, color, thickness)
                        # Check if the same class has been detected continuously for the specified duration
                        if last_detection_time is None or datetime.now() - last_detection_time > timedelta(seconds=sound_trigger_duration):
                            # Create a thread to play the sound with delay
                            sound_thread = threading.Thread(target=play_sound_with_delay)
                            sound_thread.start()

                            last_detection_time = datetime.now()

            # Convert the image to bytes for PySimpleGUI
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            # Update the image element in the GUI
            window['image'].update(data=imgbytes)

            # Update the image in the new window
            imgbytes_display = cv2.imencode('.png', frame)[1].tobytes()
            image_window['image_display'].update(data=imgbytes_display)

        else:
            video.release()
            run_model = False

# Close the new image window
image_window.close()
window.close()
