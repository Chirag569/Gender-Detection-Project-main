# ---------------------------- Importing Libraries -------------------------------------------------------
import cv2
import torch
import time
from PIL import Image, ImageTk
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import tkinter as tk
from tkinter import filedialog, Label, Button
import numpy as np

# ---------------------------- Load Pretrained Model -----------------------------------------------------
print("Loading Hugging Face gender classification model...")
extractor = AutoFeatureExtractor.from_pretrained("rizvandwiki/gender-classification-2")
model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification-2")

# Load OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------------------- Gender Prediction Function ------------------------------------------------
def predict_gender(face):
    # Convert OpenCV (BGR) to PIL (RGB)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(face_rgb)

    # Preprocess image and predict
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        confidence = torch.softmax(logits, dim=1)[0][predicted_label].item()
        label_text = model.config.id2label[predicted_label]

    return label_text, confidence

# ---------------------------- Image Upload GUI -----------------------------------------------------------
def launch_image_classifier():
    def upload_image():
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            result_label.config(text="No face detected. Try another image.")
            return

        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            gender, confidence = predict_gender(face_roi)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"{gender} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

        # ---- Resize for GUI display ----
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        max_width, max_height = 500, 400  # display size inside the window
        img_pil.thumbnail((max_width, max_height))  # maintain aspect ratio

        img_tk = ImageTk.PhotoImage(img_pil)
        image_label.configure(image=img_tk)
        image_label.image = img_tk

        result_label.config(text=f"Detected: {gender} ({confidence*100:.1f}%)")

    def clear_image():
        image_label.config(image="")
        image_label.image = None
        result_label.config(text="")

    # Create GUI window
    window = tk.Tk()
    window.title("Gender Classification - Upload Image Mode")
    window.geometry("700x600")
    window.configure(bg="#ECECEC")

    heading = Label(window, text="Gender Classification (Upload Image)",
                    font=("Arial", 18, "bold"), bg="#ECECEC", fg="#333")
    heading.pack(pady=20)

    upload_btn = Button(window, text="Upload Image", command=upload_image,
                        bg="#364156", fg="white", font=("Arial", 12, "bold"),
                        padx=10, pady=5)
    upload_btn.pack(pady=10)

    clear_btn = Button(window, text="Clear Image", command=clear_image,
                       bg="#999999", fg="white", font=("Arial", 12, "bold"),
                       padx=10, pady=5)
    clear_btn.pack(pady=5)

    image_label = Label(window, bg="#ECECEC")
    image_label.pack(pady=20)

    result_label = Label(window, text="", bg="#ECECEC", fg="#111", font=("Arial", 14, "bold"))
    result_label.pack(pady=10)

    window.mainloop()

# ---------------------------- Webcam Detection -----------------------------------------------------------
def run_webcam():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit webcam window.")
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            gender, confidence = predict_gender(face_roi)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{gender} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Gender Classification - Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------- Main Menu -------------------------------------------------------
def main():
    print("\nChoose an option:")
    print("1. Run Live Webcam Gender Detection")
    print("2. Upload and Classify Image (GUI)")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        run_webcam()
    elif choice == "2":
        launch_image_classifier()
    else:
        print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
