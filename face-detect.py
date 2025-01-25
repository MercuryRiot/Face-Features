import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox, Frame
from PIL import Image, ImageTk

# Load Haar cascade files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

class FaceEyeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face and Eye Detection")
        self.root.state('zoomed')  # Maximize the window

        # Create a frame for the buttons
        button_frame = Frame(root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.upload_button = Button(button_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.detect_button = Button(button_frame, text="Use Webcam", command=self.use_webcam)
        self.detect_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.quit_button = Button(button_frame, text="Quit", command=root.quit)
        self.quit_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Create a frame for the image
        self.image_frame = Frame(root)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.image_label = Label(self.image_frame)
        self.image_label.pack(expand=True)

        self.cap = None
        self.running = False

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                image = cv2.imread(file_path)
                self.detect_features(image, display=True)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def use_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to access webcam")
            return

        self.running = True
        self.detect_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)
        self.run_webcam()

    def stop_webcam(self):
        self.running = False
        self.detect_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def run_webcam(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.detect_features(frame, display=False)
                cv2.imshow("Webcam - Press 'q' to exit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_webcam()
                    break

    def detect_features(self, frame, display):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, 'Eye', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
                cv2.putText(roi_color, 'Mouth', (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if display:
            self.display_image(frame)

    def display_image(self, frame):
        # Get the dimensions of the window
        window_width = self.image_frame.winfo_width()
        window_height = self.image_frame.winfo_height()

        # Resize the image to fit the window while maintaining aspect ratio
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / frame_height

        if window_width / window_height > aspect_ratio:
            new_height = window_height
            new_width = int(window_height * aspect_ratio)
        else:
            new_width = window_width
            new_height = int(window_width / aspect_ratio)

        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Convert the image to RGB format for Tkinter
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image

# Initialize the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEyeDetectionApp(root)
    root.mainloop()