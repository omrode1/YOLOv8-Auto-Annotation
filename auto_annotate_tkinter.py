import tkinter as tk
from tkinter import filedialog, ttk
import os
from ultralytics import YOLO
from PIL import Image
import threading

class AutoAnnotateGUI:
    def __init__(self, master):
        self.master = master
        master.title("YOLOv8 Auto-Annotation Tool")
        master.geometry("700x600")

        self.model_path = tk.StringVar()
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.conf_threshold = tk.DoubleVar(value=0.25)

        self.create_widgets()

    def create_widgets(self):
        # Model selection
        tk.Label(self.master, text="YOLOv8 Model:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self.master, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.master, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5, pady=5)

        # Input folder selection
        tk.Label(self.master, text="Input Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self.master, textvariable=self.input_folder, width=50).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.master, text="Browse", command=self.browse_input).grid(row=1, column=2, padx=5, pady=5)

        # Output folder selection
        tk.Label(self.master, text="Output Folder:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(self.master, textvariable=self.output_folder, width=50).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(self.master, text="Browse", command=self.browse_output).grid(row=2, column=2, padx=5, pady=5)

        # Confidence threshold
        tk.Label(self.master, text="Confidence Threshold:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        tk.Scale(self.master, variable=self.conf_threshold, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky="we", padx=5, pady=5)

        # Start button
        tk.Button(self.master, text="Start Auto-Annotation", command=self.start_annotation).grid(row=4, column=1, pady=20)

        # Progress bar
        self.progress = ttk.Progressbar(self.master, length=400, mode='determinate')
        self.progress.grid(row=5, column=0, columnspan=3, padx=20, pady=10)

        # Status label
        self.status_label = tk.Label(self.master, text="")
        self.status_label.grid(row=6, column=0, columnspan=3, pady=5)

    def browse_model(self):
        filename = filedialog.askopenfilename(filetypes=[("PyTorch files", "*.pt")])
        self.model_path.set(filename)

    def browse_input(self):
        folder = filedialog.askdirectory()
        self.input_folder.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory()
        self.output_folder.set(folder)

    def start_annotation(self):
        if not all([self.model_path.get(), self.input_folder.get(), self.output_folder.get()]):
            self.status_label.config(text="Please fill in all fields.")
            return
        
        # Updating UI before starting the thread
        self.status_label.config(text="Starting annotation...")
        
        # Start annotation in a separate thread
        threading.Thread(target=self.run_annotation, daemon=True).start()
        print("Annotation started!")

    def run_annotation(self):
        try:
            # Load the YOLOv8 model
            model = YOLO(self.model_path.get())
            input_folder = self.input_folder.get()
            output_folder = self.output_folder.get()
            conf_threshold = self.conf_threshold.get()

            os.makedirs(output_folder, exist_ok=True)

            image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            total_files = len(image_files)

            # Set progress bar maximum
            self.progress['value'] = 0
            self.progress['maximum'] = total_files

            for i, filename in enumerate(image_files):
                image_path = os.path.join(input_folder, filename)
                
                # Perform inference
                results = model(image_path, conf=conf_threshold)[0]
                
                # Create annotation file
                base_name = os.path.splitext(filename)[0]
                annotation_path = os.path.join(output_folder, f"{base_name}.txt")
                
                # Get image dimensions
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
                
                # Write annotations
                with open(annotation_path, 'w') as f:
                    for box in results.boxes:
                        class_id = int(box.cls)
                        x, y, w, h = box.xywhn[0]
                        f.write(f"{class_id} {x} {y} {w} {h}\n")
                
                # Update progress bar
                self.progress['value'] = i + 1
                self.status_label.config(text=f"Processed {i+1}/{total_files}: {filename}")
                self.master.update_idletasks()

            self.status_label.config(text="Annotation completed!")
        
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoAnnotateGUI(root)
    root.mainloop()
