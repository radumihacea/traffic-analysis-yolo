# traffic-analysis-yolo
This project represents an end-to-end implementation of a Knowledge-Based System (KBS) for detecting and classifying road objects, utilizing a Convolutional Neural Network (CNN) architecture inspired by YOLO (You Only Look Once) v1.
Developed in **PyTorch**, the system is capable of identifying vehicles, pedestrians, and road infrastructure in real time.

## 🎯 Objective
The goal of this project is to demonstrate the capability of neural networks to perceive and interpret complex traffic environments. This serves as a foundational step for "Smart City" applications (e.g., intelligent traffic light control, traffic monitoring) and Advanced Driver Assistance Systems (ADAS) for autonomous vehicles.

## 📊 Dataset
The model was trained on a dataset containing real traffic footage from Polish roads (Krakow area):
* **Volume:** ~11,000 images (including data augmentation to simulate various lighting conditions).
* **Annotation Format:** Standard YOLO format (normalized coordinates).
* **Number of Classes:** 11 distinct classes.

**Detected Classes:**
`Car`, `Truck`, `Motorcycle`, `Pedestrian`, `Pedestrian-Crossing`, `Green-Traffic-Light`, `Red-Traffic-Light`, `Speed-Limit-Sign`, `Warning-Sign`, `Prohibition-Sign`, `Different-Traffic-Sign`.

## 🛠️ Technologies Used
* **Language:** Python
* **Deep Learning Framework:** PyTorch & Torchvision
* **Data Processing & Visualization:** OpenCV, Pandas, Matplotlib, Seaborn
* **Training Environment:** Google Colab (GPU Acceleration - Tesla T4)

## 🧠 Model Architecture
The system uses a custom CNN architecture (`SimpleYOLO`):
* **Input:** Images resized to `416x416` pixels.
* **Backbone:** 5 sequential convolutional blocks using `LeakyReLU` activations and `MaxPool`, reducing the image to a spatial feature grid.
* **Detection Head:** Predicts objects on a `13x13` grid. Each grid cell predicts 2 Bounding Boxes (B=2).
* **Custom Loss Function:** A from-scratch implementation of the YOLOv1 loss function, which simultaneously optimizes localization (coordinates $x, y, w, h$), object confidence, and semantic classification.

## ⚙️ Key Code Features
* **Custom PyTorch Dataset:** Dynamic image loading and parsing of `.txt` annotation files to optimize memory usage.
* **Data Pipeline:** Implementation of a custom `collate_fn` to handle a variable number of object targets per image during batch generation.
* **Non-Maximum Suppression (NMS):** Smart post-processing technique used to filter out duplicate overlapping predictions and retain only the highest-confidence bounding boxes.

## 🚀 Results
The model was trained for 100 epochs using the Adam optimizer. The custom loss function demonstrated a stable and consistent convergence throughout the training loop.

*(You can add a screenshot of your results here. Upload an image to your repo and use the format below:)*
> `![Detection Result](link-to-your-image.png)`

## 👨‍💻 Author
* **[Your Name]** - Project developed for the Knowledge-Based Systems course.
