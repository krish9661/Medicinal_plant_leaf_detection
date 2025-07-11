import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# CONFIG 
MODEL_PATH = 'plant_classifier_model.keras'
INPUT_SIZE = (224, 224)
CLASS_NAMES = [
    'Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala', 'Balloon_Vine', 'Bamboo',
    'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte',
    'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry',
    'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger',
    'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine',
    'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut',
    'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion',
    'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin',
    'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind',
    'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'ashoka', 'camphor', 'kamakasturi',
    'kepala'
]
ROLLING_WINDOW = 5
CONFIDENCE_THRESHOLD = 0.5

# Load FLAN-T5 Model for Ayurvedic Info 
FLAN_MODEL_PATH = "Models"  #flan-t5 model path
print("Loading FLAN-T5 model...")
tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_PATH)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL_PATH)
print("FLAN-T5 model loaded.")

def get_ayurvedic_info(plant_name):
    prompt = ("You are an Ayurvedic expert. Given the name of a plant, explain its traditional medicinal uses in 1-2 sentences.\n\n"
        "Example:\n"
        "Plant: Tulsi\n"
        "Uses: Used to boost immunity, reduce stress, and relieve respiratory issues.\n\n"
        "Plant: Neem\n"
        "Uses: Acts as an antiseptic, supports skin health, and purifies blood.\n\n"
        f"Plant: {plant_name}\n"
        "Uses:")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = flan_model.generate(**inputs, max_new_tokens=80)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load classification model 
print("Loading plant classifier model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Setup camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

predictions_buffer = deque(maxlen=ROLLING_WINDOW)
print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    h, w, _ = frame.shape
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    frame_cropped = frame[top:top+min_dim, left:left+min_dim]

    resized = cv2.resize(frame_cropped, INPUT_SIZE)
    normalized = resized.astype("float32") / 127.5 - 1.0
    img_array = np.expand_dims(normalized, axis=0)

    preds = model.predict(img_array, verbose=0)
    class_id = int(np.argmax(preds))
    confidence = float(preds[0][class_id])

    predictions_buffer.append(class_id)
    final_pred = max(set(predictions_buffer), key=predictions_buffer.count)
    final_label = CLASS_NAMES[final_pred]

    if confidence > CONFIDENCE_THRESHOLD:
        text = f"{final_label} ({confidence*100:.1f}%)"
        info = get_ayurvedic_info(final_label)
        print(f"{final_label}: {info}")
        short_info = info[:80] + "..." if len(info) > 80 else info
    else:
        text = "Uncertain"
        short_info = ""

    # Annotate frame
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    if short_info:
        cv2.putText(frame, short_info, (10, 70), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
    cv2.rectangle(frame, (5, 5), (520, 90), (0, 255, 0), 2)

    cv2.imshow("Medicinal Plant Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
