import tensorflow as tf
import cv2
import numpy as np
import os
from pathlib import Path


MODEL_PATH = r"C:\VIT\AIML\fire_extinguisher_classifier_v2.h5"
TEST_FOLDER = r"C:\VIT\AIML\test"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.5

CLASS_NAMES = ['fire_ext', 'not_fire']  


print("="*82)
print("FIRE EXTINGUISHER CLASSIFIER - TEST MODE")
print("="*82)
print(f"Model: {os.path.basename(MODEL_PATH)}")
print(f"Test folder: {TEST_FOLDER}")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print("="*82 + "\n")


print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully\n")


image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
image_files = [f for f in Path(TEST_FOLDER).glob('*') if f.suffix in image_extensions]

if not image_files:
    print(f"No images found in {TEST_FOLDER}")
    print("Please add test images to the folder.")
    exit()

print(f"Found {len(image_files)} test images\n")


print(f"{'Filename':<45} {'Prediction':<25} {'Confidence':<12}")
print("="*82)

fire_count = 0
not_fire_count = 0
results = []

for img_path in sorted(image_files):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Skipping {img_path.name} (unable to read)")
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    prediction_prob = model.predict(img_batch, verbose=0)[0][0]
    
    if prediction_prob > CONFIDENCE_THRESHOLD:
        predicted_class = 1 
        label = "Not Fire Extinguisher"
        confidence = prediction_prob
        not_fire_count += 1
    else:
        predicted_class = 0 
        label = "Fire Extinguisher"
        confidence = 1 - prediction_prob
        fire_count += 1
    
    results.append({
        'filename': img_path.name,
        'label': label,
        'confidence': confidence,
        'probability': prediction_prob
    })
    
    print(f"{img_path.name:<45} {label:<25} {confidence*100:.1f}%")


print("="*82)
print("\nPREDICTION SUMMARY:")
print(f"   Fire Extinguisher: {fire_count}")
print(f"   Not Fire Extinguisher: {not_fire_count}")
print(f"   Total Images: {fire_count + not_fire_count}")
#testtest

print("\nCONFIDENCE BREAKDOWN:")
high_conf = [r for r in results if r['confidence'] > 0.8]
med_conf = [r for r in results if 0.6 <= r['confidence'] <= 0.8]
low_conf = [r for r in results if r['confidence'] < 0.6]

print(f"   High confidence (>80%): {len(high_conf)}")
print(f"   Medium confidence (60-80%): {len(med_conf)}")
print(f"   Low confidence (<60%): {len(low_conf)}")

if low_conf:
    print("\nLOW CONFIDENCE PREDICTIONS:")
    for r in low_conf:
        print(f"   {r['filename']:<45} {r['label']:<25} {r['confidence']*100:.1f}%")

print("\n" + "="*82)
print("Testing complete!")
print("="*82)
