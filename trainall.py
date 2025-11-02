import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os



DATASET_DIR = r"C:\VIT\AIML\data"
BATCH_SIZE = 8
IMG_SIZE = (224, 224)
EPOCHS = 20
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 5e-4
L2_REGULARIZATION = 1e-4
DROPOUT_CONV = 0.3
DROPOUT_DENSE = 0.5

print(f"Dataset directory: {DATASET_DIR}")



train_gen = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    validation_split=VALIDATION_SPLIT,
    subset='training'
)

val_gen = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    validation_split=VALIDATION_SPLIT,
    subset='validation'
)


print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)
print(f"Class names: {train_gen.class_names}")
print(f"Class 0: {train_gen.class_names[0]}")
print(f"Class 1: {train_gen.class_names[1]}")
print("="*60 + "\n")

print(f"Datasets loaded successfully")


total_samples = 1324
fire_ext_count = 400
not_fire_count = 924

class_weight = {
    0: total_samples / (2 * fire_ext_count),    
    1: total_samples / (2 * not_fire_count)     
}

print("\n" + "="*60)
print("CLASS WEIGHT BALANCING")
print("="*60)
print(f"{train_gen.class_names[0]} (Class 0): {fire_ext_count} images, weight: {class_weight[0]:.3f}")
print(f"{train_gen.class_names[1]} (Class 1): {not_fire_count} images, weight: {class_weight[1]:.3f}")
print(f"Imbalance ratio: {not_fire_count/fire_ext_count:.2f}:1")
print("="*60 + "\n")


def normalize_data(images, labels):
    return images / 255.0, labels

train_gen = train_gen.map(normalize_data)
val_gen = val_gen.map(normalize_data)


data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.25),
    layers.RandomFlip("horizontal"),
    layers.RandomFlip("vertical"),
    layers.RandomZoom(0.25),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.1),
])

train_gen_augmented = train_gen.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_gen_augmented = train_gen_augmented.prefetch(tf.data.AUTOTUNE)
val_gen = val_gen.prefetch(tf.data.AUTOTUNE)


model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                  kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
                  input_shape=(224,224,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(DROPOUT_CONV),
    
    layers.Conv2D(64, (3,3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(DROPOUT_CONV),
    
    layers.Conv2D(128, (3,3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(DROPOUT_CONV),
    
    layers.Conv2D(256, (3,3), activation='relu', padding='same',
                  kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(DROPOUT_CONV),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu',
                 kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
    layers.BatchNormalization(),
    layers.Dropout(DROPOUT_DENSE),
    
    layers.Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
    layers.BatchNormalization(),
    layers.Dropout(DROPOUT_DENSE),
    
    layers.Dense(1, activation='sigmoid')
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

model.summary()


callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]


print("\n" + "="*60)
print("STARTING TRAINING WITH CLASS WEIGHTS")
print("="*60)

history = model.fit(
    train_gen_augmented,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight, 
    verbose=1
)


print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model_path = os.path.join(DATASET_DIR, 'fire_extinguisher_classifier_v2.h5')
model.save(model_path)
print(f" Model saved to: {model_path}")


print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(val_gen)
print(f"\nValidation Accuracy: {val_acc:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation AUC: {val_auc:.4f}")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(DATASET_DIR, 'training_history.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n Training history plot saved to: {plot_path}")
plt.show()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
