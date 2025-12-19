import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import json
from datetime import datetime

#KONFIGURASI - SESUAIKAN PATH ANDA

#Path dataset (hasil organize)
DATASET_PATH = r"C:\Users\INFINIX\Documents\Deteksi Penyakit Sapi\dataset"

train_dir = os.path.join(DATASET_PATH, 'train')
validation_dir = os.path.join(DATASET_PATH, 'validation')
test_dir = os.path.join(DATASET_PATH, 'test')

#Hyperparameters
IMG_SIZE = 128
BATCH_SIZE = 4  #Kurangi jika RAM/GPU terbatas
EPOCHS = 50
LEARNING_RATE = 0.001

#Output paths
MODEL_OUTPUT_DIR = "models"
RESULTS_OUTPUT_DIR = "results"

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

#FUNGSI UTILITAS

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def save_training_info(info, filename="training_info.json"):
    """Save training information"""
    filepath = os.path.join(RESULTS_OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Training info saved: {filepath}")

#1.LOAD & VERIFY DATASET

print_header("STEP 1: LOADING DATASET")

#Data augmentation untuk training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#Hanya rescale untuk validation & test
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#Load data
try:
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\nDataset loaded successfully!")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")
    
    #Save class names for later use
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("\nTroubleshooting:")
    print(f"1. Check paths exist:")
    print(f"   - {train_dir}")
    print(f"   - {validation_dir}")
    print(f"   - {test_dir}")
    print("2. Check folder structure (must have subfolders for each class)")
    exit(1)

#2.BUILD MODEL

print_header("STEP 2: BUILDING MODEL")

def create_model(num_classes):
    #Base model (pre-trained on ImageNet)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    #Freeze base model initially
    base_model.trainable = False
    
    #Build complete model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

model, base_model = create_model(num_classes)

#Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\nModel built successfully!")
print(f"Model parameters: {model.count_params():,}")
model.summary()

#3.CALLBACKS

print_header("STEP 3: SETTING UP CALLBACKS")

#Early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

#Model checkpoint
checkpoint_path = os.path.join(MODEL_OUTPUT_DIR, 'best_model.h5')
model_checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

#Reduce learning rate on plateau
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

#TensorBoard 
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=os.path.join(RESULTS_OUTPUT_DIR, 'logs'),
    histogram_freq=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr, tensorboard_callback]

print("Callbacks configured")

#4.TRAINING PHASE 1 (Frozen Base)

print_header("STEP 4: TRAINING PHASE 1 (FROZEN BASE)")

print("\nStarting training...")
print(f"Expected time: ~30-60 minutes (with GPU) or 2-4 hours (CPU)")

history_phase1 = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nPhase 1 training complete!")

#5.FINE-TUNING PHASE 2 (Unfreeze)

print_header("STEP 5: FINE-TUNING PHASE 2")

#Unfreeze base model
base_model.trainable = True

#Freeze early layers, unfreeze later layers
for layer in base_model.layers[:100]:
    layer.trainable = False

#Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\nStarting fine-tuning...")

history_phase2 = model.fit(
    train_generator,
    epochs=EPOCHS,
    initial_epoch=len(history_phase1.history['loss']),
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nFine-tuning complete!")

#6.EVALUATION

print_header("STEP 6: MODEL EVALUATION")

#Evaluate on test set
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)

#Calculate F1-score
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)

print(f"\nTEST RESULTS:")
print(f"   Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")
print(f"   F1-Score:  {f1_score:.4f}")

#Get predictions
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

#Classification report
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=class_names))

#7.VISUALIZATIONS

print_header("STEP 7: GENERATING VISUALIZATIONS")

import tensorflow as tf

#Deteksi versi TensorFlow
tf_version = tf.__version__
print(f"\nTensorFlow version detected: {tf_version}")

#Tentukan kemungkinan nama metric berdasarkan versi
if tf_version.startswith("2.8") or tf_version.startswith("2.9"):
    possible_train_acc_keys = ['accuracy', 'acc']
    possible_val_acc_keys = ['val_accuracy', 'val_acc']
else:
    possible_train_acc_keys = ['accuracy', 'acc', 'categorical_accuracy']
    possible_val_acc_keys = ['val_accuracy', 'val_acc', 'val_categorical_accuracy']

#Tampilkan info metric yang ada
print("\nAvailable metrics in history_phase1:", history_phase1.history.keys())
print("Available metrics in history_phase2:", history_phase2.history.keys())

#Fungsi bantu agar lebih fleksibel
def get_metric(history, possible_keys):
    for key in possible_keys:
        if key in history:
            print(f"Using metric key: {key}")
            return history[key]
    print(f"Warning: None of {possible_keys} found in history, using zeros as fallback.")
    return [0] * len(history.get('loss', []))

#Ambil data metric
train_acc1 = get_metric(history_phase1.history, possible_train_acc_keys)
train_acc2 = get_metric(history_phase2.history, possible_train_acc_keys)
val_acc1 = get_metric(history_phase1.history, possible_val_acc_keys)
val_acc2 = get_metric(history_phase2.history, possible_val_acc_keys)

#Gabungkan history
history_combined = {
    'accuracy': train_acc1 + train_acc2,
    'val_accuracy': val_acc1 + val_acc2,
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
}

#MULAI PLOT

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

#Accuracy plot
axes[0, 0].plot(history_combined['accuracy'], label='Train')
axes[0, 0].plot(history_combined['val_accuracy'], label='Validation')
axes[0, 0].axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--', label='Fine-tuning starts')
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

#Loss plot
axes[0, 1].plot(history_combined['loss'], label='Train')
axes[0, 1].plot(history_combined['val_loss'], label='Validation')
axes[0, 1].axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--', label='Fine-tuning starts')
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

#Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1, 0], cbar_kws={'label': 'Count'})
axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('True Label')
axes[1, 0].set_xlabel('Predicted Label')

#Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)
colors = ['#2ecc71' if acc > 0.8 else '#f39c12' if acc > 0.6 else '#e74c3c' for acc in per_class_acc]
bars = axes[1, 1].bar(range(len(class_names)), per_class_acc, color=colors)
axes[1, 1].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Class')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_xticks(range(len(class_names)))
axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
axes[1, 1].set_ylim([0, 1.1])
axes[1, 1].grid(True, axis='y', alpha=0.3)

#Tambahkan label di atas bar
for bar, acc in zip(bars, per_class_acc):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(RESULTS_OUTPUT_DIR, 'training_results.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Training plots saved: {plot_path}")

plt.show()

#8.SAVE MODELS

print_header("STEP 8: SAVING MODELS")

#Save full model
final_model_path = os.path.join(MODEL_OUTPUT_DIR, 'final_model.h5')
model.save(final_model_path)
print(f"Full model saved: {final_model_path}")

#Save model for deployment (TFLite - for mobile/bot)
tflite_path = os.path.join(MODEL_OUTPUT_DIR, 'model.tflite')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved: {tflite_path}")

#Save class names
class_names_path = os.path.join(MODEL_OUTPUT_DIR, 'class_names.json')
with open(class_names_path, 'w') as f:
    json.dump({
        'class_names': class_names,
        'class_indices': train_generator.class_indices
    }, f, indent=2)
print(f"Class names saved: {class_names_path}")

#Save training info
training_info = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': {
        'train_samples': train_generator.samples,
        'validation_samples': validation_generator.samples,
        'test_samples': test_generator.samples,
        'num_classes': num_classes,
        'class_names': class_names
    },
    'hyperparameters': {
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE
    },
    'results': {
        'test_accuracy': float(test_accuracy),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1_score': float(f1_score)
    },
    'per_class_accuracy': {
        class_names[i]: float(per_class_acc[i]) 
        for i in range(len(class_names))
    }
}

save_training_info(training_info)

#9. PREDICTION FUNCTION (UNTUK TELEGRAM BOT)

print_header("STEP 9: TESTING PREDICTION FUNCTION")

def predict_from_image(image_path, model, class_names):
    #Load and preprocess image
    img = keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    #Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    predicted_class = class_names[predicted_class_idx]
    
    #Create result dictionary
    result = {
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'all_probabilities': {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }
    }
    
    return result

#Save prediction function as module
predict_module_path = os.path.join(MODEL_OUTPUT_DIR, 'predictor.py')
predictor_code = f'''"""
Predictor module for Telegram Bot
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import os

# Configuration
IMG_SIZE = {IMG_SIZE}
MODEL_PATH = "{final_model_path}"
CLASS_NAMES_PATH = "{class_names_path}"

# Load model
model = keras.models.load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_data = json.load(f)
    class_names = class_data['class_names']

def predict_from_image(image_path):
    """
    Predict disease from image path
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: Prediction results
    """
    # Load and preprocess
    img = keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    predicted_class = class_names[predicted_class_idx]
    
    # Format result
    result = {{
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'all_probabilities': {{
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }}
    }}
    
    return result

def get_diagnosis_message(result):
    """
    Convert prediction to human-readable diagnosis message
    
    Args:
        result: Prediction result dictionary
        
    Returns:
        str: Formatted diagnosis message
    """
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    
    # Map class to diagnosis
    diagnosis_map = {{
        'kulit_sehat': {{
            'status': 'SEHAT',
            'description': 'Kulit sapi terlihat sehat dan normal.',
            'recommendation': 'Lanjutkan perawatan rutin dan pemantauan kesehatan.'
        }},
        'kulit_lumpy_skin': {{
            'status': 'TERDETEKSI PENYAKIT',
            'description': 'Terdeteksi indikasi Lumpy Skin Disease (LSD).',
            'recommendation': 'Segera konsultasikan dengan dokter hewan. Isolasi sapi dari kawanan lain untuk mencegah penyebaran.'
        }}
    }}
    
    diag = diagnosis_map.get(predicted_class, {{
        'status': 'TIDAK DIKETAHUI',
        'description': 'Tidak dapat mengidentifikasi kondisi.',
        'recommendation': 'Mohon foto dengan lebih jelas atau konsultasi dokter hewan.'
    }})
    
    message = f"""
HASIL DIAGNOSIS KESEHATAN SAPI

{{diag['status']}}

Kepercayaan: {{confidence*100:.1f}}%

Deskripsi:
{{diag['description']}}

Rekomendasi:
{{diag['recommendation']}}

Catatan: Hasil ini adalah prediksi AI. Untuk diagnosis definitif, silakan konsultasi dengan dokter hewan.
"""
    
    return message.strip()
'''

with open(predict_module_path, 'w', encoding='utf-8') as f:
    f.write(predictor_code)

print(f"Predictor module saved: {predict_module_path}")

#Test prediction on a sample
print("\nTesting prediction function...")
sample_image = test_generator.filepaths[0]
test_result = predict_from_image(sample_image, model, class_names)

print(f"\nSample image: {os.path.basename(sample_image)}")
print(f"Predicted: {test_result['predicted_class']}")
print(f"Confidence: {test_result['confidence']*100:.2f}%")
print(f"All probabilities:")
for cls, prob in test_result['all_probabilities'].items():
    print(f"   - {cls}: {prob*100:.2f}%")

#FINAL SUMMARY

print_header("TRAINING COMPLETE!")

print(f"""
Model successfully trained and saved!

Output Files:
    - Model (H5): {final_model_path}
    - Model (TFLite): {tflite_path}
    - Class names: {class_names_path}
    - Predictor module: {predict_module_path}
    - Training plots: {plot_path}
    - Training info: {os.path.join(RESULTS_OUTPUT_DIR, 'training_info.json')}

Final Results:
    - Test Accuracy: {test_accuracy*100:.2f}%
    - Test F1-Score: {f1_score:.4f}

Next Steps:
    1. Review training_results.png
    2. Test model with new images
    3. Deploy to Telegram Bot (see telegram_bot.py)
    4. Monitor performance

To use this model in Telegram Bot:
    - Use predictor.py module
    - Import: from models.predictor import predict_from_image, get_diagnosis_message
    - Call: result = predict_from_image(image_path)
    - Get message: message = get_diagnosis_message(result)
""")

print("="*70)