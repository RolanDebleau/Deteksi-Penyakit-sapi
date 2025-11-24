"""
TELEGRAM BOT - DETEKSI PENYAKIT SAPI
Bot untuk mendiagnosis kesehatan sapi dari foto

Features:
- Upload foto sapi
- Deteksi otomatis penyakit
- Hasil diagnosis lengkap dengan rekomendasi
"""

import os
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import json

# ==========================================
# KONFIGURASI
# ==========================================

# Telegram Bot Token (EDIT INI!)
# Dapatkan dari @BotFather di Telegram
BOT_TOKEN = "8307089980:AAGSUI4K_irBEDsPvPTbGL5hAg9JsyF9NVc"  # ← GANTI DENGAN TOKEN BOT ANDA

# Path ke model yang sudah di-training
MODEL_PATH = r"models\final_model.h5"
CLASS_NAMES_PATH = r"models\class_names.json"

# Model configuration
IMG_SIZE = 128

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure you have trained the model first!")
    exit(1)

# Load class names
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_data = json.load(f)
        class_names = class_data['class_names']
    print(f"✅ Classes: {class_names}")
except Exception as e:
    print(f"❌ Error loading class names: {e}")
    exit(1)

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_from_bytes(image_bytes):
    """
    Predict disease from image bytes
    
    Args:
        image_bytes: Image data in bytes
        
    Returns:
        dict: Prediction results
    """
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to array
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        predicted_class = class_names[predicted_class_idx]
        
        # Create result
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None

# ==========================================
# DIAGNOSIS MESSAGE FORMATTER
# ==========================================

def get_diagnosis_message(result):
    """
    Convert prediction to human-readable message
    
    Args:
        result: Prediction result dictionary
        
    Returns:
        str: Formatted diagnosis message
    """
    if result is None:
        return "❌ Gagal memproses gambar. Silakan coba lagi dengan foto yang lebih jelas."
    
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    
    # Diagnosis mapping
    diagnosis_map = {
        'kulit_sehat': {
            'emoji': '✅',
            'status': 'SAPI SEHAT',
            'description': 'Kulit sapi terlihat sehat dan normal. Tidak ditemukan tanda-tanda penyakit.',
            'symptoms': [
                'Permukaan kulit halus',
                'Tidak ada benjolan atau lesi',
                'Warna kulit normal',
                'Kondisi bulu baik'
            ],
            'recommendation': '• Lanjutkan perawatan rutin\n• Pemantauan kesehatan berkala\n• Vaksinasi sesuai jadwal\n• Jaga kebersihan kandang',
            'action': 'Tidak ada tindakan khusus diperlukan.',
            'severity': 'low'
        },
        'kulit_lumpy_skin': {
            'emoji': '⚠️',
            'status': 'TERDETEKSI LUMPY SKIN DISEASE',
            'description': 'Terdeteksi indikasi Lumpy Skin Disease (LSD), penyakit virus yang menyerang sapi.',
            'symptoms': [
                'Nodul/benjolan pada kulit',
                'Pembengkakan',
                'Lesi kulit',
                'Kemungkinan demam'
            ],
            'recommendation': '🚨 TINDAKAN SEGERA:\n• Isolasi sapi dari kawanan lain\n• Hubungi dokter hewan SEGERA\n• Jangan pindahkan sapi\n• Laporkan ke dinas peternakan\n• Tingkatkan biosecurity',
            'action': 'SEGERA konsultasi dengan dokter hewan!',
            'severity': 'high'
        }
    }
    
    diag = diagnosis_map.get(predicted_class, {
        'emoji': '❓',
        'status': 'TIDAK DIKETAHUI',
        'description': 'Tidak dapat mengidentifikasi kondisi dengan jelas.',
        'symptoms': [],
        'recommendation': 'Mohon foto dengan lebih jelas atau konsultasi dokter hewan.',
        'action': 'Upload foto yang lebih jelas.',
        'severity': 'medium'
    })
    
    # Build message
    confidence_bar = '█' * int(confidence * 10) + '░' * (10 - int(confidence * 10))
    
    message = f"""
🐄 **HASIL DIAGNOSIS KESEHATAN SAPI**

{diag['emoji']} **{diag['status']}**

📊 **Tingkat Kepercayaan:**
{confidence_bar} {confidence*100:.1f}%

📝 **Deskripsi:**
{diag['description']}
"""
    
    # Add symptoms if available
    if diag['symptoms']:
        message += "\n\n🔍 **Indikator:**\n"
        for symptom in diag['symptoms']:
            message += f"• {symptom}\n"
    
    # Add recommendation
    message += f"\n\n💊 **Rekomendasi:**\n{diag['recommendation']}"
    
    # Add action
    message += f"\n\n⚡ **Tindakan:** {diag['action']}"
    
    # Add probability details
    message += "\n\n📈 **Detail Probabilitas:**\n"
    for cls, prob in result['all_probabilities'].items():
        cls_display = cls.replace('kulit_', '').replace('_', ' ').title()
        prob_bar = '▓' * int(prob * 20) + '░' * (20 - int(prob * 20))
        message += f"{cls_display:15} {prob_bar} {prob*100:5.1f}%\n"
    
    # Add disclaimer
    message += """
\n⚕️ **Catatan Penting:**
Hasil ini adalah prediksi AI berdasarkan analisis gambar. Untuk diagnosis definitif dan penanganan yang tepat, silakan konsultasi dengan dokter hewan profesional.

📞 **Kontak Darurat:**
• Dinas Peternakan setempat atau hubungi kelompok 7
• Dokter hewan terdekat atau serahkan pada kami kelompok 7
• Hotline: Rahasia karena nomor pribadi pokoke kelompok 7
"""
    
    return message.strip()

# ==========================================
# BOT HANDLERS
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when /start is issued"""
    welcome_message = """
🐄 **Selamat Datang di Bot Deteksi Penyakit Sapi!**

Bot ini menggunakan AI untuk mendeteksi penyakit sapi dari foto kulit.

📸 **Cara Menggunakan:**
1. Kirim foto kulit sapi
2. Bot akan menganalisis foto
3. Dapatkan hasil diagnosis dan rekomendasi

🎯 **Yang Dapat Dideteksi:**
• Kulit Sehat
• Lumpy Skin Disease (LSD)

💡 **Tips Foto yang Baik:**
• Fokus pada area kulit sapi
• Pencahayaan yang cukup
• Jarak 20-50 cm dari objek
• Foto jelas (tidak blur)
• Hindari foto terlalu gelap/terang

⚠️ **Penting:**
Bot ini adalah alat bantu screening. Untuk diagnosis definitif, konsultasi dengan dokter hewan.

📝 **Perintah:**
/start - Tampilkan pesan ini
/help - Panduan penggunaan
/about - Info tentang bot

Kirim foto sekarang untuk memulai diagnosa!
"""
    await update.message.reply_text(welcome_message, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message"""
    help_text = """
❓ **PANDUAN PENGGUNAAN**

**Langkah-langkah:**
1️⃣ Ambil foto kulit sapi yang jelas
2️⃣ Kirim foto ke bot (sebagai foto, bukan file)
3️⃣ Tunggu beberapa detik untuk analisis
4️⃣ Baca hasil diagnosis dengan seksama

**Tips Foto yang Baik:**
✅ Fokus pada area kulit
✅ Pencahayaan natural/terang
✅ Jarak optimal: 20-50 cm
✅ Tidak blur atau gelap
✅ Tampilkan detail permukaan kulit

❌ **Hindari:**
• Foto blur atau goyang
• Terlalu gelap/terang
• Jarak terlalu jauh
• Objek terpotong

📊 **Interpretasi Hasil:**
• Kepercayaan >80%: Hasil sangat reliable
• Kepercayaan 60-80%: Hasil cukup reliable
• Kepercayaan <60%: Coba foto lebih baik

⚕️ **Disclaimer:**
Bot ini BUKAN pengganti dokter hewan. Untuk diagnosis dan perawatan definitif, selalu konsultasi dengan profesional.

Ada pertanyaan? Hubungi administrator bot.
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send about message"""
    about_text = """
ℹ️ **TENTANG BOT INI**

**Bot Deteksi Penyakit Sapi v1.0**

🤖 **Teknologi:**
• Deep Learning (CNN)
• Transfer Learning (MobileNetV2)
• TensorFlow/Keras
• Python Telegram Bot

📊 **Model:**
• Akurasi: ~85-95%
• Dataset: 900+ gambar
• Classes: 2 (Sehat, Lumpy Skin Disease)

👨‍💻 **Developer:**
[Selawase]

📅 **Version:** 1.0.0
📅 **Last Updated:** 2025

🎯 **Purpose:**
Membantu peternak melakukan screening awal kesehatan sapi untuk deteksi dini penyakit kulit, khususnya Lumpy Skin Disease.

⚖️ **Disclaimer:**
Bot ini dikembangkan untuk tujuan edukasi dan screening awal. Hasil prediksi tidak menggantikan diagnosis medis profesional.

📧 **Contact:**
[Your Email/Contact Info]

🔗 **Source Code:**
[GitHub Link]

**Terima kasih telah menggunakan bot ini!**
Bersama kita jaga kesehatan ternak Indonesia 🇮🇩
"""
    await update.message.reply_text(about_text, parse_mode='Markdown')

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages"""
    
    # Get photo
    photo = update.message.photo[-1]  # Get highest resolution
    
    # Send processing message
    processing_msg = await update.message.reply_text(
        "⏳ Memproses foto...\n"
        "Mohon tunggu beberapa saat untuk analisis AI."
    )
    
    try:
        # Download photo
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        
        # Predict
        logger.info(f"Processing image from user {update.effective_user.id}")
        result = predict_from_bytes(bytes(photo_bytes))
        
        # Generate diagnosis message
        diagnosis = get_diagnosis_message(result)
        
        # Send result
        await processing_msg.edit_text(diagnosis, parse_mode='Markdown')
        
        logger.info(f"Prediction sent: {result['predicted_class']} ({result['confidence']:.2f})")
        
    except Exception as e:
        logger.error(f"Error processing photo: {e}")
        await processing_msg.edit_text(
            "❌ Terjadi kesalahan saat memproses foto.\n\n"
            "Silakan coba lagi dengan foto yang lebih jelas atau hubungi administrator."
        )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document/file messages"""
    await update.message.reply_text(
        "⚠️ **Mohon kirim sebagai foto, bukan file!**\n\n"
        "Caranya:\n"
        "1. Klik ikon 📎 (attachment)\n"
        "2. Pilih **Gallery/Camera**\n"
        "3. Pilih foto\n"
        "4. Kirim langsung (JANGAN compress/edit)\n\n"
        "Atau ambil foto baru dengan kamera dan kirim sebagai foto.",
        parse_mode='Markdown'
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    await update.message.reply_text(
        "📸 **Silakan kirim foto kulit sapi untuk diagnosa.**\n\n"
        "Ketik /help untuk panduan lengkap.",
        parse_mode='Markdown'
    )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors"""
    logger.error(f"Update {update} caused error {context.error}")

# ==========================================
# MAIN
# ==========================================

def main():
    """Start the bot"""
    
    print("="*70)
    print("  STARTING TELEGRAM BOT")
    print("="*70)
    
    # Check bot token
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("\n❌ ERROR: Bot token not configured!")
        print("\nPlease:")
        print("1. Go to @BotFather on Telegram")
        print("2. Create new bot or use existing bot")
        print("3. Copy bot token")
        print("4. Update BOT_TOKEN in this script")
        print("5. Run script again")
        return
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.Document.IMAGE, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(error_handler)
    
    # Start bot
    print("\n✅ Bot is running!")
    print("📱 Open Telegram and search for your bot")
    print("💬 Send /start to begin")
    print("\n⏹️  Press Ctrl+C to stop\n")
    
    # Run bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Bot stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()