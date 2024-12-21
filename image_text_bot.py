import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import os
import sys
import traceback
import json
from datetime import datetime
import sqlite3
from pathlib import Path
import numpy as np

# Set Tesseract path and check if it exists
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if not os.path.exists(TESSERACT_PATH):
    print(f"Error: Tesseract not found at {TESSERACT_PATH}")
    print("Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
    sys.exit(1)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Database setup
DB_PATH = 'bot_stats.db'

def init_database():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS stats
                 (user_id TEXT, timestamp TEXT, action TEXT, success INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS processed_images
                 (user_id TEXT, timestamp TEXT, chars_extracted INTEGER)''')
    conn.commit()
    conn.close()

def log_action(user_id, action, success=True):
    """Log user actions to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO stats VALUES (?, ?, ?, ?)",
              (str(user_id), datetime.now().isoformat(), action, int(success)))
    conn.commit()
    conn.close()

def log_processed_image(user_id, chars_extracted):
    """Log processed image statistics."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO processed_images VALUES (?, ?, ?)",
              (str(user_id), datetime.now().isoformat(), chars_extracted))
    conn.commit()
    conn.close()

def get_user_stats(user_id):
    """Get statistics for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get total processed images
    c.execute("SELECT COUNT(*) FROM processed_images WHERE user_id = ?", (str(user_id),))
    total_images = c.fetchone()[0]
    
    # Get total characters extracted
    c.execute("SELECT SUM(chars_extracted) FROM processed_images WHERE user_id = ?", (str(user_id),))
    total_chars = c.fetchone()[0] or 0
    
    # Get success rate
    c.execute("SELECT COUNT(*) FROM stats WHERE user_id = ? AND action = 'process_image'", (str(user_id),))
    total_attempts = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM stats WHERE user_id = ? AND action = 'process_image' AND success = 1", (str(user_id),))
    successful_attempts = c.fetchone()[0]
    
    conn.close()
    
    success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
    
    return {
        'total_images': total_images,
        'total_chars': total_chars,
        'success_rate': success_rate
    }

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='bot_log.txt'
)

logger = logging.getLogger(__name__)

# Replace with your bot token
TOKEN = '7916193700:AAHwh76-fhh8eeyFaHx5PFKrdRgjoaax5is'

# User settings storage
USER_SETTINGS_FILE = 'user_settings.json'
DEFAULT_SETTINGS = {
    'enhancement_level': 'normal',  # normal, high, extreme
    'language': 'ara',  # ara, ara+eng
    'auto_clean': True,
    'preferred_quality': 'balanced'  # balanced, speed, accuracy
}

def load_user_settings():
    try:
        if os.path.exists(USER_SETTINGS_FILE):
            with open(USER_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading user settings: {e}")
    return {}

def save_user_settings(settings):
    try:
        with open(USER_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving user settings: {e}")

# Load settings at startup
user_settings = load_user_settings()

def get_user_settings(user_id):
    str_user_id = str(user_id)
    if str_user_id not in user_settings:
        user_settings[str_user_id] = DEFAULT_SETTINGS.copy()
        save_user_settings(user_settings)
    return user_settings[str_user_id]

def enhance_image_for_arabic(image, contrast=2.0, brightness=1.2):
    """Enhanced image processing specifically for Arabic text."""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for better OCR (minimum 2500px width for Arabic)
        min_width = 3000  # Increased minimum width for better detail
        if image.width < min_width:
            ratio = min_width / image.width
            new_size = (min_width, int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        image = ImageOps.grayscale(image)
        
        # Apply adaptive thresholding
        def adaptive_threshold(img, block_size=35, c=10):
            img_array = np.array(img)
            height, width = img_array.shape
            output = np.zeros_like(img_array)
            
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    block = img_array[i:min(i+block_size, height), j:min(j+block_size, width)]
                    if block.size > 0:
                        threshold = np.mean(block) - c
                        output[i:min(i+block_size, height), j:min(j+block_size, width)] = \
                            (block > threshold) * 255
            
            return Image.fromarray(output)
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        # Increase brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        
        # Apply unsharp mask for better edge definition
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # Apply edge enhancement
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Convert to binary using adaptive thresholding
        image = adaptive_threshold(image)
        
        # Add padding around the image
        border = 50  # Increased border
        image = ImageOps.expand(image, border=border, fill='white')
        
        return image
    except Exception as e:
        logger.error(f"Error in image enhancement: {e}")
        return image

def clean_arabic_text(text):
    """Clean and format Arabic text."""
    try:
        # Remove empty lines and whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Define Arabic text cleaning rules
        def clean_line(line):
            # Fix common OCR errors in Arabic
            replacements = {
                'آ': 'ا',
                'إ': 'ا',
                'أ': 'ا',
                'ة': 'ه',
                '|': 'ا',
                '&': 'و',
                ')': '',
                '(': '',
                ']': '',
                '[': '',
                '}': '',
                '{': '',
                '_': '',
                '،': '،',
                '؛': '؛',
                'ـ': '',
                '١': '1',
                '٢': '2',
                '٣': '3',
                '٤': '4',
                '٥': '5',
                '٦': '6',
                '٧': '7',
                '٨': '8',
                '٩': '9',
                '٠': '0',
                '»': '',
                '«': '',
                '•': '.',
                '..': '.',
                '...': '.',
                '….': '.',
                '  ': ' ',
                ' .': '.',
                '. ': '.',
                ' ،': '،',
                '، ': '،',
                'ى': 'ي',
                'ي0': 'ى',
                'بي': 'ب',
                'فى': 'في',
                'ه ': 'ة ',
                ' ه ': ' ة ',
                'اا': 'ا',
                'الل': 'ال',
                'ااا': 'ا',
                'اه ': 'اة ',
                ' 1 ': ' ',
                ' 0 ': ' ',
                '1 ': '',
                '0 ': '',
                ' 1': '',
                ' 0': '',
                'ظ ': '',
                'ظظ': '',
                'ظظظ': '',
            }
            
            # Apply replacements
            for old, new in replacements.items():
                line = line.replace(old, new)
            
            # Remove repeated characters more than twice
            prev_char = ''
            repeat_count = 0
            cleaned_chars = []
            
            for char in line:
                if char == prev_char:
                    repeat_count += 1
                    if repeat_count < 2:  # Keep maximum two repeated characters
                        cleaned_chars.append(char)
                else:
                    repeat_count = 0
                    cleaned_chars.append(char)
                prev_char = char
            
            line = ''.join(cleaned_chars)
            
            # Remove non-Arabic, non-numeric, and non-punctuation characters
            allowed_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي ،.()0123456789')
            line = ''.join(c for c in line if c in allowed_chars)
            
            # Remove multiple spaces
            line = ' '.join(line.split())
            
            return line
        
        # Clean each line
        cleaned_lines = []
        for line in lines:
            cleaned = clean_line(line)
            if cleaned and not cleaned.isspace():
                # Remove lines that are just numbers or very short
                if not all(c.isdigit() or c.isspace() for c in cleaned) and len(cleaned) > 3:
                    cleaned_lines.append(cleaned)
        
        # Join lines and return
        return '\n'.join(cleaned_lines)
    except Exception as e:
        logger.error(f"Error in text cleaning: {e}")
        return text

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user_id = update.effective_user.id
    log_action(user_id, 'start')
    
    keyboard = [
        [
            InlineKeyboardButton("تعليمات الاستخدام 📖", callback_data='help'),
            InlineKeyboardButton("الإعدادات ⚙️", callback_data='settings')
        ],
        [
            InlineKeyboardButton("معلومات عن البوت ℹ️", callback_data='about'),
            InlineKeyboardButton("إحصائياتي 📊", callback_data='mystats')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        'مرحباً! أنا بوت استخراج النص من الصور. 🔍\n'
        'يمكنني قراءة النصوص العربية والإنجليزية من الصور التي ترسلها لي.\n\n'
        'فقط أرسل لي صورة وسأقوم باستخراج النص منها! 📸✨\n\n'
        'اختر من القائمة أدناه للمزيد من الخيارات:',
        reply_markup=reply_markup
    )

async def show_my_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show user's personal statistics."""
    query = update.callback_query
    user_id = query.from_user.id
    stats = get_user_stats(user_id)
    
    stats_text = (
        'إحصائياتك 📊\n\n'
        f'• عدد الصور المعالجة: {stats["total_images"]} 🖼\n'
        f'• مجموع الأحرف المستخرجة: {stats["total_chars"]} 📝\n'
        f'• نسبة النجاح: {stats["success_rate"]:.1f}% ✨\n\n'
        'استمر في استخدام البوت لتحسين إحصائياتك! 🚀'
    )
    
    keyboard = [[InlineKeyboardButton("رجوع 🔙", callback_data='back_to_main')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(stats_text, reply_markup=reply_markup)

async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process the received image and extract text."""
    user_id = update.effective_user.id
    settings = get_user_settings(user_id)
    
    try:
        # Log the attempt
        log_action(user_id, 'process_image', success=False)  # Will update to True if successful
        
        # Inform user that processing has started
        processing_message = await update.message.reply_text(
            "جاري معالجة الصورة... ⏳\n"
            "قد تستغرق العملية بضع ثوانٍ، يرجى الانتظار."
        )
        
        # Get the photo with the highest resolution
        photo = update.message.photo[-1]
        
        # Download the photo
        try:
            file = await context.bot.get_file(photo.file_id)
            photo_bytes = await file.download_as_bytearray()
        except Exception as e:
            logger.error(f"Error downloading photo: {e}")
            await processing_message.edit_text("عذراً، حدث خطأ أثناء تحميل الصورة. الرجاء المحاولة مرة أخرى. ❌")
            return
        
        # Process image based on user settings
        try:
            image = Image.open(io.BytesIO(photo_bytes))
            
            # Apply enhancements based on user settings
            if settings['enhancement_level'] == 'high':
                # Increase contrast and sharpness for high enhancement
                processed_image = enhance_image_for_arabic(image, contrast=3.0, brightness=1.4)
            elif settings['enhancement_level'] == 'extreme':
                # Maximum enhancement for difficult images
                processed_image = enhance_image_for_arabic(image, contrast=3.5, brightness=1.5)
            else:
                # Normal enhancement
                processed_image = enhance_image_for_arabic(image)
                
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await processing_message.edit_text("عذراً، حدث خطأ أثناء معالجة الصورة. الرجاء المحاولة مرة أخرى. ❌")
            return
        
        # Extract text using pytesseract
        try:
            # Configure OCR based on user settings
            config = f'--oem 1 --psm 6 -l {settings["language"]} --dpi 300'
            
            if settings['preferred_quality'] == 'accuracy':
                # Use multiple OCR passes for better accuracy
                results = []
                psm_modes = [6, 3, 4]  # Different page segmentation modes
                
                for psm in psm_modes:
                    config = f'--oem 1 --psm {psm} -l {settings["language"]} --dpi 300'
                    text = pytesseract.image_to_string(processed_image, config=config)
                    if text.strip():
                        results.append(text)
                
                # Choose the best result
                if results:
                    text = max(results, key=len)
                else:
                    text = ""
            else:
                # Single pass for speed
                text = pytesseract.image_to_string(processed_image, config=config)
            
            # Clean text if auto_clean is enabled
            if settings['auto_clean']:
                text = clean_arabic_text(text)
            
            # Log successful processing
            chars_extracted = len(text)
            log_processed_image(user_id, chars_extracted)
            log_action(user_id, 'process_image', success=True)
            
            if text:
                await processing_message.edit_text(
                    f"النص المستخرج من الصورة ✨:\n\n{text}\n\n"
                    f"عدد الأحرف: {chars_extracted} 📝"
                )
            else:
                await processing_message.edit_text("عذراً، لم أتمكن من العثور على أي نص في هذه الصورة. ❌")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in OCR process: {error_msg}")
            if "tesseract is not installed" in error_msg.lower():
                await processing_message.edit_text("عذراً، برنامج استخراج النص غير مثبت بشكل صحيح. الرجاء التواصل مع مدير البوت. ⚠️")
            else:
                await processing_message.edit_text("عذراً، حدث خطأ أثناء استخراج النص. الرجاء المحاولة مرة أخرى. ❌")
            return
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await update.message.reply_text(
            "عذراً، حدث خطأ غير متوقع. الرجاء المحاولة مرة أخرى. ⚠️"
        )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button presses."""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'help':
        await help_command(update, context, from_button=True)
    elif query.data == 'settings':
        await show_settings(update, context)
    elif query.data == 'about':
        await about_command(update, context, from_button=True)
    elif query.data == 'stats':
        await stats_command(update, context, from_button=True)
    elif query.data.startswith('set_'):
        await handle_settings_change(update, context)
    elif query.data == 'mystats':
        await show_my_stats(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        'كيفية استخدام البوت:\n\n'
        '1. أرسل صورة تحتوي على نص 📸\n'
        '2. انتظر قليلاً ريثما أقوم بمعالجة الصورة 🔄\n'
        '3. سأرسل لك النص المستخرج من الصورة ✨\n\n'
        'للحصول على أفضل النتائج:\n'
        '• تأكد من أن الصورة واضحة وعالية الدقة 📱\n'
        '• تجنب الصور المائلة ↔️\n'
        '• تأكد من أن الإضاءة جيدة 💡\n'
        '• تجنب الخلفيات المعقدة 🎨\n'
        '• حاول أن يكون النص بحجم مناسب 📝\n\n'
        'الأوامر المتاحة:\n'
        '/start - بدء استخدام البوت 🚀\n'
        '/help - عرض هذه المساعدة ❓\n'
        '/settings - تعديل إعدادات البوت ⚙️\n'
        '/about - معلومات عن البوت ℹ️\n'
        '/stats - إحصائيات الاستخدام 📊'
    )
    
    if from_button:
        await update.callback_query.edit_message_text(help_text)
    else:
        await update.message.reply_text(help_text)

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False) -> None:
    """Send information about the bot."""
    about_text = (
        'بوت استخراج النص من الصور 🤖\n\n'
        'يستخدم هذا البوت تقنية التعرف الضوئي على الحروف (OCR) '
        'لاستخراج النصوص العربية والإنجليزية من الصور.\n\n'
        'المميزات:\n'
        '• دعم اللغة العربية والإنجليزية 🌐\n'
        '• معالجة متقدمة للصور 🖼\n'
        '• تنظيف وتحسين النص المستخرج ✨\n'
        '• إعدادات قابلة للتخصيص ⚙️\n\n'
        'إذا واجهتك أي مشكلة أو لديك اقتراحات للتحسين، '
        'يمكنك التواصل مع مطور البوت.'
    )
    
    if from_button:
        await update.callback_query.edit_message_text(about_text)
    else:
        await update.message.reply_text(about_text)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE, from_button=False) -> None:
    """Show bot usage statistics."""
    # You can implement actual statistics tracking here
    stats_text = (
        'إحصائيات استخدام البوت 📊\n\n'
        'سيتم إضافة الإحصائيات قريباً!'
    )
    
    if from_button:
        await update.callback_query.edit_message_text(stats_text)
    else:
        await update.message.reply_text(stats_text)

async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show settings menu."""
    query = update.callback_query
    user_id = query.from_user.id
    settings = get_user_settings(user_id)
    
    keyboard = [
        [
            InlineKeyboardButton(
                f"مستوى التحسين: {settings['enhancement_level']} 🔍",
                callback_data='set_enhancement'
            )
        ],
        [
            InlineKeyboardButton(
                f"اللغة: {settings['language']} 🌐",
                callback_data='set_language'
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅' if settings['auto_clean'] else '❌'} التنظيف التلقائي",
                callback_data='set_auto_clean'
            )
        ],
        [
            InlineKeyboardButton(
                f"نوعية الصورة: {settings['preferred_quality']} 📸",
                callback_data='set_quality'
            )
        ],
        [
            InlineKeyboardButton("رجوع 🔙", callback_data='back_to_main')
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        'إعدادات البوت ⚙️\n\n'
        'اختر الإعداد الذي تريد تغييره:',
        reply_markup=reply_markup
    )

async def handle_settings_change(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle settings changes."""
    query = update.callback_query
    user_id = query.from_user.id
    settings = get_user_settings(user_id)
    
    if query.data == 'set_enhancement':
        current = settings['enhancement_level']
        settings['enhancement_level'] = {
            'normal': 'high',
            'high': 'extreme',
            'extreme': 'normal'
        }[current]
    elif query.data == 'set_language':
        settings['language'] = 'ara+eng' if settings['language'] == 'ara' else 'ara'
    elif query.data == 'set_auto_clean':
        settings['auto_clean'] = not settings['auto_clean']
    elif query.data == 'set_quality':
        settings['preferred_quality'] = 'speed' if settings['preferred_quality'] == 'balanced' else 'balanced'
    elif query.data == 'back_to_main':
        await start(update, context)
        return
    
    user_settings[str(user_id)] = settings
    save_user_settings(user_settings)
    await show_settings(update, context)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a telegram message to notify the developer."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = ''.join(tb_list)
    logger.error(f"Full traceback: {tb_string}")

def main() -> None:
    """Start the bot."""
    try:
        # Initialize database
        init_database()
        
        # Create the Application
        app = Application.builder().token(TOKEN).build()

        # Add handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("about", about_command))
        app.add_handler(CallbackQueryHandler(button_handler))
        app.add_handler(MessageHandler(filters.PHOTO, process_image))
        
        # Add error handler
        app.add_error_handler(error_handler)

        # Start the Bot
        logger.info("Starting bot...")
        app.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
