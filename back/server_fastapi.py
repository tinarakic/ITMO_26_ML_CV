# model_server_fastapi.py
import io
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Конфигурация
# ----------------------------
ONNX_MODEL_PATH = "/app/models/hybrid_model.onnx"
INPUT_SIZE = (224, 224)
CLASS_NAMES = {0: "real", 1: "ai_generated"}  # 0 = Real, 1 = Fake (AI)

# ----------------------------
# Загрузка ONNX модели
# ----------------------------
try:
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logger.info(f"ONNX модель загружена. Input: {input_name}, Output: {output_name}")
except Exception as e:
    logger.error(f"Ошибка загрузки ONNX модели: {e}")
    raise

# ----------------------------
# Функции предобработки
# ----------------------------
def compute_fft(image: np.ndarray) -> np.ndarray:
    """
    Вычисление преобразования Фурье для изображения (одноканальный выход).
    Ожидается RGB изображение (H, W, 3) в формате uint8 или float.
    Возвращает magnitude спектра (H, W) float32.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    fft = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft))
    return magnitude.astype(np.float32)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Предобработка изображения для подачи в ONNX модель.
    Возвращает numpy массив формы (1, 4, 224, 224) float32.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Невозможно прочитать изображение: {e}")

    img_np = np.array(img)  # uint8, (H, W, 3)

    # Вычисление FFT (ожидается 2D массив: H, W)
    fft = compute_fft(img_np)  # (H, W) float32

    # Ресайз RGB и FFT до 224x224
    img_resized = cv2.resize(img_np, INPUT_SIZE)          # (224, 224, 3)
    fft_resized = cv2.resize(fft, INPUT_SIZE)             # может быть (224, 224) или (224, 224, 1)

    # Приведение fft_resized к строго 2D массиву (224, 224)
    if fft_resized.ndim == 3:
        # Если последняя ось имеет размер 1, просто удаляем её
        if fft_resized.shape[-1] == 1:
            fft_resized = fft_resized.squeeze(axis=-1)
        else:
            # Если вдруг размер > 1 (что маловероятно), берём первый канал
            fft_resized = fft_resized[:, :, 0]
    # Теперь fft_resized гарантированно 2D

    # Преобразование типов и добавление канала для FFT
    img_resized = img_resized.astype(np.float32)                  # (224, 224, 3)
    fft_resized = fft_resized[..., np.newaxis].astype(np.float32) # (224, 224, 1)

    # Конкатенация по каналам: RGB + FFT → 4 канала
    combined = np.concatenate([img_resized, fft_resized], axis=2) # (224, 224, 4)

    # Транспонирование в формат (C, H, W) и добавление batch измерения
    combined = np.transpose(combined, (2, 0, 1))  # (4, 224, 224)
    combined = np.expand_dims(combined, axis=0)   # (1, 4, 224, 224)

    return combined.astype(np.float32)

# ----------------------------
# FastAPI приложение
# ----------------------------
app = FastAPI(title="AI Image Detector API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Эндпоинт для классификации изображения.
    Возвращает предсказание ('real' или 'ai_generated') и уверенность модели.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    try:
        # Чтение байтов
        image_bytes = await file.read()
        
        # Предобработка
        input_tensor = preprocess_image(image_bytes)
        
        # Инференс ONNX
        outputs = session.run([output_name], {input_name: input_tensor})
        logits = outputs[0]  # (1, num_classes)
        
        # Softmax для вероятностей
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        pred_idx = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0, pred_idx])
        
        prediction = CLASS_NAMES.get(pred_idx, "unknown")
        
        response = {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prediction: {prediction}, confidence: {confidence:.4f}, file: {file.filename}")
        return response
        
    except ValueError as ve:
        logger.error(f"Ошибка валидации: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Внутренняя ошибка сервера: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки изображения")

if __name__ == "__main__":
    import uvicorn
    logger.info("Запуск FastAPI сервера на http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")