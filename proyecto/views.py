from base64 import b64decode, b64encode
from functools import lru_cache
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
from urllib.request import urlopen

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

DATASET_HANDLE = 'grassknoted/asl-alphabet'
DATASET_DIR = Path(__file__).resolve().parent / 'datasets' / 'asl_alphabet'
TRAIN_DIR = DATASET_DIR / 'asl_alphabet_train' / 'asl_alphabet_train'
MODEL_DIR = Path(__file__).resolve().parent / 'models'
IMAGE_MODEL_PATH = MODEL_DIR / 'asl_alphabet_mobilenet_v2.keras'
IMAGE_CLASS_NAMES_PATH = MODEL_DIR / 'asl_alphabet_class_names.json'
LANDMARK_MODEL_PATH = MODEL_DIR / 'asl_landmarks_mlp.keras'
LANDMARK_METADATA_PATH = MODEL_DIR / 'asl_landmarks_metadata.json'
LANDMARK_DATA_CACHE_PATH = MODEL_DIR / 'asl_landmarks_dataset_v1.npz'
HAND_LANDMARKER_MODEL_PATH = MODEL_DIR / 'hand_landmarker.task'
HAND_LANDMARKER_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
)
MEDIAPIPE_CACHE_DIR = Path(__file__).resolve().parent / '.cache' / 'matplotlib'
MEDIAPIPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MEDIAPIPE_CACHE_DIR))

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64
BASE_TRAIN_EPOCHS = 3
FINE_TUNE_EPOCHS = 1
VALIDATION_SPLIT = 0.1
PREDICTION_THRESHOLD = 0.72
HAND_MARGIN_RATIO = 0.30
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.45
LANDMARK_FEATURE_SIZE = 63
LANDMARK_CACHE_VERSION = 1
LANDMARK_TRAIN_EPOCHS = 24
LANDMARK_BATCH_SIZE = 512
LANDMARK_MODEL_WEIGHT = 0.78
IMAGE_MODEL_WEIGHT = 0.22
LANDMARK_EXCLUDED_CLASSES = {'nothing'}
NO_HAND_LABEL = 'no_hand'
IMAGE_MODEL_MIN_BYTES = 100000
LANDMARK_MODEL_MIN_BYTES = 50000

SPECIAL_ACTIONS = {
    'del': 'delete',
    'nothing': 'ignore',
    'space': 'space',
}
DISPLAY_LABELS = {
    'del': 'BORRAR',
    'no_hand': 'SIN MANO',
    'nothing': 'SIN SENA',
    'space': 'ESPACIO',
}


def main(request):
    return render(request, 'index.html', build_home_context())


def prediccion(request):
    context = build_home_context()
    if request.method != 'POST':
        return render(request, 'index.html', context)

    if not settings.TRAINING_ENABLED:
        context['error_message'] = (
            'En este despliegue el entrenamiento esta desactivado. '
            'La version de Render esta preparada solo para inferencia.'
        )
        return render(request, 'index.html', context)

    try:
        ensure_image_dataset()
        ensure_hand_landmarker_model()
        image_model = get_or_train_image_model()
        landmark_model = get_or_train_landmark_model()
        landmark_dataset = load_landmark_dataset()
        global_class_names = load_class_names()

        eval_result = landmark_model.evaluate(
            landmark_dataset['val_x'],
            landmark_dataset['val_y'],
            verbose=0,
        )
        loss = float(eval_result[0])
        accuracy = float(eval_result[1])

        preview_examples = build_preview_examples(
            landmark_dataset['supported_class_names'],
            sample_size=12,
        )

        image_results = []
        predicted_tokens = []
        actual_tokens = []
        for idx, example in enumerate(preview_examples, start=1):
            predicted_label, confidence, metadata = predict_pil_image(
                example['image'],
                image_model=image_model,
                landmark_model=landmark_model,
                global_class_names=global_class_names,
                supported_class_names=landmark_dataset['supported_class_names'],
            )
            actual_label = example['actual_label']
            image_results.append({
                'id': idx,
                'image': pil_image_to_base64(example['image']),
                'actual': pretty_label(actual_label),
                'predicted': pretty_label(predicted_label),
                'confidence': f'{confidence:.2f}',
                'mode': metadata['prediction_mode'],
            })
            predicted_tokens.append(format_preview_token(predicted_label))
            actual_tokens.append(format_preview_token(actual_label))

        train_detected = int(landmark_dataset['train_detected_count'])
        train_total = int(landmark_dataset['train_total_count'])
        val_detected = int(landmark_dataset['val_detected_count'])
        val_total = int(landmark_dataset['val_total_count'])
        train_coverage = train_detected / train_total if train_total else 0.0
        val_coverage = val_detected / val_total if val_total else 0.0

        context.update({
            'results': image_results,
            'accuracy': f'{accuracy:.4f}',
            'loss': f'{loss:.4f}',
            'sample_count': int(landmark_dataset['val_x'].shape[0]),
            'preview_count': len(image_results),
            'train_count': int(landmark_dataset['train_x'].shape[0]),
            'predicted_word': ' '.join(predicted_tokens),
            'actual_word': ' '.join(actual_tokens),
            'status_message': (
                'Modelo hibrido listo. Se uso el clasificador por landmarks de MediaPipe '
                f'y la CNN de apoyo. Cobertura de landmarks: train {train_detected}/{train_total} '
                f'({train_coverage:.1%}) y validacion {val_detected}/{val_total} ({val_coverage:.1%}).'
            ),
        })
    except Exception as exc:
        context['error_message'] = str(exc)

    return render(request, 'index.html', context)

@csrf_exempt
def camera_predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    try:
        payload = json.loads(request.body.decode('utf-8'))
        image_data = payload.get('image')
        if not image_data:
            raise ValueError('No se recibio ninguna imagen para procesar.')

        image_model = get_or_train_image_model()
        landmark_model = get_or_train_landmark_model()
        predicted_label, confidence, metadata = predict_camera_image(
            image_data=image_data,
            image_model=image_model,
            landmark_model=landmark_model,
        )
        display = pretty_label(predicted_label)
        action = resolve_prediction_action(
            predicted_label,
            confidence,
            hand_detected=metadata['hand_detected'],
        )
        token = build_action_token(
            predicted_label,
            confidence,
            hand_detected=metadata['hand_detected'],
        )

        return JsonResponse({
            'predicted': display,
            'raw_label': predicted_label,
            'confidence': f'{confidence:.2f}',
            'action': action,
            'token': token,
            'hand_detected': metadata['hand_detected'],
            'preprocess_mode': metadata['preprocess_mode'],
            'prediction_mode': metadata['prediction_mode'],
            'landmark_label': metadata.get('landmark_label', ''),
            'landmark_confidence': metadata.get('landmark_confidence', 0.0),
            'image_label': metadata.get('image_label', ''),
            'image_confidence': metadata.get('image_confidence', 0.0),
            'handedness': metadata.get('handedness', 'Unknown'),
        })
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=400)


def build_home_context():
    dataset_ready = dataset_is_ready() if settings.TRAINING_ENABLED else False
    image_model_ready = model_file_looks_ready(IMAGE_MODEL_PATH, IMAGE_MODEL_MIN_BYTES) and IMAGE_CLASS_NAMES_PATH.exists()
    landmark_model_ready = model_file_looks_ready(LANDMARK_MODEL_PATH, LANDMARK_MODEL_MIN_BYTES) and LANDMARK_METADATA_PATH.exists()
    stats = get_dataset_stats() if dataset_ready else None
    return {
        'dataset_ready': dataset_ready,
        'model_ready': image_model_ready and landmark_model_ready,
        'image_model_ready': image_model_ready,
        'landmark_model_ready': landmark_model_ready,
        'dataset_name': DATASET_HANDLE,
        'dataset_classes': stats['classes'] if stats else 0,
        'dataset_total_images': stats['images'] if stats else 0,
        'train_count': stats['train_images'] if stats else 0,
        'sample_count': stats['validation_images'] if stats else 0,
        'prediction_threshold': f'{PREDICTION_THRESHOLD:.2f}',
        'training_enabled': settings.TRAINING_ENABLED,
    }


def dataset_is_ready():
    return TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir())


@lru_cache(maxsize=1)
def get_dataset_stats():
    ensure_image_dataset()
    class_dirs = sorted([path for path in TRAIN_DIR.iterdir() if path.is_dir()])
    total_images = 0
    for class_dir in class_dirs:
        total_images += sum(1 for _ in class_dir.glob('*'))

    return {
        'classes': len(class_dirs),
        'images': total_images,
        'train_images': int(total_images * (1 - VALIDATION_SPLIT)),
        'validation_images': int(total_images * VALIDATION_SPLIT),
    }


def import_ml_dependencies():
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            'Faltan dependencias de machine learning. Activa .venv312 e instala requirements.txt.'
        ) from exc

    return np, tf


def import_image_dependencies():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            'Falta Pillow. Instala las dependencias del proyecto antes de usar la camara.'
        ) from exc

    return Image, ImageOps


def import_mediapipe_dependencies():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            'Falta MediaPipe. Activa .venv312 e instala requirements.txt para usar landmarks.'
        ) from exc

    return mp


def ensure_hand_landmarker_model():
    if HAND_LANDMARKER_MODEL_PATH.exists():
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(HAND_LANDMARKER_MODEL_URL) as response:
            HAND_LANDMARKER_MODEL_PATH.write_bytes(response.read())
    except Exception as exc:
        if HAND_LANDMARKER_MODEL_PATH.exists():
            HAND_LANDMARKER_MODEL_PATH.unlink()
        raise RuntimeError(
            'No fue posible descargar el modelo de deteccion de mano de MediaPipe.'
        ) from exc


def ensure_image_dataset():
    if dataset_is_ready():
        return

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            'Falta kagglehub para descargar el dataset grande de Kaggle.'
        ) from exc

    try:
        kagglehub.dataset_download(
            DATASET_HANDLE,
            output_dir=str(DATASET_DIR),
            force_download=False,
        )
    except Exception as exc:
        raise RuntimeError(
            'No fue posible descargar el dataset de Kaggle. Verifica tu conexion o vuelve a intentar.'
        ) from exc

    if not dataset_is_ready():
        raise RuntimeError(
            'La descarga termino, pero no se encontro la carpeta esperada del dataset ASL.'
        )


def build_training_datasets(tf):
    ensure_image_dataset()
    common_kwargs = {
        'directory': str(TRAIN_DIR),
        'seed': 123,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'label_mode': 'categorical',
        'validation_split': VALIDATION_SPLIT,
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset='training',
        shuffle=True,
        **common_kwargs,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset='validation',
        shuffle=False,
        **common_kwargs,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds


def model_file_looks_ready(model_path, min_bytes):
    return model_path.exists() and model_path.stat().st_size >= min_bytes


def load_saved_keras_model(tf, model_path, min_bytes):
    if not model_file_looks_ready(model_path, min_bytes):
        if model_path.exists():
            try:
                model_path.unlink()
            except OSError:
                pass
        return None

    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
        try:
            model_path.unlink()
        except OSError:
            pass
        return None


@lru_cache(maxsize=1)
def get_or_train_image_model():
    _, tf = import_ml_dependencies()
    class_names = load_class_names()

    loaded_model = load_saved_keras_model(tf, IMAGE_MODEL_PATH, IMAGE_MODEL_MIN_BYTES)
    if loaded_model is not None:
        return loaded_model

    if not settings.TRAINING_ENABLED:
        raise RuntimeError(
            'El modelo de imagen no esta disponible en producción y el entrenamiento esta desactivado.'
        )

    ensure_image_dataset()

    train_ds, val_ds = build_training_datasets(tf)
    model = build_transfer_model(tf, len(class_names))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=IMAGE_MODEL_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=BASE_TRAIN_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    feature_extractor = next(
        (layer for layer in model.layers if layer.name.startswith('mobilenetv2')),
        None,
    )
    if feature_extractor is not None:
        feature_extractor.trainable = True
        for layer in feature_extractor.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=BASE_TRAIN_EPOCHS,
            epochs=BASE_TRAIN_EPOCHS + FINE_TUNE_EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )

    save_class_names(class_names)
    return tf.keras.models.load_model(IMAGE_MODEL_PATH)


def build_transfer_model(tf, num_classes):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name='augmentation',
    )

    feature_extractor = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35,
    )
    feature_extractor.trainable = False
    feature_extractor._name = 'feature_extractor'

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = feature_extractor(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


@lru_cache(maxsize=1)
def get_or_train_landmark_model():
    _, tf = import_ml_dependencies()

    loaded_model = load_saved_keras_model(tf, LANDMARK_MODEL_PATH, LANDMARK_MODEL_MIN_BYTES)
    if loaded_model is not None:
        return loaded_model

    if not settings.TRAINING_ENABLED:
        raise RuntimeError(
            'El modelo de landmarks no esta disponible en producción y el entrenamiento esta desactivado.'
        )

    landmark_dataset = load_landmark_dataset()

    model = build_landmark_model(tf, len(landmark_dataset['supported_class_names']))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_landmark_metadata(landmark_dataset['supported_class_names'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=LANDMARK_MODEL_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
        ),
    ]

    model.fit(
        landmark_dataset['train_x'],
        landmark_dataset['train_y'],
        validation_data=(landmark_dataset['val_x'], landmark_dataset['val_y']),
        epochs=LANDMARK_TRAIN_EPOCHS,
        batch_size=LANDMARK_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    return tf.keras.models.load_model(LANDMARK_MODEL_PATH)


def build_landmark_model(tf, num_classes):
    inputs = tf.keras.Input(shape=(LANDMARK_FEATURE_SIZE,))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def load_landmark_dataset():
    np, _ = import_ml_dependencies()
    current_class_names = load_class_names()
    current_supported = get_supported_landmark_class_names(current_class_names)

    if LANDMARK_DATA_CACHE_PATH.exists():
        with np.load(LANDMARK_DATA_CACHE_PATH, allow_pickle=False) as cached:
            cache_version = int(cached['cache_version'][0])
            cached_supported = list(cached['supported_class_names'].tolist())
            cached_global = list(cached['global_class_names'].tolist())
            if (
                cache_version == LANDMARK_CACHE_VERSION
                and cached_supported == current_supported
                and cached_global == current_class_names
            ):
                return {
                    'train_x': cached['train_x'].astype('float32'),
                    'train_y': cached['train_y'].astype('int32'),
                    'val_x': cached['val_x'].astype('float32'),
                    'val_y': cached['val_y'].astype('int32'),
                    'val_paths': cached['val_paths'].tolist(),
                    'supported_class_names': cached_supported,
                    'global_class_names': cached_global,
                    'train_detected_count': int(cached['train_detected_count'][0]),
                    'train_total_count': int(cached['train_total_count'][0]),
                    'val_detected_count': int(cached['val_detected_count'][0]),
                    'val_total_count': int(cached['val_total_count'][0]),
                }

    return build_landmark_dataset(current_class_names, current_supported)


def build_landmark_dataset(global_class_names, supported_class_names):
    np, _ = import_ml_dependencies()
    Image, _ = import_image_dependencies()
    ensure_image_dataset()
    ensure_hand_landmarker_model()

    rng = random.Random(123)
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    val_paths = []
    train_detected_count = 0
    train_total_count = 0
    val_detected_count = 0
    val_total_count = 0

    for class_index, class_name in enumerate(supported_class_names):
        class_dir = TRAIN_DIR / class_name
        image_paths = sorted(
            [path for path in class_dir.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        )
        rng.shuffle(image_paths)
        split_index = int(len(image_paths) * (1 - VALIDATION_SPLIT))
        split_index = max(1, min(split_index, len(image_paths) - 1))
        train_paths = image_paths[:split_index]
        eval_paths = image_paths[split_index:]

        for image_path in train_paths:
            train_total_count += 1
            image = Image.open(image_path).convert('RGB')
            analysis = analyze_hand_image(image)
            image.close()
            if not analysis['hand_detected']:
                continue

            train_detected_count += 1
            train_x.append(analysis['feature_vector'])
            train_y.append(class_index)

        for image_path in eval_paths:
            val_total_count += 1
            image = Image.open(image_path).convert('RGB')
            analysis = analyze_hand_image(image)
            image.close()
            if not analysis['hand_detected']:
                continue

            val_detected_count += 1
            val_x.append(analysis['feature_vector'])
            val_y.append(class_index)
            val_paths.append(str(image_path))

    if not train_x or not val_x:
        raise RuntimeError(
            'No fue posible construir el dataset de landmarks. MediaPipe no detecto suficientes manos.'
        )

    train_x = np.asarray(train_x, dtype='float32')
    train_y = np.asarray(train_y, dtype='int32')
    val_x = np.asarray(val_x, dtype='float32')
    val_y = np.asarray(val_y, dtype='int32')

    np.savez_compressed(
        LANDMARK_DATA_CACHE_PATH,
        cache_version=np.asarray([LANDMARK_CACHE_VERSION], dtype='int32'),
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        val_paths=np.asarray(val_paths),
        supported_class_names=np.asarray(supported_class_names),
        global_class_names=np.asarray(global_class_names),
        train_detected_count=np.asarray([train_detected_count], dtype='int32'),
        train_total_count=np.asarray([train_total_count], dtype='int32'),
        val_detected_count=np.asarray([val_detected_count], dtype='int32'),
        val_total_count=np.asarray([val_total_count], dtype='int32'),
    )
    save_landmark_metadata(supported_class_names)
    return {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'val_paths': val_paths,
        'supported_class_names': list(supported_class_names),
        'global_class_names': list(global_class_names),
        'train_detected_count': train_detected_count,
        'train_total_count': train_total_count,
        'val_detected_count': val_detected_count,
        'val_total_count': val_total_count,
    }


def get_supported_landmark_class_names(global_class_names=None):
    if global_class_names is None:
        global_class_names = load_class_names()
    return [class_name for class_name in global_class_names if class_name not in LANDMARK_EXCLUDED_CLASSES]


def load_landmark_metadata():
    if LANDMARK_METADATA_PATH.exists():
        return json.loads(LANDMARK_METADATA_PATH.read_text(encoding='utf-8'))

    supported_class_names = get_supported_landmark_class_names()
    metadata = {
        'supported_class_names': supported_class_names,
        'cache_version': LANDMARK_CACHE_VERSION,
    }
    save_landmark_metadata(supported_class_names)
    return metadata


def save_landmark_metadata(supported_class_names):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LANDMARK_METADATA_PATH.write_text(
        json.dumps(
            {
                'supported_class_names': list(supported_class_names),
                'cache_version': LANDMARK_CACHE_VERSION,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding='utf-8',
    )


@lru_cache(maxsize=1)
def get_hand_detector():
    mp = import_mediapipe_dependencies()
    ensure_hand_landmarker_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def predict_camera_image(image_data, image_model, landmark_model):
    Image, _ = import_image_dependencies()
    encoded = image_data.split(',', 1)[1] if ',' in image_data else image_data
    image = Image.open(BytesIO(b64decode(encoded))).convert('RGB')
    try:
        return predict_pil_image(image, image_model=image_model, landmark_model=landmark_model)
    finally:
        image.close()


def predict_pil_image(
    image,
    image_model=None,
    landmark_model=None,
    global_class_names=None,
    supported_class_names=None,
):
    np, _ = import_ml_dependencies()
    global_class_names = global_class_names or load_class_names()
    supported_class_names = supported_class_names or load_landmark_metadata()['supported_class_names']

    analysis = analyze_hand_image(image)
    if not analysis['hand_detected']:
        return NO_HAND_LABEL, 0.0, {
            'hand_detected': False,
            'preprocess_mode': 'no-hand-detected',
            'prediction_mode': 'no-hand-detected',
            'image_variant': 'none',
        }

    if landmark_model is None:
        landmark_model = get_or_train_landmark_model()
    if image_model is None:
        image_model = get_or_train_image_model()

    landmark_prediction = landmark_model.predict(
        analysis['feature_vector'].reshape(1, -1),
        verbose=0,
    )[0]
    landmark_probs = expand_landmark_probabilities(
        landmark_prediction,
        supported_class_names,
        global_class_names,
    )
    image_probs, image_variant = build_image_probability_vector(
        image_model,
        analysis['crop_image'],
    )

    nothing_index = (
        global_class_names.index('nothing')
        if 'nothing' in global_class_names
        else None
    )
    landmark_probs = normalize_probability_vector(landmark_probs)
    image_probs = normalize_probability_vector(image_probs)
    combined_probs = (
        LANDMARK_MODEL_WEIGHT * landmark_probs
        + IMAGE_MODEL_WEIGHT * image_probs
    )
    if nothing_index is not None:
        combined_probs[nothing_index] = 0.0
    combined_probs = normalize_probability_vector(combined_probs)

    predicted_index = int(np.argmax(combined_probs))
    landmark_index = int(np.argmax(landmark_probs))
    image_index = int(np.argmax(image_probs))
    predicted_label = global_class_names[predicted_index]

    return predicted_label, float(combined_probs[predicted_index]), {
        'hand_detected': True,
        'preprocess_mode': 'mediapipe-landmarks',
        'prediction_mode': 'hybrid-landmarks-image',
        'image_variant': image_variant,
        'landmark_label': global_class_names[landmark_index],
        'landmark_confidence': float(landmark_probs[landmark_index]),
        'image_label': global_class_names[image_index],
        'image_confidence': float(image_probs[image_index]),
        'handedness': analysis['handedness'],
    }


def analyze_hand_image(image):
    np, _ = import_ml_dependencies()
    mp = import_mediapipe_dependencies()
    detector = get_hand_detector()

    image_array = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    results = detector.detect(mp_image)
    if not results.hand_landmarks:
        return {
            'hand_detected': False,
            'crop_image': None,
            'feature_vector': None,
            'handedness': 'Unknown',
        }

    best_index = select_best_hand_index(results.hand_landmarks)
    hand_landmarks = results.hand_landmarks[best_index]
    handedness = 'Unknown'
    if results.handedness and len(results.handedness) > best_index and results.handedness[best_index]:
        handedness = results.handedness[best_index][0].category_name

    crop_image = build_hand_crop_from_landmarks(image, hand_landmarks)
    feature_vector = compute_landmark_feature_vector(hand_landmarks)
    if crop_image is None or feature_vector is None:
        return {
            'hand_detected': False,
            'crop_image': None,
            'feature_vector': None,
            'handedness': handedness,
        }

    return {
        'hand_detected': True,
        'crop_image': crop_image,
        'feature_vector': feature_vector,
        'handedness': handedness,
    }


def select_best_hand_index(hand_landmarks_list):
    best_index = 0
    best_area = -1.0
    for index, hand_landmarks in enumerate(hand_landmarks_list):
        xs = [landmark.x for landmark in hand_landmarks]
        ys = [landmark.y for landmark in hand_landmarks]
        area = max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best_index = index
    return best_index


def build_hand_crop_from_landmarks(image, hand_landmarks):
    width, height = image.size
    xs = [landmark.x for landmark in hand_landmarks]
    ys = [landmark.y for landmark in hand_landmarks]
    min_x = max(0.0, min(xs))
    max_x = min(1.0, max(xs))
    min_y = max(0.0, min(ys))
    max_y = min(1.0, max(ys))

    box_width = max(1.0, (max_x - min_x) * width)
    box_height = max(1.0, (max_y - min_y) * height)
    center_x = ((min_x + max_x) / 2.0) * width
    center_y = ((min_y + max_y) / 2.0) * height
    side = int(max(box_width, box_height) * (1.0 + HAND_MARGIN_RATIO * 2.0))
    side = max(1, min(side, width, height))
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))


def compute_landmark_feature_vector(hand_landmarks):
    np, _ = import_ml_dependencies()
    coords = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
        dtype='float32',
    )
    if coords.shape != (21, 3):
        return None

    coords = coords - coords[0]
    reference_xy = coords[9, :2]
    reference_norm = float(np.linalg.norm(reference_xy))
    if reference_norm < 1e-6:
        return None

    angle = math.atan2(float(reference_xy[1]), float(reference_xy[0]))
    rotation = (-math.pi / 2.0) - angle
    cos_value = math.cos(rotation)
    sin_value = math.sin(rotation)
    rotated_x = coords[:, 0] * cos_value - coords[:, 1] * sin_value
    rotated_y = coords[:, 0] * sin_value + coords[:, 1] * cos_value
    coords[:, 0] = rotated_x
    coords[:, 1] = rotated_y

    if coords[5, 0] > coords[17, 0]:
        coords[:, 0] *= -1.0

    scale = float(np.max(np.linalg.norm(coords, axis=1)))
    if scale < 1e-6:
        return None

    coords /= scale
    return coords.reshape(-1).astype('float32')


def expand_landmark_probabilities(prediction, supported_class_names, global_class_names):
    np, _ = import_ml_dependencies()
    expanded = np.zeros(len(global_class_names), dtype='float32')
    for index, class_name in enumerate(supported_class_names):
        expanded[global_class_names.index(class_name)] = float(prediction[index])
    return expanded


def build_image_probability_vector(image_model, crop_image):
    np, _ = import_ml_dependencies()
    Image, ImageOps = import_image_dependencies()

    normalized = ImageOps.fit(crop_image, IMAGE_SIZE, method=Image.Resampling.LANCZOS)
    mirrored = ImageOps.mirror(normalized)
    candidates = [
        ('original', normalized),
        ('mirrored', mirrored),
    ]
    batch = np.stack(
        [np.asarray(candidate_image).astype('float32') for _, candidate_image in candidates],
        axis=0,
    )
    predictions = image_model.predict(batch, verbose=0)
    best_index = max(
        range(len(candidates)),
        key=lambda candidate_index: float(np.max(predictions[candidate_index])),
    )
    return predictions[best_index].astype('float32'), candidates[best_index][0]


def normalize_probability_vector(probabilities):
    np, _ = import_ml_dependencies()
    total = float(np.sum(probabilities))
    if total <= 0.0:
        return np.zeros_like(probabilities)
    return probabilities / total


def build_preview_examples(class_names, sample_size=12):
    Image, _ = import_image_dependencies()
    rng = random.Random(123)
    selected_classes = class_names[:]
    rng.shuffle(selected_classes)
    selected_classes = selected_classes[:min(sample_size, len(selected_classes))]

    examples = []
    for class_name in selected_classes:
        class_dir = TRAIN_DIR / class_name
        image_paths = sorted(
            [path for path in class_dir.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        )
        if not image_paths:
            continue

        image_path = rng.choice(image_paths)
        image = Image.open(image_path).convert('RGB')
        examples.append({
            'path': str(image_path),
            'actual_label': class_name,
            'image': image.copy(),
        })
        image.close()

    return examples


def pil_image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    encoded = b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'


def load_class_names():
    if IMAGE_CLASS_NAMES_PATH.exists():
        return json.loads(IMAGE_CLASS_NAMES_PATH.read_text(encoding='utf-8'))

    if not settings.TRAINING_ENABLED:
        raise RuntimeError(
            'No se encontro asl_alphabet_class_names.json y el entrenamiento esta desactivado.'
        )

    ensure_image_dataset()
    class_names = sorted([path.name for path in TRAIN_DIR.iterdir() if path.is_dir()])
    save_class_names(class_names)
    return class_names


def save_class_names(class_names):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CLASS_NAMES_PATH.write_text(
        json.dumps(list(class_names), ensure_ascii=True, indent=2),
        encoding='utf-8',
    )


def resolve_prediction_action(label, confidence, hand_detected=True):
    if not hand_detected or label == NO_HAND_LABEL:
        return 'ignore'
    if confidence < PREDICTION_THRESHOLD:
        return 'ignore'
    return SPECIAL_ACTIONS.get(label, 'append')


def build_action_token(label, confidence, hand_detected=True):
    if not hand_detected or label == NO_HAND_LABEL:
        return ''
    if confidence < PREDICTION_THRESHOLD:
        return ''
    if label == 'space':
        return ' '
    if label in {'del', 'nothing'}:
        return ''
    return label


def pretty_label(label):
    return DISPLAY_LABELS.get(label, label)


def format_preview_token(label):
    if label == 'space':
        return '<espacio>'
    if label == 'del':
        return '<borrar>'
    if label == NO_HAND_LABEL:
        return '<sin-mano>'
    if label == 'nothing':
        return '<sin-sena>'
    return label
