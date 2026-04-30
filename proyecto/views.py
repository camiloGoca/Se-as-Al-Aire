from base64 import b64decode, b64encode
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import random

from django.http import JsonResponse
from django.shortcuts import render

DATASET_HANDLE = 'grassknoted/asl-alphabet'
DATASET_DIR = Path(__file__).resolve().parent / 'datasets' / 'asl_alphabet'
TRAIN_DIR = DATASET_DIR / 'asl_alphabet_train' / 'asl_alphabet_train'
MODEL_DIR = Path(__file__).resolve().parent / 'models'
MODEL_PATH = MODEL_DIR / 'asl_alphabet_mobilenet_v2.keras'
CLASS_NAMES_PATH = MODEL_DIR / 'asl_alphabet_class_names.json'
IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64
BASE_TRAIN_EPOCHS = 3
FINE_TUNE_EPOCHS = 1
VALIDATION_SPLIT = 0.1
PREDICTION_THRESHOLD = 0.72
SPECIAL_ACTIONS = {
    'del': 'delete',
    'nothing': 'ignore',
    'space': 'space',
}
DISPLAY_LABELS = {
    'del': 'BORRAR',
    'nothing': 'SIN SENA',
    'space': 'ESPACIO',
}


def main(request):
    return render(request, 'index.html', build_home_context())


def prediccion(request):
    context = build_home_context()
    if request.method != 'POST':
        return render(request, 'index.html', context)

    try:
        np, tf = import_ml_dependencies()
        ensure_image_dataset()
        stats = get_dataset_stats()
        train_ds, val_ds, class_names = build_training_datasets(tf)
        model = get_or_train_model()

        loss, accuracy = model.evaluate(val_ds, verbose=0)
        preview_images, preview_labels = build_preview_samples(class_names, sample_size=12)
        predictions = model.predict(preview_images, verbose=0)

        image_results = []
        predicted_tokens = []
        actual_tokens = []
        for idx, prediction in enumerate(predictions, start=1):
            predicted_index = int(np.argmax(prediction))
            actual_index = int(np.argmax(preview_labels[idx - 1]))
            predicted_label = class_names[predicted_index]
            actual_label = class_names[actual_index]

            image_results.append({
                'id': idx,
                'image': image_to_base64(preview_images[idx - 1]),
                'actual': pretty_label(actual_label),
                'predicted': pretty_label(predicted_label),
                'confidence': f'{float(np.max(prediction)):.2f}',
            })
            predicted_tokens.append(format_preview_token(predicted_label))
            actual_tokens.append(format_preview_token(actual_label))

        context.update({
            'results': image_results,
            'accuracy': f'{accuracy:.4f}',
            'loss': f'{loss:.4f}',
            'sample_count': stats['validation_images'],
            'preview_count': len(image_results),
            'train_count': stats['train_images'],
            'predicted_word': ' '.join(predicted_tokens),
            'actual_word': ' '.join(actual_tokens),
            'status_message': (
                'Modelo listo. Se entrenó o cargó el clasificador y se evaluaron '
                f"{stats['validation_images']} imágenes de validación. "
                f"Abajo solo se muestran {len(image_results)} ejemplos aleatorios."
            ),
        })
    except Exception as exc:
        context['error_message'] = str(exc)

    return render(request, 'index.html', context)


def camera_predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    try:
        payload = json.loads(request.body.decode('utf-8'))
        image_data = payload.get('image')
        if not image_data:
            raise ValueError('No se recibió ninguna imagen para procesar.')

        model = get_or_train_model()
        predicted_label, confidence = predict_camera_image(model, image_data)
        display = pretty_label(predicted_label)
        action = resolve_prediction_action(predicted_label, confidence)
        token = build_action_token(predicted_label, confidence)

        return JsonResponse({
            'predicted': display,
            'raw_label': predicted_label,
            'confidence': f'{confidence:.2f}',
            'action': action,
            'token': token,
        })
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=400)


def build_home_context():
    dataset_ready = dataset_is_ready()
    model_ready = MODEL_PATH.exists() and CLASS_NAMES_PATH.exists()
    stats = get_dataset_stats() if dataset_ready else None
    return {
        'dataset_ready': dataset_ready,
        'model_ready': model_ready,
        'dataset_name': DATASET_HANDLE,
        'dataset_classes': stats['classes'] if stats else 0,
        'dataset_total_images': stats['images'] if stats else 0,
        'train_count': stats['train_images'] if stats else 0,
        'sample_count': stats['validation_images'] if stats else 0,
        'prediction_threshold': f'{PREDICTION_THRESHOLD:.2f}',
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
            'Faltan dependencias de Machine Learning. Activa .venv312 e instala requirements.txt.'
        ) from exc

    return np, tf


def import_image_dependencies():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            'Falta Pillow. Instala las dependencias del proyecto antes de usar la cámara.'
        ) from exc

    return Image, ImageOps


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
            'No fue posible descargar el dataset de Kaggle. Verifica tu conexión o vuelve a intentar.'
        ) from exc

    if not dataset_is_ready():
        raise RuntimeError(
            'La descarga terminó, pero no se encontró la carpeta esperada del dataset ASL.'
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
    class_names = list(train_ds.class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


@lru_cache(maxsize=1)
def get_or_train_model():
    _, tf = import_ml_dependencies()
    ensure_image_dataset()
    class_names = load_class_names()

    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH)

    train_ds, val_ds, class_names = build_training_datasets(tf)
    model = build_transfer_model(tf, len(class_names))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
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
    if feature_extractor is None:
        save_class_names(class_names)
        return tf.keras.models.load_model(MODEL_PATH)
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
    return tf.keras.models.load_model(MODEL_PATH)


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


def predict_camera_image(model, image_data):
    np, _ = import_ml_dependencies()
    Image, ImageOps = import_image_dependencies()
    class_names = load_class_names()

    encoded = image_data.split(',', 1)[1] if ',' in image_data else image_data
    image = Image.open(BytesIO(b64decode(encoded))).convert('RGB')
    image = ImageOps.fit(image, IMAGE_SIZE, method=Image.Resampling.LANCZOS)
    array = np.array(image).astype('float32')
    array = array.reshape(1, *IMAGE_SIZE, 3)

    prediction = model.predict(array, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    return class_names[predicted_index], float(np.max(prediction))


def image_to_base64(image_array):
    Image, _ = import_image_dependencies()
    image = Image.fromarray(image_array.astype('uint8'))
    image = image.resize((160, 160), Image.Resampling.NEAREST)
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    encoded = b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{encoded}'


def build_preview_samples(class_names, sample_size=12):
    np, _ = import_ml_dependencies()
    Image, ImageOps = import_image_dependencies()

    rng = random.Random(123)
    selected_classes = class_names[:]
    rng.shuffle(selected_classes)
    selected_classes = selected_classes[:min(sample_size, len(selected_classes))]

    images = []
    labels = []
    for class_name in selected_classes:
        class_dir = TRAIN_DIR / class_name
        image_paths = sorted(
            [path for path in class_dir.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        )
        if not image_paths:
            continue

        image_path = rng.choice(image_paths)
        image = Image.open(image_path).convert('RGB')
        image = ImageOps.fit(image, IMAGE_SIZE, method=Image.Resampling.LANCZOS)
        images.append(np.array(image).astype('float32'))

        one_hot = np.zeros(len(class_names), dtype='float32')
        one_hot[class_names.index(class_name)] = 1.0
        labels.append(one_hot)

    return np.stack(images), np.stack(labels)


def load_class_names():
    if CLASS_NAMES_PATH.exists():
        return json.loads(CLASS_NAMES_PATH.read_text(encoding='utf-8'))

    ensure_image_dataset()
    class_names = sorted([path.name for path in TRAIN_DIR.iterdir() if path.is_dir()])
    save_class_names(class_names)
    return class_names


def save_class_names(class_names):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(
        json.dumps(list(class_names), ensure_ascii=True, indent=2),
        encoding='utf-8',
    )


def resolve_prediction_action(label, confidence):
    if confidence < PREDICTION_THRESHOLD:
        return 'ignore'
    return SPECIAL_ACTIONS.get(label, 'append')


def build_action_token(label, confidence):
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
    if label == 'nothing':
        return '<sin-sena>'
    return label
