# Senas Al Aire

Aplicacion web en Django para reconocer senas del alfabeto con camara usando un sistema hibrido:

- CNN entrenada sobre `grassknoted/asl-alphabet`
- landmarks de MediaPipe para mejorar la deteccion de la mano

## Que hace

- Abre la camara y empieza a reconocer senas automaticamente.
- Forma palabras a partir de letras detectadas.
- Pronuncia la palabra al detectar `space`, al detener la camara o despues de una pausa.
- En local permite entrenar y evaluar el modelo.
- En Render queda lista para inferencia sin reentrenar.

## Requisitos locales

- Windows 10 o superior.
- Python `3.11` o `3.12`.
- Conexion a internet en la primera ejecucion local si necesitas reconstruir artefactos.

## Clonar el proyecto

```powershell
git clone https://github.com/camiloGoca/Se-as-Al-Aire.git
cd Se-as-Al-Aire
```

## Ejecutarlo localmente

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Luego abre:

```text
http://127.0.0.1:8000/
```

## Archivos que si van al repo para Render

Estos artefactos se versionan para que Render no tenga que reentrenar:

- `proyecto/models/asl_alphabet_mobilenet_v2.keras`
- `proyecto/models/asl_alphabet_class_names.json`
- `proyecto/models/asl_landmarks_mlp.keras`
- `proyecto/models/asl_landmarks_metadata.json`
- `proyecto/models/hand_landmarker.task`

## Archivos que no se suben

Se siguen ignorando porque son pesados o regenerables:

- `proyecto/datasets/`
- `proyecto/.cache/`
- `proyecto/models/asl_landmarks_dataset_v1.npz`
- entornos virtuales y `db.sqlite3`

## Despliegue en Render

La configuracion del proyecto ya queda preparada para Render:

- usa variables de entorno para `SECRET_KEY`, `DEBUG` y hosts
- sirve estaticos con WhiteNoise
- usa Postgres si existe `DATABASE_URL`
- desactiva entrenamiento en produccion con `ALLOW_TRAINING=false`

### Opcion recomendada: despliegue manual

1. Sube tus cambios a GitHub.
2. En Render crea una base de datos PostgreSQL.
3. Copia la `Internal Database URL`.
4. En Render crea un `Web Service` conectado a tu repo.
5. Usa estos valores:

```text
Language: Python 3
Build Command: bash build.sh
Start Command: python -m gunicorn proyecto.asgi:application -k uvicorn.workers.UvicornWorker
```

6. Agrega estas variables de entorno:

```text
PYTHON_VERSION=3.11.11
DEBUG=false
ALLOW_TRAINING=false
WEB_CONCURRENCY=4
DATABASE_URL=<internal database url de Render>
SECRET_KEY=<valor generado por Render>
ALLOWED_HOSTS=<tu-servicio>.onrender.com
```

7. Despliega.
8. Cuando termine, abre el `Shell` del servicio y crea un superusuario si lo necesitas:

```bash
python manage.py createsuperuser
```

## Nota importante sobre Render

En Render la app debe usarse solo para inferencia. El entrenamiento y la evaluacion grande quedan pensados para local, no para produccion.

## Estructura importante

- `manage.py`: entrada principal de Django
- `proyecto/settings.py`: configuracion local y de produccion
- `proyecto/views.py`: logica de inferencia, landmarks y entrenamiento
- `proyecto/templates/index.html`: interfaz principal
- `proyecto/static/css/style.css`: estilos
- `build.sh`: build de Render
- `requirements.txt`: dependencias
