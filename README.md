# Señas Al Aire

Aplicacion web en Django para reconocer senas del alfabeto con camara usando un modelo de vision entrenado sobre el dataset `grassknoted/asl-alphabet` de Kaggle.

## Que hace

- Abre la camara y empieza a reconocer senas automaticamente.
- Forma palabras a partir de letras detectadas.
- Pronuncia la palabra cuando detecta `space`, cuando detienes la camara o despues de una pausa corta.
- Usa MediaPipe para ubicar la mano antes de clasificar, lo que mejora el recorte en webcam real.
- Permite entrenar o evaluar el modelo desde la interfaz.

## Requisitos

- Windows 10 o superior.
- Python `3.10`, `3.11`, `3.12` o `3.13`.
- Conexion a internet en la primera ejecucion para descargar el dataset y los pesos base del modelo.

## Clonar el proyecto

```powershell
git clone https://github.com/camiloGoca/Se-as-Al-Aire.git
cd Se-as-Al-Aire
```

## Ejecutarlo en otro dispositivo

Abre PowerShell dentro de la carpeta del proyecto y ejecuta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Luego abre en el navegador:

```text
http://127.0.0.1:8000/
```

## Primera ejecucion

En la primera ejecucion el proyecto puede tardar bastante porque:

- descarga el dataset `grassknoted/asl-alphabet`
- descarga los pesos base de `MobileNetV2`
- descarga el modelo `hand_landmarker.task` de MediaPipe
- entrena o carga el modelo

Esos archivos no se suben al repositorio porque se pueden regenerar o descargar automaticamente.

## Uso basico

1. Entra a la pagina principal.
2. Presiona `Entrenar o evaluar modelo` si necesitas generar o validar el modelo.
3. Presiona `Abrir camara`.
4. Mantén la mano dentro del recuadro.
5. Usa la seña de `space` para cerrar una palabra.
6. Usa la seña de `del` para borrar el ultimo caracter.

## Estructura importante

- `manage.py`: entrada principal de Django.
- `proyecto/views.py`: logica del modelo, dataset y endpoints.
- `proyecto/templates/index.html`: interfaz principal.
- `proyecto/static/css/style.css`: estilos.
- `requirements.txt`: dependencias del proyecto.

## Notas

- El dataset y los modelos entrenados se ignoran en Git para mantener el repo liviano.
- Si ya existe un modelo entrenado en `proyecto/models/`, el proyecto lo reutiliza.
- Para mejores resultados usa buena luz y un fondo limpio.
