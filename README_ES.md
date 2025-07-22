![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)

# Dependencias

La instalación de este software no requiere de ninguna dependencia extra más alla de Python. No obstante, es recomendable el uso de algún gestor de versiones de Python, como lo es [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) o [venv](https://docs.python.org/3/library/venv.html), para realizar la instalación en un entorno limpio y evitar posibles conflictos.

# Creación del entorno

## Conda

```bash
conda create --name tutor_gpt python=3.10
conda activate tutor_gpt
```

## Venv

```bash
python -m venv venv
source venv/bin/activate
```

# Instalación

Una vez creado el entorno, nos descargamos el repositorio con el comando (actualmente es un repositorio privado, si se me permite ponerlo público, este comando funciona. En otro caso le tengo que pasar el código por correo):

```bash
git clone https://github.com/jzar21/tutor_gpt
```

Nos movemos al directorio descargado:

```bash
cd tutor_gpt
```

Instalamos las dependencias con pip:

```bash
pip install -r requirements.txt
```

Este proceso puede tardar un tiempo considerable.

Por último, creamos la carpeta `data_test` donde se van a guardar las métricas de tiempos.

```bash
mkdir -p data_test
```

# Ejecución

Hay dos formas de ejecutar la app, via terminal y o como una API wrapper para OpenWebUI.

Ambos comparten varias características en común, pero la más destacable es el uso de un archivo .json como configuración del LLM a usar. Estas vienen en el archivo `rag_config.py` con las siguientes configuraciones por defecto:

```python
model: str = 'llama3.2'
model_embbedding: str = 'llama3.2'
url: str = 'http://localhost:11434'
url_embed: str = 'http://localhost:11434'
chunk_size: int = 1000
chunk_overlap: int = 100
temperature: float = 0
top_k: int = 5
fetch_k: int = 5
stream: bool = False
seed: int = 42
rag_type: str = 'naive'
open_ai_api: bool = False
rag_search_tipe: str = 'mmr'
```

Esta configuración puede ser cambiada por la que nos sea conveniente indicándola en un .json. Por ejemplo, este archivo hace uso de `Phi4` en lugar de `llama3.2` como LLM.

```json
{
    "model": "phi4"
}
```


## Modelos Soportados

Actualmente, únicamente se puede hacer uso de los modelos disponibles de Ollama y Gemini.

#### WARNING

Si se desea usar gemini como LLM, debe existir en el entorno la variable `GOOGLE_API_KEY` donde esté la API de gemini.

## LLM en terminal

La ejecución depende del archivo `main.py`. Este se puede ejecutar de dos formas diferentes:

- Modo Interactivo.
- Evaluación Offline.

Cuando termine la ejecución de ambas opciones, se actualiza el archivo `data_test/datos_tiempos_ollama.csv` donde se guardan métricas de tiempo de ejecución del sistema RAG.


#### Modo Interactivo

Para la ejecución como un chat LLM, se debe pasar las siguientes opciones:

- `--config_path`: Path al archivo de configuración.
- `--files`: Lista de los archivos a usar como base de conocimiento. Se soporta archivos .pdf, .md y .txt.
- `--interactive true`: Activa el modo interactivo.

Ejemplo:

```bash
python main.py --config_path config_llama.json  --files Info.pdf Info2.pdf --interactive true
```

El programa termina cuando se introduce una cadena vacía.

#### Evaluación Batch

Desde el archivo `main.py`, se puede ejecutar una opción, la cual, mediante un archivo CSV, se puede realizar una evaluación del sistema RAG.

Para ello, el archivo CSV debe tener los siguientes campos:

- `pregunta`: Pregunta a realizar al sistema.
- `path`: Path del archivo donde se encuentra la respuesta.
- `paginas`: Páginas donde se encuentra la respuesta. Este debe ir entre comillas y corchetes.

Ejemplo de configuración:

```csv
pregunta,path,paginas
¿Cuando se entrega?,guion.pdf,"[2]"
¿Que algoritmos hay que implementar?,guion.pdf,"[4,5,6]"
```

Los argumentos a pasar al archivo son los siguientes:

- `--config_path`: Path al archivo de configuración.
- `--files`: Lista de los archivos a usar como base de conocimiento. Se soporta archivos .pdf, .md y .txt.
- `batch_questions`: Archivo CSV.

Una vez terminado se imprimirán por pantalla las métricas obtenidas.

## Wrapper

La otra configuración es levantar un servidor para hacer de Wrapper sobre la API de Ollama para que sea usada por servicios como OpenWebUI.

De esto se encarga el archivo `app.py`. Para ello, se le deben pasar las siguientes configuraciones:

- `--config_path`: Path al archivo de configuración.
- `--files`: Lista de los archivos a usar como base de conocimiento. Se soporta archivos .pdf, .md y .txt.
- `--port`: Puerto donde corre el servidor. Por defecto 5001.

Ejemplo:

```bash
python app.py --config_path config_llama.json  --files guion.pdf --port 5001
```

# OpenWebUI

En primera instancia, se debe levantar el servidor de [OpenWebUI](https://openwebui.com/).

Una vez que se tenga el servidor configurado, para hacer uso del sistema RAG, se debe cambiar la dirección del servidor a usar por la del servidor configurado en los pasos anteriores. Además, de debe cambiar la dirección del servidor de embeddings. Se recomienda usar modelo de embeddings `mxbai-embed-large`.


# Licencia

Este proyecto está bajo la licencia Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional. Consulta el archivo [LICENSE](LICENSE) para más detalles.
