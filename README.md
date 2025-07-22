![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)

# Dependencies

The installation of this software does not require any extra dependencies other than Python. However, it is recommended to use a Python version manager, such as [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [venv](https://docs.python.org/3/library/venv.html), to install it in a clean environment and avoid possible conflicts.

# Creating the Environment

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

# Installation

Once the environment is created, clone the repository with the following command (currently, it is a private repository. If I am allowed to make it public, this command will work. Otherwise, I need to send you the code via email):

```bash
git clone https://github.com/jzar21/tutor_gpt
```

Navigate to the downloaded directory:

```bash
cd tutor_gpt
```

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

This process may take some time.

Finally, create the `data_test` folder where the time metrics will be saved.

```bash
mkdir -p data_test
```

# Running the Application

There are two ways to run the app: via terminal or as an API wrapper for OpenWebUI.

Both share several common features, but the most notable one is the use of a `.json` file to configure the LLM to be used. The default configurations are located in the `rag_config.py` file with the following settings:

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

This configuration can be changed to a more suitable one by specifying it in a `.json` file. For example, this file uses `Phi4` instead of `llama3.2` as the LLM.

```json
{
    "model": "phi4"
}
```

## Supported Models

Currently, only the models available from Ollama and Gemini can be used.

#### WARNING

If you want to use Gemini as the LLM, the `GOOGLE_API_KEY` variable must be set in the environment with the Gemini API key.

## LLM in Terminal

The execution depends on the `main.py` file. It can be run in two different modes:

* Interactive Mode.
* Offline Evaluation.

When the execution of either option is finished, the file `data_test/datos_tiempos_ollama.csv` will be updated, storing time metrics of the RAG system execution.

#### Interactive Mode

To run it as an LLM chat, pass the following options:

* `--config_path`: Path to the configuration file.
* `--files`: List of files to use as knowledge base. `.pdf`, `.md`, and `.txt` files are supported.
* `--interactive true`: Activates interactive mode.

Example:

```bash
python main.py --config_path config_llama.json  --files Info.pdf Info2.pdf --interactive true
```

The program will terminate when an empty string is entered.

#### Batch Evaluation

From the `main.py` file, a batch evaluation option can be executed using a CSV file.

For this, the CSV file must have the following fields:

* `pregunta`: Question to ask the system.
* `path`: Path of the file where the answer is located.
* `paginas`: Pages where the answer is found. This should be enclosed in quotes and brackets.

Example configuration:

```csv
pregunta,path,paginas
¿Cuando se entrega?,guion.pdf,"[2]"
¿Que algoritmos hay que implementar?,guion.pdf,"[4,5,6]"
```

The arguments to pass to the file are as follows:

* `--config_path`: Path to the configuration file.
* `--files`: List of files to use as knowledge base. `.pdf`, `.md`, and `.txt` files are supported.
* `batch_questions`: CSV file.

Once finished, the obtained metrics will be printed on the screen.

## Wrapper

The other configuration is to run a server to act as a Wrapper around the Ollama API so that it can be used by services like OpenWebUI.

This is handled by the `app.py` file. To run it, pass the following configurations:

* `--config_path`: Path to the configuration file.
* `--files`: List of files to use as knowledge base. `.pdf`, `.md`, and `.txt` files are supported.
* `--port`: Port where the server runs. Default is 5001.

Example:

```bash
python app.py --config_path config_llama.json  --files guion.pdf --port 5001
```

# OpenWebUI

First, you must start the [OpenWebUI](https://openwebui.com/) server.

Once the server is configured, to use the RAG system, you need to change the server address to the one configured in the previous steps. Additionally, you must change the embedding server address. It is recommended to use the embedding model `mxbai-embed-large`.

# License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See the [LICENSE](LICENSE) file for more details.