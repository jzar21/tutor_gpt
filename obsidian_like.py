import dash_cytoscape as cyto
from dash import html, Output, Input, dcc, State
import dash
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS
import json
from tqdm import tqdm
from rag_config import RAGArgs
from llm import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd

plt.style.use(['science', 'notebook', 'grid'])

with open('all-chats-export-1749751488909.json') as f:
    data = json.load(f)

user_questions = []
for chat in data:
    for message in chat['chat']['messages']:
        if message['role'] == 'user':
            user_questions.append(message['content'])

args = RAGArgs()
args.model_embbedding = 'mxbai-embed-large'
embedder = OllamaEmbedder(args)
x = []
for query in tqdm(user_questions[:5000]):
    x.append(embedder.embed_query(query))

x = np.array(x)
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(x)

app = dash.Dash(__name__)
initial_clusters = [
    {
        'data': {'id': f'cluster_{i}', 'label': f'Tema {i}', 'tooltip': f'Cluster {i}'},
        'classes': 'cluster'
    } for i in range(kmeans.n_clusters)
]

app.layout = html.Div([
    html.H3("Preguntas agrupadas por tema (haz clic para expandir múltiples clusters)"),

    # Almacena qué clusters han sido abiertos
    dcc.Store(id='store-expanded-clusters', data=[]),

    cyto.Cytoscape(
        id='cytoscape',
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '600px'},
        elements=initial_clusters,
        stylesheet=[
            {'selector': 'node', 'style': {
                'label': 'data(label)',
                'background-color': '#95afc0',
                'width': 'data(size)',
                'height': 'data(size)',
                'text-valign': 'center',
                'color': '#2d3436',
                'font-size': '12px',
            }},
            {'selector': '.cluster', 'style': {
                'background-color': '#fd79a8',
                'width': 60,
                'height': 60,
                'font-size': '16px'
            }},
            {'selector': '.pregunta', 'style': {
                'background-color': '#74b9ff',
                'width': 20,
                'height': 20,
                'font-size': '10px'
            }},
            {'selector': 'node:hover', 'style': {
                'background-color': '#ffeaa7',
                'cursor': 'pointer'
            }},
            {'selector': 'edge', 'style': {
                'line-color': '#dfe6e9',
                'width': 2
            }},
        ]
    )
])


@app.callback(
    Output('cytoscape', 'elements'),
    Output('store-expanded-clusters', 'data'),
    Input('cytoscape', 'tapNodeData'),
    State('store-expanded-clusters', 'data'),
    prevent_initial_call=True
)
def expand_node(data, expanded_clusters):
    if not data or not data['id'].startswith("cluster_"):
        return dash.no_update, expanded_clusters

    cluster_id = int(data['id'].split('_')[1])

    # Agregar cluster si no está ya expandido
    if cluster_id not in expanded_clusters:
        expanded_clusters.append(cluster_id)

    # Regenerar elementos visibles con todos los clusters expandidos
    elements = initial_clusters.copy()
    for cid in expanded_clusters:
        for idx, pregunta in enumerate(user_questions):
            if labels[idx] == cid:
                pid = f"pregunta_{idx}"
                elements.append({
                    'data': {
                        'id': pid,
                        'label': pregunta,
                        'tooltip': pregunta,
                        'size': 20
                    },
                    'classes': 'pregunta'
                })
                elements.append({
                    'data': {
                        'source': f'cluster_{cid}',
                        'target': pid
                    }
                })

    return elements, expanded_clusters


if __name__ == '__main__':
    app.run(debug=True)
