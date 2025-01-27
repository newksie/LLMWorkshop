import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import re
import plotly.graph_objs as go
from plotly.offline import iplot
from plotly.offline import init_notebook_mode

nlp = spacy.load("en_core_web_lg")
model = SentenceTransformer('all-MiniLM-L6-v2')


def scatter(x, y, labels, text):
    data = [
        go.Scatter(
            x=x[labels == label],
            y=y[labels == label],
            mode="markers",
            opacity=0.7,
            text=text[labels == label],
            name=label,
            marker={"size": 15, "line": {"width": 0.5, "color": "white"}},
        )
        for label in set(labels)
    ]
    layout = go.Layout(
        xaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
        yaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
        hovermode="closest",
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, config={"displayModeBar": False})

def get_embedding_glove(text):
    doc = nlp(text)
    return doc.vector.reshape(1, 300)

def get_embedding_SBERT(text):
    embeddings = model.encode(text)
    return embeddings.reshape(1, 384)

def truncate_text(text, max_length=100):
    """Function to truncate text to a maximum length with ellipsis."""
    return text.apply(lambda x: x[100:max_length+100] + '...' if len(x) > max_length else x)

def hover_plot(X_input, df):
    # Create a scatter plot with hover information
    data = []
    for label in df['Class Name'].unique():
        idx = df['Class Name'] == label
        # Add a scatter trace for each class
        trace = go.Scatter(
            x=X_input[idx, 0],  # x coordinates of points for the current class
            y=X_input[idx, 1],  # y coordinates of points for the current class
            mode='markers',
            opacity=0.7,
            text=truncate_text(df['Text'][idx]),  # Use review text as hover text
            name=f"Class {label}",
            marker={'size': 15, 'line': {'width': 0.5, 'color': 'white'}},
            hoverinfo='text'  # Only show the text on hover
        )
        data.append(trace)

    # Layout for the plot
    layout = go.Layout(
        title="Embedding Plot with Class Labels",
        xaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
        yaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
        width = 1200,
        height = 800,
        hovermode="closest",  # Hover mode set to "closest" to display the text on hover
    )

    # Create figure and plot
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, config={"displayModeBar": False})