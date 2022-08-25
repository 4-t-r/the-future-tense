import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import plotly.express as px

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

import networkx as nx

import warnings
warnings.filterwarnings("ignore")
print(os.getcwd())
#path = '../../datasets/test_dataset_model_pipeline/future_statements.csv'
path = '../../datasets/test_dataset_model_pipeline/future_statements_altered.csv'
df = pd.read_csv(path, sep='|', error_bad_lines=False)
### Sentiments
# only for exploration - delete in production
#df['sentiment'] = np.random.choice(sentiments, size=len(df))
d_sentiment = {'NEG':-1,'NEU':0,'POS':1}

def sentiment_label(row):
    if row['sentiment'] == 'NEG':
        return -1
    elif row['sentiment'] == 'NEU':
        return 0
    elif row['sentiment'] == 'POS':
        return 1
df['n_sentiment'] = df.apply(lambda row:sentiment_label(row),axis=1)
### Visualization
dfg_t = {'count' : df.groupby(['topic']).size()}
dfg_t = pd.DataFrame(dfg_t).reset_index()

dfg_s = {'count' : df.groupby(['sentiment']).size()}
dfg_s = pd.DataFrame(dfg_s).reset_index()

dfg_ts = {'count' : df.groupby(['topic', 'n_sentiment']).size()}
dfg_ts = pd.DataFrame(dfg_ts).reset_index()

dfg_tm = df.groupby(['topic'])\
        .agg({'sentiment':'size', 'n_sentiment':'mean'}) \
        .rename(columns={'sentiment':'count', 'n_sentiment':'mean_sen'}).reset_index()

dfg_tsm = df.groupby(['topic', 'subtopic'])\
        .agg({'sentiment':'size', 'n_sentiment':'mean'}) \
        .rename(columns={'sentiment':'count', 'n_sentiment':'mean_sen'}).reset_index()
# sentiment_colors = pd.tools.plotting._get_standard_colors(len(sentiment), color_type='random')
# dfg.groupby(['topic', 'sentiment']).size().plot(kind='bar',
#                                                 color=dfg['sentiment'],
#                                                 figsize=(10, 5)
#                                                 )
### Seaborn Init
#dfg_tsm = dfg_tsm.astype({'count':'object', 'mean_sen':'object'})
mid_sen = 0
mean_sen = df['n_sentiment'].mean()
sns.set(style="darkgrid")
sns.set(rc={'figure.figsize':(20,10)})

sns.color_palette("coolwarm_r", as_cmap=True)
palette_1 = sns.color_palette("coolwarm_r", as_cmap=True)
palette_2 = sns.color_palette("coolwarm_r", 3)
colors_topics_pastel = sns.color_palette('pastel')[0:3]
palette_c = {}

# for q in set(scores.Question):
#     avr = (np.average(scores[scores.Question == q].Score))
#     if avr < 1:
#         palette_c[q] = 'r'
#     elif avr < 2.5:
#         palette_c[q] = 'y'
#     else:
#         palette_c[q] = 'g'
### Seaborn Viz
fig, ax = plt.subplots(figsize=(20, 10))

# topic + sentiment (bar)
sns.barplot(x = 'topic'
            , y = 'count'
            , data = dfg_ts
            , hue='n_sentiment'
            , palette= palette_2
            #, dodge=False
            , ax=ax
            )
#sunburst chart of topics+planned subtopics
#https://plotly.com/python/sunburst-charts/

fig = px.sunburst(dfg_tsm
    ,path=['topic']
    ,values='count'
    ,branchvalues='total'
    ,title="Topics by Occurence"
    )
fig.update_layout(
    autosize=False,
    width=750,
    height=750).show()
#sunburst chart of topics+planned subtopics
#https://plotly.com/python/sunburst-charts/
fig = px.sunburst(dfg_tsm[dfg_tsm['subtopic']!='undefined']
    ,path=['subtopic']
    ,values='count'
    ,branchvalues='total'
    ,title="Subtopics by Occurence"
    )
fig.update_layout(
    autosize=False,
    width=750,
    height=750).show()
#sunburst chart of topics+planned subtopics
#https://plotly.com/python/sunburst-charts/
fig = px.sunburst(dfg_tsm
    ,path=['topic', 'subtopic']
    ,values='count'
    #,branchvalues='total'
    ,title="Topics & Subtopics by Occurence"
    )
fig.update_layout(
    autosize=False,
    width=750,
    height=750).show()
fig = px.sunburst(dfg_tsm #dfg_tsm[dfg_tsm['subtopic']!='undefined']
    ,path=['topic', 'subtopic']
    ,values='count'
    ,color='mean_sen'
    ,color_continuous_scale='RdBu'
    ,color_continuous_midpoint=mid_sen
    ,title="Topics & Subtopics (sentiment: NEUTRAL=0)"
    )
fig.update_layout(
    autosize=False,
    width=750,
    height=750).show()
fig = px.sunburst(dfg_stm #dfg_tsm[dfg_tsm['subtopic']!='undefined']
    ,path=['topic', 'subtopic']
    ,values='count'
    ,color='mean_sen'
    ,color_continuous_scale='RdBu'
    ,color_continuous_midpoint=mean_sen
    ,title="Topics & Subtopics (sentiment: MEAN=%s)"%round(mean_sen,4)
    )
fig.update_layout(
    autosize=False,
    width=750,
    height=750).show()
G = nx.Graph()
pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
G.add_node('money') #size=)
G.add_node({'forex':-0.4}) #size=)
G.add_node({'crypto':0.9}) #size=)
G.add_edge('money', 'forex', weight=5)
G.add_edge('forex', 'crypto', weight=1)
G.add_edge('money', 'crypto', weight=4)
for node in G:
    print(node)
color_map = ['red' if node.size < 0 else 'blue' for node.size in G]
nx.draw_spring(G, node_color=)
nx.draw_networkx_nodes(G, pos, node_color='tab:red')
