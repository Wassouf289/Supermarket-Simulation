
"""
calculate transition matrix for the whole week 
"""
from __future__ import division  
import pandas as pd
import datetime
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 
import pydot
import graphviz


# read data for ech week day
monday = pd.read_csv("./data/monday.csv",sep=";", parse_dates=True)

tuesday = pd.read_csv("./data/tuesday.csv", sep=";", parse_dates=True)

wednesday = pd.read_csv("./data/wednesday.csv", sep=";", parse_dates=True)

thursday = pd.read_csv("./data/thursday.csv", sep=";", parse_dates=True)

friday = pd.read_csv("./data/thursday.csv",sep=";", parse_dates=True)


""" add weekday column """
monday["weekday"] = "monday"
tuesday["weekday"] = "tuesday"
wednesday["weekday"] = "wednesday"
thursday["weekday"] = "thursday"
friday["weekday"] = "friday"


"""Edit customer no to include the weekday in order to make it unique"""
def edit_customer_no(df, weekday):
    df["customer_no"] = df["customer_no"].apply(lambda x: str(x)+"_"+weekday)



edit_customer_no(monday, "monday")
edit_customer_no(tuesday, "tuesday")
edit_customer_no(wednesday, "wednesday")
edit_customer_no(thursday, "thursday")
edit_customer_no(friday, "friday")



"""there is no checkout state for the last customers of the day so this should be fixed"""

def add_missing_checkout_state(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    checkout_customers = set(df[df["location"] == "checkout"]
                   ["customer_no"].unique())
    whole_customers = set(df["customer_no"].unique())
    diff = whole_customers.difference(checkout_customers)

    date = df.index.date[0]
    new_time_stemp = f"{date} 22:00:00"

    df.reset_index(inplace=True)

    for customer in diff:
        df = df.append({"timestamp": pd.to_datetime(new_time_stemp),
                            "customer_no": customer,
                            "location": "checkout"},
                            ignore_index=True)
    return df



monday = add_missing_checkout_state(monday)
tuesday = add_missing_checkout_state(tuesday)
wednesday = add_missing_checkout_state(wednesday)
thursday = add_missing_checkout_state(thursday)
friday = add_missing_checkout_state(friday)


"""get a dataframe for the whole week"""
week_data = pd.concat([monday, tuesday, wednesday, thursday, friday])
week_data.set_index("timestamp", inplace=True)


#group data by customer 
week_data = week_data.groupby("customer_no").resample(rule="1T").ffill()


#get the next transition for every customer
del week_data["customer_no"]
week_data.reset_index(inplace=True)
week_data.sort_values(["customer_no", "timestamp"], inplace=True)
week_data["after"] = week_data["location"].shift(-1)
week_data.rename(columns={"location":"before"},inplace=True)


#the next transition for checkout state is checkout state
week_data.loc[(week_data.before == 'checkout'), 'after'] = 'checkout'


transitions=week_data[["before","after"]]

# calculate transition probabilities matrix
transition_matrix = pd.crosstab(week_data['before'],
                                week_data['after'],
                                normalize=0)

# export the transition probabilities matrix to csv
transition_matrix.to_csv("output/transition_matrix.csv")
#print(transition_matrix)

#Visualize the probabilities as a heat map
plt.rcParams['figure.figsize'] = (10.0, 6.0)
plt.rcParams['font.family'] = "serif"
sns.heatmap(transition_matrix,cmap='coolwarm', annot=True)



Q = pd.read_csv('data/transition_matrix.csv', index_col=0)
Q=Q.round(2)
states = ['entrance','drinks','dairy','fruit','spices','checkout']

# create a function that maps transition probability dataframe
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx,col)] = Q.loc[idx,col]
    return edges

edges_wts = _get_markov_edges(Q)

# create graph object
G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(states)

# edges represent transition probabilities
for k, v in edges_wts.items():

    if v > 0.0:
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)

pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
nx.draw_networkx(G, pos)
# create edge labels for jupyter plot
edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
nx.drawing.nx_pydot.write_dot(G, 'output/markov.dot')

(graph,) = pydot.graph_from_dot_file('output/markov.dot')
graph.write_png('output/markov.png')


# aisles visited first
week_data['firsts'] = week_data.duplicated('customer_no')
firsts = week_data[week_data['firsts'] == False]
# initial_state_vector
initial_state_abs = firsts.groupby('before').count()['customer_no']
initial_state_vector = initial_state_abs/initial_state_abs.sum()
#print(initial_state_vector)
initial_state_vector.to_csv('output/initial_state_vector.csv')





