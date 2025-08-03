import networkx as nx
from pathlib import Path
from bokeh.io import output_file, show
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool, MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool)
from bokeh.palettes import Spectral4, Spectral8, Category20_20, Set1_3
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.plotting import from_networkx
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot, layout
from datetime import date
from random import randint
from bokeh.io import show
from bokeh.models import ColumnDataSource, Div, DataTable, DateFormatter, TableColumn
from bokeh.io import show
from bokeh.models import PreText
from bokeh.io import curdoc
from bokeh.models import DataTable, TableColumn, PointDrawTool, ColumnDataSource
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.io import output_file, show
from fa2 import ForceAtlas2

def report_GSNAc(performance_summary, parameters_list, graphs_to_plot_dict, output_folder_path=""):

    
    #Web page header
    main_header = Div(text='<h1 style="text-align: center">Generalized Social Network Analysis-based Classifier Performance Comparison</h1>')
    #Summary text
    pretext = Div(text='<h2 style="text-align: center">Parameters are: ' + str(parameters_list) + '  </h2>')
    #Performance summary table
    performance_summary = performance_summary.reset_index()
    data_table = DataTable(
                    columns=[TableColumn(field=Ci, title=Ci) for Ci in performance_summary.columns],
                    source=ColumnDataSource(performance_summary),
                    width=600)   
    
    #Generate plots; grouped by folds or classifier
   
    node_positions_dict = nx.kamada_kawai_layout(list(graphs_to_plot_dict.values())[0])
    plots = []
    graph_titles = []
    for title, graph in graphs_to_plot_dict.items():
        prediction_plot = graph_visualizer(graph, graph_title = title, node_positions_dict = node_positions_dict)
        plots.append([prediction_plot])
        graph_titles.append([Div(text='<h2 style="text-align: center">' + title + '  </h2>')])
        

    #Generate presentation layout
    layout_list = [[main_header], [pretext], [data_table]]
    for t, p in zip(graph_titles, plots):
        layout_list.append(t)
        layout_list.append(p)
    
    # make a grid
    output_file(output_folder_path + "D. Result.html")
    
    grid = layout(layout_list, sizing_mode=('scale_width'))
    #grid = layout(layout_list)
    
    show(grid)

    
def graph_visualizer(G, color_by = 'Prediction_result', graph_title = "Temp graph", node_positions_dict=nx.fruchterman_reingold_layout):
    plot = Plot(width = 300, height = 300, 
                x_range=Range1d(-1.1,1.1), 
                y_range=Range1d(-1.1,1.1))
    
    plot.title.text = graph_title
    
    plot.add_tools(HoverTool(tooltips=None), TapTool(), BoxSelectTool())   
    """
    node_hover_tool = HoverTool(tooltips=[("Name", "@train_test_flag"), ("DoD","@degree_of_domesticity")])
    plot.add_tools(node_hover_tool)
    """
    graph_renderer = from_networkx(G, layout_function = node_positions_dict, scale = 1, center = (0,0))    
    
    factors_dict = nx.get_node_attributes(G, color_by)
    factors = list(set(factors_dict.values()))
    factors.sort()
    
    #color_map = factor_cmap(field_name = color_by, palette = Spectral8, factors = factors)
    color_map = factor_cmap(field_name = color_by, palette = Set1_3, factors = factors)
       
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=color_map)
    graph_renderer.node_renderer.selection_glyph = Circle(size=28, fill_color=Spectral4[2])
    graph_renderer.node_renderer.hover_glyph = Circle(size=40, fill_color=Spectral4[1])
    
    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)
    
    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = EdgesAndLinkedNodes()

    plot.renderers.append(graph_renderer)
    
    #output_file("interactive_graphs.html")
    #show(plot)
    
    return plot


"""

G1 = nx.random_geometric_graph(200, 0.125)

G2 = nx.read_gexf(Path(__file__).parent / "graph.gexf")
#pos = nx.spring_layout(G)
#pos = nx.fruchterman_reingold_layout(G, iterations=500)

#pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)

#nx.set_node_attributes(G, pos, 'pos')
import pandas as pd
import numpy as np
psumm=pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
parameters_list = ["asds", 1,2,"asdasdasd"]
report_GSNAc(psumm, parameters_list, [G1,G2,G1])


forceatlas2 = ForceAtlas2(
                            # Behavior alternatives
                            outboundAttractionDistribution=True,  # Dissuade hubs
                            linLogMode=False,  # NOT IMPLEMENTED
                            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                            edgeWeightInfluence=1.0,
    
                            # Performance
                            jitterTolerance=1.0,  # Tolerance
                            barnesHutOptimize=True,
                            barnesHutTheta=1.2,
                            multiThreaded=False,  # NOT IMPLEMENTED
    
                            # Tuning
                            scalingRatio=2.0,
                            strongGravityMode=False,
                            gravity=1.0,
    
                            # Log
                            verbose=True)
temp_G = list(graphs_to_plot_dict.values())[0]
node_positions_dict = forceatlas2.forceatlas2_networkx_layout(temp_G, pos=None, iterations=2000)
poss=list(node_positions_dict.values())
print(poss)

"""
