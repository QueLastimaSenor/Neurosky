import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import threading
from data_extraction import extract_data
import numpy as np
import multiprocessing
from spectr_extraction import extract_spectr
import pickle
import time


#Глобальные переменные для работы с гарнитурой
jobs = []
process_job = multiprocessing.Queue()
file = "data.csv"
duration = 15
num_spec = 4
segmentation = 4

#Инициализация Dash приложения
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.MINTY],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}],
                update_title=None)

app.config.suppress_callback_exceptions = True

#Обертка приложения
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Nav([
                dbc.NavItem(dbc.Button("Start", id="button", disabled=False, n_clicks=0, color="primary", className="mr-1",
                            style={"padding": "10px 40px","font-size": "20px", "margin": "10px", "margin-top": "20px"})),
                dbc.NavItem(dbc.NavLink(id="live-text", style={ "color": "#ffffff", 
                                                                "font-size": "20px",
                                                                "margin": "10px",
                                                                "padding-top": "20px"}))
            ])
        ]),
    ], style={"background-color": "#3c3f50",}),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="live-graph"),
            dcc.Interval(
                id="interval-component",
                interval=100,
                n_intervals=0   
            )
        ], style={"margin-top": "10px"}, width={"size": 6}),
        dbc.Col([
            dcc.Graph(id="live-heatmap"),
            dcc.Interval(
                id="interval-heatmap",
                interval=1500,
                n_intervals=0   
            )
        ], style={"margin-top": "10px"}, width={"size": 3}),
    ], justify="center"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="live-svm"),
            dcc.Interval(
                id="interval-stats",
                interval=1500,
                n_intervals=0   
            )
        ], style={"margin-top": "10px"}, width={"size": 3}),
        dbc.Col([
            dcc.Graph(id="live-xgboost"),
        ], style={"margin-top": "10px"}, width={"size": 3}),
        dbc.Col([
            dcc.Graph(id="live-percentage"),
        ], style={"margin-top": "10px"}, width={"size": 3}),
        html.P(id="live-prediction", style={"display": "none"})
    ], justify="center")
], fluid=True)

#Словарь с состояниями работы программы
state = {
    "prediction": [],
    "button": False,
    "process_prediction": 0,
    "process_extraction": 0,
    "jobs": "Data exctraction isn't done",
    "disabled": False,
    "labels_svm": [],
    "values_svm": [],
    "labels_xg": [],
    "values_xg": []
}

#Mетоды Dash приложения для взаимодействия гарнитуры с интерфейсом
@app.callback([Output("button", "disabled"),
              Output("button", "n_clicks")],
              [Input("button", "n_clicks"),
               Input("interval-heatmap", "n_intervals")])
def button_live(clicks, intervals):
    if clicks > 0:
        state["process_extraction"] = 0
        state["process_prediction"] = 0
        extract_thread = threading.Thread(target=extract_data, kwargs={"jobs": jobs})
        extract_thread.start()
        # extract_data(jobs=jobs)
        state["button"] = True
        state["disabled"] = False
        state['jobs'] = "Data exctraction isn't done"
        return True, 0
    else:
        return state["button"], 0

@app.callback([Output("live-text", "children"),
               Output("interval-component", "disabled")],
              [Input("interval-heatmap", "n_intervals")])
def update_jobs(n):
    if jobs:
        if state["process_extraction"] == 0:
            #Обработка сетрограмм вызвана отдельным процессом, потому
            #что matplotlib не может быть вызван в дочернем треде
            multiprocessing.Process(target=extract_spectr, args=(file, num_spec, 128 * 3, process_job)).start()
            state["process_extraction"] = 1
            state["button"] = False
            time.sleep(0.1)
            state["jobs"] = jobs[0]
            jobs.clear()
            state["disabled"] = True
            
        return [html.P(f"{state['jobs']}")], state["disabled"]
    else:
        return [html.P(state['jobs'])], state["disabled"]

@app.callback(Output("live-prediction", "children"),
              [Input("interval-heatmap", "n_intervals")])
def update_pred(n):
    if state["process_prediction"] == 0:
        if not process_job.empty():
            model_xgboost = pickle.load(open("xgboost_model.sav", 'rb'))
            model_svm = pickle.load(open("svm_model.sav", 'rb'))
            dataset = process_job.get()
            state["prediction"] = [list(model_svm.predict(dataset))]
            state["prediction"].append(list(model_xgboost.predict(dataset)))
            state["labels_svm"] = list(set(state["prediction"][0]))
            state["values_svm"] = [state["prediction"][0].count(x) for x in state["labels_svm"]]
            state["labels_xg"] = list(set(state["prediction"][1]))
            state["values_xg"] = [state["prediction"][1].count(x) for x in state["labels_xg"]]
            state["process_prediction"] = 1
            return True

        else:
            return False

    return False


@app.callback(Output("live-graph", "figure"),
              [Input("interval-component", "n_intervals")])
def update_graph_scatter(n):
    try:
        df = pd.read_csv("data.csv")
        rawEeg = df["rawEeg"]

        X = np.arange(0, rawEeg.size, 1)/128
        Y = df["rawEeg"]
        fig = go.Figure(go.Scatter(
                x=X,
                y=Y,
                name="Scatter",
                mode="lines",
                line={"color": "#20c997"}
        ))
        fig.update_layout(title="Voltage versus time plot", 
                          paper_bgcolor="#dbe9f0")
        fig.update_yaxes(title_text="Voltage, mV")
        fig.update_xaxes(title_text="Time, s")
        return fig

    except Exception as e:
        with open("errors.txt","a") as f:
            f.write(str(e))
            f.write("\n")

@app.callback(Output("live-heatmap", "figure"),
              [Input("interval-heatmap", "n_intervals")])
def update_heatmap(n):
    if state["process_extraction"] != 0:
        data_plot = pickle.load(open("data_plot", "rb"))
        fig = go.Figure()
        fig.set_subplots(rows=2, cols=2, horizontal_spacing=0.1)

        fig.add_trace(
            go.Heatmap(z=data_plot[0], showscale=False, colorscale = 'Viridis'),
            row=1, col=1
        )

        fig.add_trace(
            go.Heatmap(z=data_plot[1], showscale=False, colorscale = 'Viridis'),
            row=1, col=2
        )

        fig.add_trace(
            go.Heatmap(z=data_plot[2], showscale=False, colorscale = 'Viridis'),
            row=2, col=1
        )

        fig.add_trace(
            go.Heatmap(z=data_plot[3], colorscale = 'Viridis'),
            row=2, col=2
        )
        fig.update_layout(title="Spectrogram of a signal",
                          paper_bgcolor="#dbe9f0")     
        return fig
    else:
        fig = go.Figure()
        fig.update_layout(title="Spectrogram of a signal",
                          paper_bgcolor="#dbe9f0")
        return fig

@app.callback(Output("live-svm", "figure"),
              [Input("interval-stats", "n_intervals")])
def update_svm(n):
    if state["process_extraction"] != 0:
        fig = go.Figure(
            go.Pie(
                labels=state["labels_svm"], 
                values=state["values_svm"]
            )
        )
        fig.update_layout(title="SVM Prediction",
                          paper_bgcolor="#dbe9f0")
        return fig
    else:
        fig = go.Figure()
        fig.update_layout(title="SVM Prediction",
                          paper_bgcolor="#dbe9f0")
        return fig

@app.callback(Output("live-xgboost", "figure"),
              [Input("interval-stats", "n_intervals")])
def update_xgboost(n):
    if state["process_extraction"] != 0:
        fig = go.Figure(
            go.Pie(
                labels=state["labels_xg"], 
                values=state["values_xg"]
            )
        )
        fig.update_layout(title="XGBoost Prediction",
                          paper_bgcolor="#dbe9f0")
        return fig
    else:
        fig = go.Figure()
        fig.update_layout(title="XGBoost Prediction",
                          paper_bgcolor="#dbe9f0")
        return fig

@app.callback(Output("live-percentage", "figure"),
              [Input("interval-stats", "n_intervals")])
def update_percentage(n):
    if state["process_extraction"] != 0:
        fig = go.Figure(
            # go.Pie(

            # )
        )
        fig.update_layout(title="Total statistics",
                          paper_bgcolor="#dbe9f0")
        return fig
    else:
        fig = go.Figure()
        fig.update_layout(title="Total statistics",
                          paper_bgcolor="#dbe9f0")
        return fig



if __name__ == "__main__":  
    app.run_server(debug=False)