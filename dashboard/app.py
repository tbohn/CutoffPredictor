#!/usr/bin/env/python
# -*- coding: utf-8 -*-
import argparse
import os.path
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import flask
import plotly.plotly as ply
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import math

# -------------------------------------------------------------------- #
#
# Define functions called by dashboard objects
#
# -------------------------------------------------------------------- #

# Fetch the dataset
def read_df(config):

    mode = 'predict'
    today_str = config['PREDICTION']['REF_DATE']
    option_cut_prior = config['PREDICTION']['FEATURES_CUT_PRIOR']
    option_metadata = config['PREDICTION']['FEATURES_METADATA']
    option_anom = config['PREDICTION']['FEATURES_ANOM']
    opt_str = '{:s}.{:s}.{:s}.{:s}.{:s}'.format(mode, today_str,
                                                option_cut_prior,
                                                option_metadata, option_anom)

    # Read the feature file
    feature_dir = config['PATHS']['FEATURE_TABLE_DIR_PRED']
    feature_file = feature_dir + '/feature_table.' + opt_str + '.best.csv'
    df = pd.read_csv(feature_file)

    # Merge the probabilities into the feature table
    prediction_dir = config['PATHS']['PREDICTIONS_DIR_PRED']
    prob_file = prediction_dir + '/probabilities.' + opt_str + '.best.csv'
    probabilities = pd.read_csv(prob_file)
    df['p_cutoff'] = probabilities['p_cutoff']

    # Convert fractions to percentages
    for col in ['p_cutoff', 'f_late', 'f_zero_vol', 'f_anom3_vol', 
                'f_manom3_vol']:
        df[col] *= 100

    # Drop duplicates
    df.drop_duplicates('location_id', keep='first', inplace=True)

    # Anonymize the addresses
    tmp = df['meter_address'].copy()
    tmp.iloc[0:-1:4] = '1234 N MAIN ST, ANYTOWN, USA'
    tmp.iloc[1:-1:4] = '2345 N MAIN ST, ANYTOWN, USA'
    tmp.iloc[2:-1:4] = '3456 N MAIN ST, ANYTOWN, USA'
    tmp.iloc[3:-1:4] = '4567 N MAIN ST, ANYTOWN, USA'
    df['meter_address'] = tmp

    return df


#returns top indicator div
def indicator(id_text, id_value):
    return html.Div(
        [
            html.P(
                id = id_text,
                className="twelve columns indicator_text"
            ),
            html.P(
                id = id_value,
                className="indicator_value"
            ),
        ],
        className="four columns indicator",
    )


# returns map figure based on threshold and feature_name
def scatter_map(feature_name, df, mapbox_access_token, lat_center, lon_center):

    data = [
        go.Scattermapbox(
            lat=df['latitude'],
            lon=df['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=9,
#                cmax=250,
#                cmin=0,
                color=df[feature_name].values.tolist(),
                colorscale='Portland',
                colorbar=dict(thickness=20),
            ),
            text=df['meter_address'],
        )
    ]

    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=lat_center,
                lon=lon_center
            ),
            pitch=0,
            zoom=9,
        ),
        margin=dict(l=10, r=10, t=0, b=0),
    )

    return dict(data=data, layout=layout)


# returns pie chart that shows customer metadata
def metadata_pie(feature_name, df):

    n_custs = len(df.index)
    categories = df[feature_name].unique().tolist()
    values = []

    # compute % for each category
    for cat in categories:
        n_cat = df[df[feature_name] == cat].shape[0]
        values.append(n_cat / n_custs * 100)

    trace = go.Pie(
        labels=categories,
        values=values,
        marker={"colors": ["#264e86", "#0074e4", "#74dbef", "#eff0f4"]},
    )

    layout = dict(margin=dict(l=15, r=10, t=0, b=65),
                  legend=dict(orientation="h"))
    return dict(data=[trace], layout=layout)


# displays a table
def generate_table(df, max_rows=1000):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +

        # Body
        [html.Tr([
            html.Td(df.iloc[i][col]) for col in df.columns
        ]) for i in range(min(len(df), max_rows))]
    )

# -------------------------------------------------------------------- #
#
# Run the dashboard
#
# -------------------------------------------------------------------- #
def dashboard(config):

    # -------------------------------------------------------------------- #
    #
    # Set up the app
    #
    # -------------------------------------------------------------------- #
    server = flask.Flask(__name__)
    app = dash.Dash(__name__, server=server)
    app.config.suppress_callback_exceptions = True

#    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#    app.css.append_css({'external_url': external_stylesheets})


    # -------------------------------------------------------------------- #
    #
    # Global definitions
    #
    # -------------------------------------------------------------------- #
#    # Read config_file
#    if isinstance(config_file, dict):
#        config = config_file
#    else:
#        config = read_config(config_file)
    # Use config file to set up definitions
    mapbox_access_token = config['MAPPING']['MAPBOX_ACCESS_TOKEN']
    lat_center = config['MAPPING']['MAP_CENTER_LAT']
    lon_center = config['MAPPING']['MAP_CENTER_LON']
    p_cut_col = 'p_cutoff'
    feature_longname = {
        p_cut_col: 'Cutoff Probability (%)',
        'mean_charge_tot': 'Mean Monthly Charges ($/mo)',
        'mean_charge_late': 'Mean Late Charges ($/mo)',
        'mean_vol': 'Mean Volume Usage (kGal/mo)',
        'f_late': 'Late Payment Pct (%)',
        'nCutPrior': 'Number of Prior Cutoffs',
        'meter_address': 'Address',
        'move_in_date': 'Move-In Date',
        'cust_type': 'Customer Type',
        'municipality': 'Municipality',
        'meter_size': 'Meter Size',
    }
    feature_total_str = {
        p_cut_col: 'Predicted Cutoffs',
        'mean_charge_tot': 'Total Revenue Loss ($K/mo)',
        'mean_charge_late': 'Total Late Charges ($K/mo)',
        'mean_vol': 'Total Volume Loss (kGal/mo)',
        'f_late': 'Total Late Payment Pct (%)',
        'nCutPrior': 'Total No. Prior Cutoffs',
    }
    feature_mean_str = {
        p_cut_col: 'Predicted Cutoffs',
        'mean_charge_tot': 'Mean Revenue Loss ($K/mo)',
        'mean_charge_late': 'Mean Late Charges ($K/mo)',
        'mean_vol': 'Mean Volume Loss (kGal/mo)',
        'f_late': 'Mean Late Payment Pct (%)',
        'nCutPrior': 'Mean No. Prior Cutoffs',
    }
    factor_total = {
        p_cut_col: 1,
        'mean_charge_tot': 0.001,
        'mean_charge_late': 0.001,
        'mean_vol': 1,
        'f_late': 1,
        'nCutPrior': 1,
    }

    # -------------------------------------------------------------------- #
    #
    # Define the page layout
    #
    # -------------------------------------------------------------------- #
    app.layout = html.Div(
        [
            # header
            html.Div(
                [
                    html.Span("CutoffPredictor", className='app-title'),
                    html.Div(
                        children=[
                            html.Img(src='file://' + \
                                config['PATHS']['IMAGES_DIR'] + \
                                '/InsightLogo.png',height="100%"),
                            html.Img(src='file://' + \
                                config['PATHS']['IMAGES_DIR'] + \
                                '/ValorWaterLogo.png',height="100%"),
                            html.Img(src='https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png',
                                     height="100%"),
                        ],
                        style={"float":"right","height":"100%"},
                    ),
                ],
                className="row header",
            ),

            # div to save dataframe
            html.Div(read_df(config).to_json(orient="split"),
                     id="current_df",
                     style={"display": "none"}),

            # User selections
            html.Div(
                [
                    html.Div(html.P(
                                 'Thresh. {}:' \
                                     .format(feature_longname[p_cut_col]),
                                 style={"textAlign": "right"},
                                 ),
                             className="two columns"),
                    html.Div(
                        dcc.Input(
                            id="thresh_input",
                            value=50,
                            type='number',
                        ),
                        className="two columns",
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id="history_dropdown",
                            options=[
                                {"label": "Mean Monthly Charges ($)",
                                 "value": "mean_charge_tot"},
                                {"label": "Mean Late Charges ($)",
                                 "value": "mean_charge_late"},
                                {"label": "Mean Volume Used (kGal)",
                                 "value": "mean_vol"},
                                {"label": "Prior Cutoffs",
                                 "value": "nCutPrior"},
                            ],
                            value="mean_charge_tot",
                            clearable=False,
                        ),
                        className="four columns",
                    ),
                    html.Div(
                        dcc.Dropdown(
                            id="metadata_dropdown",
                            options=[
                                {"label": "Customer Type",
                                 "value": "cust_type"},
                                {"label": "Municipality",
                                 "value": "municipality"},
                                {"label": "Meter Size",
                                 "value": "meter_size"},
                            ],
                            value="cust_type",
                            clearable=False,
                        ),
                        className="four columns",
                    ),
                ],
                className="row",
                style={"marginBottom": "10"},
            ),

            # indicators row div
            html.Div(
                [
                    indicator(
                         "left_indicator_title", "left_indicator"
                    ),
                    indicator(
                         "middle_indicator_title", "middle_indicator"
                    ),
                    indicator(
                         "right_indicator_title", "right_indicator"
                    ),
                ],
                className="row",
            ),

            # charts row div
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(id="map_cutoff_title",
                                     children=feature_longname[p_cut_col],
                                     style={"textAlign": "center"}),
                            dcc.Graph(
                                id="map_cutoff",
                                style={"height": "90%", "width": "98%"},
                                config=dict(displayModeBar=False),
                            ),
                        ],
                        className="four columns chart_div"
                    ),

                    html.Div(
                        [
                            html.Div(id="map_history_title",
                                     children="Customer History",
                                     style={"textAlign": "center"}),
                            dcc.Graph(
                                id="map_history",
                                style={"height": "90%", "width": "98%"},
                                config=dict(displayModeBar=False),
                            ),
                        ],
                        className="four columns chart_div"
                    ),

                    html.Div(
                        [
                            html.Div(id="chart_metadata_title",
                                     children="Customer Metadata",
                                     style={"textAlign": "center"}),
                            dcc.Graph(
                                id="chart_metadata",
                                style={"height": "90%", "width": "98%"},
                                config=dict(displayModeBar=False),
                            ),
                        ],
                        className="four columns chart_div"
                    ),
                ],
                className="row",
                style={"marginTop": "5"},
            ),

            # table div
            html.Div(
                id="cust_table",
                className="row",
                style={
                    "maxHeight": "350px",
                    "overflowY": "scroll",
                    "padding": "8",
                    "marginTop": "5",
                    "backgroundColor":"white",
                    "border": "1px solid #C8D4E3",
                    "borderRadius": "3px"
                },
            ),

            html.Link(href="https://use.fontawesome.com/releases/v5.2.0/css/all.css",rel="stylesheet"),
            html.Link(href="https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css",rel="stylesheet"),
            html.Link(href="https://fonts.googleapis.com/css?family=Dosis", rel="stylesheet"),
            html.Link(href="https://fonts.googleapis.com/css?family=Open+Sans", rel="stylesheet"),
            html.Link(href="https://fonts.googleapis.com/css?family=Ubuntu", rel="stylesheet"),
            html.Link(href="https://cdn.rawgit.com/amadoukane96/8a8cfdac5d2cecad866952c52a70a50e/raw/cd5a9bf0b30856f4fc7e3812162c74bfc0ebe011/dash_crm.css", rel="stylesheet"),
        ],
        className="row",
        style={"margin": "0%"},
    )

    # -------------------------------------------------------------------- #
    #
    # Define the callbacks
    #
    # -------------------------------------------------------------------- #

    ## updates title of thresh slider based on value
    #@app.callback(
    #    Output("thresh_slider_output_container", "children"),
    #    [Input("thresh_slider", "value")]
    #)
    #def thresh_slider_callback(thresh):
    #    return 'Cutoff Probability: {:.1f}%'.format(thresh)

    # updates left indicator based on df updates
    @app.callback(
        [Output("left_indicator_title", "children"),
         Output("left_indicator", "children")],
        [Input("thresh_input", "value"), Input("current_df", "children")]
    )
    def left_indicator_callback(thresh, df):
        df = pd.read_json(df, orient="split")
        n_cutoffs = len(df.loc[df[p_cut_col] >= float(thresh)])
        return feature_total_str[p_cut_col], n_cutoffs

    # updates middle indicator based on df updates
    @app.callback(
        [Output("middle_indicator_title", "children"),
         Output("middle_indicator", "children")],
        [Input("thresh_input", "value"), Input("history_dropdown", "value"),
         Input("current_df", "children")]
    )
    def middle_indicator_callback(thresh, feature_name, df):
        df = pd.read_json(df, orient="split")
        df = df.loc[df[p_cut_col] >= float(thresh)]
        total = ( (df[feature_name] * df[p_cut_col] / 100).sum() *
                  factor_total[feature_name] )
        return feature_total_str[feature_name], '{:.1f}'.format(total)

    # updates right indicator based on df updates
    @app.callback(
        [Output("right_indicator_title", "children"),
         Output("right_indicator", "children")],
        [Input("thresh_input", "value"), Input("history_dropdown", "value"),
         Input("current_df", "children")]
    )
    def right_indicator_callback(thresh, feature_name, df):
        df = pd.read_json(df, orient="split")
        df = df.loc[df[p_cut_col] >= float(thresh)]
        total = ( (df[feature_name] * df[p_cut_col] / 100).sum() *
                  factor_total[feature_name] )
        n_cutoffs = len(df[p_cut_col])
        if n_cutoffs > 0:
            mean = total / n_cutoffs
            mean_str = '{:.1f}'.format(mean)
        else:
            mean_str = 'n/a'
        title_str = feature_mean_str[feature_name] + ' Per Cutoff'
        return title_str, mean_str

    # update cutoff map figure based on dropdown's value and df updates
    @app.callback(
        Output("map_cutoff", "figure"),
        [Input("thresh_input", "value"), Input("current_df", "children")],
    )
    def map_cutoff_callback(thresh, df):
        df = pd.read_json(df, orient="split")
        df = df.loc[df[p_cut_col] >= float(thresh)]
        return scatter_map(p_cut_col, df, mapbox_access_token, lat_center,
                           lon_center)

    # update history map title based on dropdown's value
    @app.callback(
        Output("map_history_title", "children"),
        [Input("history_dropdown", "value")],
    )
    def map_history_title_callback(feature_name):
        return feature_longname[feature_name]

    # update history map figure based on dropdown's value and df updates
    @app.callback(
        Output("map_history", "figure"),
        [Input("thresh_input", "value"), Input("history_dropdown", "value"),
         Input("current_df", "children")],
    )
    def map_history_callback(thresh, feature_name, df):
        df = pd.read_json(df, orient="split")
        df = df.loc[df[p_cut_col] >= float(thresh)]
        return scatter_map(feature_name, df, mapbox_access_token, lat_center,
                           lon_center)

    # update pie chart title based on dropdown's value
    @app.callback(
        Output("chart_metadata_title", "children"),
        [Input("metadata_dropdown", "value")],
    )
    def chart_metadata_title_callback(feature_name):
        return feature_longname[feature_name]

    # update pie chart figure based on dropdown's value and df updates
    @app.callback(
        Output("chart_metadata", "figure"),
        [Input("thresh_input", "value"), Input("metadata_dropdown", "value"),
         Input("current_df", "children")],
    )
    def chart_metadata_callback(thresh, feature_name, df):
        df = pd.read_json(df, orient="split")
        df = df.loc[df[p_cut_col] >= float(thresh)]
        return metadata_pie(feature_name, df)

    # update table based on dropdown's value and df updates
    @app.callback(
        Output("cust_table", "children"),
        [Input("thresh_input", "value"), Input("current_df", "children")],
    )
    def cust_table_callback(thresh, df):
        df = pd.read_json(df, orient="split")
        df = df.loc[df[p_cut_col] >= float(thresh)]
        column_list = ["meter_address", "p_cutoff",
                       "mean_charge_tot", "mean_charge_late",
                       "mean_vol", "f_late", "nCutPrior",
                       "municipality", "cust_type", "meter_size",
                      ]
        column_longnames = [feature_longname[col] for col in column_list]
        df = df.round({
                       'p_cutoff': 2,
                       'mean_charge_tot': 2,
                       'mean_charge_late': 2,
                       'mean_vol': 2,
                       'f_late': 2,
                       })
        df_new = pd.DataFrame(df[column_list].values, columns=column_longnames)
        return generate_table(df_new)

    ## re-read dataframe and convert to json
    #@app.callback(
    #    Output("current_df", "children"),
    #    [Input("refresh_button", "n_clicks")],
    #    [State("current_df", "children")],
    #)
    #def refresh_callback(n_clicks, current_df):
    #    if n_clicks > 0:
    #        df = read_df()
    #        return df.to_json(orient="split")
    #    return current_df

    # -------------------------------------------------------------------- #
    #
    # Start the dashboard
    #
    # -------------------------------------------------------------------- #
    # setting debug=True enables instant refreshing of the page
    # when changes are made to the files
    app.run_server(debug=True)

    return 0



