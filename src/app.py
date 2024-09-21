from dash import Dash, html, dcc, dash_table, callback, ctx, MATCH
from dash.dependencies import Input, Output, State
from dash.dash_table.Format import Format, Scheme, Symbol
import dash_bootstrap_components as dbc

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from datetime import date, datetime, timedelta

import yfinance as yf

from fredapi import Fred
fred = Fred(api_key='dde97fbd2c97cba90d59383ce937ccad')

import random

max_date = date.today() - timedelta(days=365)

def generate_portfolio_weights(n):
    weights = []
    for i in range(n):
        weights.append(random.random())
        
    # let's make the sum of all weights add up to 1
    weights = weights/np.sum(weights)
    return weights

def price_scaling(raw_prices_df):
    scaled_prices_df = raw_prices_df.copy()
    for i in raw_prices_df.columns[1:]:
        scaled_prices_df[i] = raw_prices_df[i]/raw_prices_df[i][0]
    return scaled_prices_df


def asset_allocation(df, weights, initial_investment):
    portfolio_df = df.copy()

    # Scale stock prices using the "price_scaling" function that we defined earlier (Make them all start at 1)
    scaled_df = price_scaling(df)

    for i, stock in enumerate(scaled_df.columns[1:]):
        portfolio_df[stock] = scaled_df[stock] * weights[i] * initial_investment

    # Sum up all values and place the result in a new column titled "portfolio value [$]" 
    # Note that we excluded the date column from this calculation
    portfolio_df['Portfolio Value [$]'] = portfolio_df[portfolio_df != 'Date'].sum(axis = 1, numeric_only = True)

    # Calculate the portfolio percentage daily return and replace NaNs with zeros
    portfolio_df['Portfolio Daily Return [%]'] = portfolio_df['Portfolio Value [$]'].pct_change(1) * 100 
    portfolio_df.replace(np.nan, 0, inplace = True)

    return portfolio_df


def simulation_engine(close_price_df ,weights, initial_investment, risk_free_rate):
    # Perform asset allocation using the random weights (sent as arguments to the function)
    portfolio_df = asset_allocation(close_price_df, weights, initial_investment)

    # Calculate the return on the investment 
    # Return on investment is calculated using the last final value of the portfolio compared to its initial value
    return_on_investment = ((portfolio_df['Portfolio Value [$]'][-1:] -
                             portfolio_df['Portfolio Value [$]'][0])/
                             portfolio_df['Portfolio Value [$]'][0]) * 100

    # Daily change of every stock in the portfolio (Note that we dropped the date, portfolio daily worth and daily % returns) 
    portfolio_daily_return_df = portfolio_df.drop(columns = ['Date', 'Portfolio Value [$]', 'Portfolio Daily Return [%]'])
    portfolio_daily_return_df = portfolio_daily_return_df.pct_change(1)

    # Portfolio Expected Return formula
    expected_portfolio_return = np.sum(weights * portfolio_daily_return_df.mean() ) * 252

    # Portfolio volatility (risk) formula
    # The risk of an asset is measured using the standard deviation which indicates the dispertion away from the mean
    # The risk of a portfolio is not a simple sum of the risks of the individual assets within the portfolio
    # Portfolio risk must consider correlations between assets within the portfolio which is indicated by the covariance 
    # The covariance determines the relationship between the movements of two random variables
    # When two stocks move together, they have a positive covariance when they move inversely, the have a negative covariance

    covariance = portfolio_daily_return_df.cov() * 252
    expected_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))

    # Check out the chart for the 10-years U.S. treasury at https://ycharts.com/indicators/10_year_treasury_rate
    rf = risk_free_rate/100 # Try to set the risk free rate of return to 1% (assumption)

    # Calculate Sharpe ratio
    sharpe_ratio = (expected_portfolio_return - rf)/expected_volatility 
    return expected_portfolio_return, expected_volatility, sharpe_ratio, portfolio_df['Portfolio Value [$]'][-1:].values[0], return_on_investment.values[0]

one_year_treasury_list = fred.get_series('DGS1')
oy_df = one_year_treasury_list.to_frame(name='Yield')

app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
server = app.server

app.layout = html.Div(id = "app-container",
    children=[html.H1("Efficency Frontier POC"),
        html.P("This is a POC analysis"),
        html.Br(),
        html.H2("Enter the number of simulations you want to run."),
        dcc.Input(value =500, type = 'number', id='monte-carlo-sims', persistence=True, persistence_type='session'),
        html.Br(),
        html.Br(),
        html.H2("Select stocks to be in the portfolio from the dropdown"),
        dcc.Dropdown(['AAPL', 'AMZN','CAT','DE', 'DUK', 'EXC', 'GOOGL', 'JNJ', 'JPM', 'META', 'PFE', 'PG', 'WEC', 'F', 'GM', 'COKE', 'Add New'],
                        ['AAPL', 'F', 'JPM']
                    , multi=True, persistence=True, persistence_type='session'
                     , id = 'stock-ticker-dropdown'),
        html.Br(),
        html.Br(),
        html.H2("Select a starting date.  The simulation will use data for one year."),
        dcc.DatePickerSingle(
            id='my-date-picker-range',
            min_date_allowed=date(2016,1, 1),
            max_date_allowed=max_date,
            date=date(2023, 5, 1),
            clearable=True
            ),
        html.P("The earliest date available is 1 year ago today"),
        html.Br(),
        html.P(id='treasury-rate-display'),
        html.Br(),
        dcc.Loading(dcc.Graph(id='stock-price-returns'), type='circle'),
        html.Br(),
        dcc.Loading(dcc.Graph(id='portfolio-line-graph'), type = 'graph'),
        html.Div(id='optimal-portfolio-stats'),
        html.Div(id='best-weights-table')
        ]
)

@callback(Output('stock-price-returns', 'figure'),
        Output('portfolio-line-graph', 'figure'),
        Output('optimal-portfolio-stats', 'children'),
        Output('treasury-rate-display', 'children'),
        Output('best-weights-table', 'children'),
        Input('my-date-picker-range', 'date'),
        Input('stock-ticker-dropdown', 'value'),
    Input('monte-carlo-sims', 'value'),
 )
def display_stock_output(start_date, tickers, sims):

    end_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=365)

    end_date = datetime.strftime(end_date, '%Y-%m-%d')

    end_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=365)

    end_date = datetime.strftime(end_date, '%Y-%m-%d')

    data = yf.download(tickers, start = start_date, end = end_date)

    closing_prices = data["Adj Close"]
    closing_prices.reset_index(inplace=True)

    new_closing_prices = closing_prices

    n = len(new_closing_prices.columns)-1

    rfr = oy_df.loc[start_date]['Yield']

    sim_runs = sims
    initial_investment = 100000000

    # Placeholder to store all weights
    weights_runs = np.zeros((sim_runs, n))

    # Placeholder to store all Sharpe ratios
    sharpe_ratio_runs = np.zeros(sim_runs)

    # Placeholder to store all expected returns
    expected_portfolio_returns_runs = np.zeros(sim_runs)

    # Placeholder to store all volatility values
    volatility_runs = np.zeros(sim_runs)

    # Placeholder to store all returns on investment
    return_on_investment_runs = np.zeros(sim_runs)

    # Placeholder to store all final portfolio values
    final_value_runs = np.zeros(sim_runs)

    # Placeholder to store all final duration values
    # final_duration_runs = np.zeros(sim_runs)


    for i in range(sim_runs):
        # Generate random weights 
        weights = generate_portfolio_weights(n)
        # Store the weights.
        weights_runs[i,:] = weights

        # Call "simulation_engine" function and store Sharpe ratio, return and volatility
        # Note that asset allocation is performed using the "asset_allocation" function  
        expected_portfolio_returns_runs[i], volatility_runs[i], sharpe_ratio_runs[i], final_value_runs[i], return_on_investment_runs[i] = simulation_engine(new_closing_prices,weights, initial_investment, rfr)

    best_weights = weights_runs[sharpe_ratio_runs.argmax(), :]

    optimal_portfolio_return, optimal_volatility, optimal_sharpe_ratio, highest_final_value, optimal_return_on_investment = simulation_engine(new_closing_prices,weights_runs[sharpe_ratio_runs.argmax(), :], initial_investment, rfr)

    optimal_portfolio_df = asset_allocation(new_closing_prices, best_weights, initial_investment)
    optimal_portfolio_df.round(2)

    sim_out_df = pd.DataFrame({'Volatility': volatility_runs.tolist(), 'Portfolio_Return': expected_portfolio_returns_runs.tolist(), 'Sharpe_Ratio': sharpe_ratio_runs.tolist() })
    sim_out_df['size'] = (sim_out_df['Sharpe_Ratio'] + 10)

    x_free = float(0)
    y_free = float(rfr/100)

    fig = px.scatter(sim_out_df, x = 'Volatility', y = 'Portfolio_Return', color = 'Sharpe_Ratio', size = 'size', hover_data = ['Sharpe_Ratio'] )
    fig.add_trace(go.Scatter(x = [x_free], y = [y_free], mode="markers", name = "Risk Free Rate", marker = dict(size=[15], color = 'green')))
    fig.update_layout(coloraxis_colorbar = dict(y = 0.7, dtick = 5))
    fig.update_layout({'plot_bgcolor': "white"})


    fig2= px.line(title = 'Total Portfolio Value [$]')
    
    # For loop that plots all stock prices in the pandas dataframe df
    # Note that index starts with 1 because we want to skip the date column

    for i in optimal_portfolio_df[['Date', 'Portfolio Value [$]']].columns[1:]:
        fig2.add_scatter(x = optimal_portfolio_df['Date'], y = optimal_portfolio_df[i], name = i)
        fig2.update_traces(line_width = 2)
        fig2.update_layout({'plot_bgcolor': "white"})


    risk_free_rate_text = ('''The 1 year treasury yield at the start is {:,.02}%.  Start date is {} and end date is {}.
'''.format(rfr, start_date, end_date))

    # separator = '<br>'
    # combine_string = separator.join(port_stats_text)

    port_stats_text = dcc.Markdown('''
    ## Best Portfolio Metrics Based on {:,} Monte Carlo Simulation Runs:
                                    
    - Portfolio Expected Annual Return = {:.02f}%
    - Portfolio Standard Deviation (Volatility) = {:.02f}%
    - Sharpe Ratio = {:.02f}
    - Final Value = ${:,.2f}
    - Return on Investment = {:.02f}%''' .format(sim_runs, optimal_portfolio_return * 100,optimal_volatility*100, optimal_sharpe_ratio, highest_final_value, optimal_return_on_investment))

    best_weights_df = optimal_portfolio_df.head(1)
    best_weights_df = pd.concat([best_weights_df,optimal_portfolio_df.tail(1)], ignore_index=True)
    ccolumn = []

    for col in best_weights_df.columns:
        types = best_weights_df[col].dtype

        if col == 'Portfolio Daily Return [%]':
            item = dict(id=col, name = col, type = 'numeric', format=Format(precision=2, scheme=Scheme.percentage))

            # ccolumn.append(item)

        elif types == "float64":
            item = dict(id=col, name = col, type = 'numeric', format=Format(precision=2, symbol=Symbol.yes, scheme=Scheme.fixed).group(True))

            ccolumn.append(item)
        else:
            item = dict(id=col, name = col, type='datetime') 
            ccolumn.append(item)



    return fig, fig2, port_stats_text, risk_free_rate_text, dash_table.DataTable(
        data=best_weights_df.to_dict('records'),
        columns=ccolumn,
        style_cell={'textAlign':'left'}
     
 
if __name__ =='__main__':
  app.run(debug=True, jupyter_mode="external")
