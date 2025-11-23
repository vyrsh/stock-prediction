import yfinance as yf

# Define the ticker symbol (e.g., Apple Inc.)
ticker_symbol = "MSFT"

# Get data for a specific period
# Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
aapl_data = yf.Ticker(ticker_symbol).history(period="1mo")

# Print the data
print(aapl_data.head())
