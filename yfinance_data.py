import yfinance as yf
import pandas as pd
import logging
from tenacity import retry, wait_exponential, stop_after_attempt, RetriableError

from config.config import TICKER_PETROLEO, TICKER_SOJA, TICKER_MILHO, TICKER_MINERIO_FERRO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def obter_preco_yf(ticker, nome="Ativo", period="5d"):
    """
    Obtém o preço de fechamento mais recente de um ticker do Yahoo Finance.
    Implementa retentativas com backoff exponencial.

    Args:
        ticker (str): Ticker do ativo.
        nome (str): Nome do ativo para logging.
        period (str): Período de dados a buscar.

    Returns:
        float: Preço de fechamento mais recente, ou None em caso de falha.
    """
    try:
        logging.info(f"Buscando preço de {nome} ({ticker})")
        dados = yf.Ticker(ticker).history(period=period)
        
        if dados.empty or 'Close' not in dados.columns:
            logging.warning(f"Dados vazios ou coluna 'Close' não encontrada para {nome} ({ticker})")
            return None
        
        preco = float(dados['Close'].dropna().iloc[-1])
        logging.info(f"Preço obtido para {nome} ({ticker}): {preco}")
        return preco
    except Exception as e:
        logging.error(f"Erro ao obter preço de {nome} ({ticker}): {e}")
        raise RetriableError(f"Erro ao obter preço de {nome} ({ticker})") from e

def obter_preco_petroleo_hist(start, end):
    """
    Baixa preço histórico mensal do petróleo Brent (BZ=F) do Yahoo Finance.
    """
    try:
        logging.info(f"Buscando histórico de petróleo de {start} a {end}")
        df = yf.download(TICKER_PETROLEO, start=start, end=end, interval="1mo", progress=False)
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            return df['Close']
        return pd.Series(dtype=float)
    except Exception as e:
        logging.error(f"Erro ao obter histórico de petróleo: {e}")
        return pd.Series(dtype=float)

def obter_preco_atual(ticker):
    """
    Obtém o preço atual de um ticker.
    """
    try:
        dados = yf.Ticker(ticker).history(period="1d")
        if not dados.empty:
            return dados['Close'].iloc[-1]
    except Exception as e:
        logging.warning(f"Erro ao obter preço atual de {ticker}: {e}")
    return None

def obter_preco_alvo(ticker):
    """
    Obtém o preço-alvo médio de um ticker do Yahoo Finance.
    """
    try:
        return yf.Ticker(ticker).info.get('targetMeanPrice', None)
    except Exception as e:
        logging.warning(f"Erro ao obter preço-alvo de {ticker}: {e}")
        return None

def calcular_media_movel(ticker, periodo="12mo", intervalo="1mo"):
    """
    Calcula a média móvel do preço de um ativo.
    """
    try:
        dados = yf.download(ticker, period=periodo, interval=intervalo, progress=False)
        if not dados.empty:
            media_movel = float(dados['Close'].mean())
            return media_movel
        else:
            logging.warning(f"Dados históricos indisponíveis para {ticker}.")
            return None
    except Exception as e:
        logging.error(f"Erro ao calcular média móvel para {ticker}: {e}")
        return None

