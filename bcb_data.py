import requests
import pandas as pd
import datetime
import logging
from tenacity import retry, wait_exponential, stop_after_attempt, RetriableError

from config.config import URL_BCB_API, CODIGO_SELIC_BCB, CODIGO_IPCA_BCB, CODIGO_DOLAR_BCB

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def get_bcb_hist(code, inicio, final):
    """
    Baixa dados históricos de séries temporais do Banco Central do Brasil (BCB) usando a API do BCB.
    Implementa retentativas com backoff exponencial.

    Args:
        code (int): Código da série temporal do BCB.
        inicio (str): Data de início no formato 'DD/MM/YYYY'.
        final (str): Data final no formato 'DD/MM/YYYY'.

    Returns:
        pd.Series: Série temporal com os valores e datas como índice, ou pd.Series vazia em caso de falha.
    Raises:
        RetriableError: Se todas as retentativas falharem.
    """
    url = URL_BCB_API.format(code=code) + f"&dataInicial={inicio}&dataFinal={final}"
    try:
        logging.info(f"Tentando buscar dados do BCB para o código {code} de {inicio} a {final}")
        r = requests.get(url, timeout=10) # Adicionado timeout
        r.raise_for_status()  # Levanta um HTTPError para códigos de status 4xx/5xx
        data = r.json()

        if not isinstance(data, list) or not data:
            logging.warning(f"Retorno vazio ou inválido da API BCB para código {code}: {data}")
            return pd.Series(dtype=float)

        df = pd.DataFrame(data)
        
        # Validação de colunas
        if 'data' not in df.columns or 'valor' not in df.columns:
            logging.error(f"Colunas 'data' ou 'valor' não encontradas no retorno do BCB para código {code}.")
            return pd.Series(dtype=float)

        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df['valor'] = df['valor'].str.replace(",", ".").astype(float)
        return df.set_index('data')['valor']
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro de requisição para o código {code}: {e}")
        raise RetriableError(f"Erro de requisição para o código {code}") from e
    except ValueError as e:
        logging.error(f"Erro de parsing JSON ou de dados para o código {code}: {e}")
        return pd.Series(dtype=float)
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar dados do BCB para o código {code}: {e}")
        return pd.Series(dtype=float)

def fetch_macro_bcb_data(start_date, end_date):
    """
    Busca dados históricos de Selic, IPCA e Dólar do BCB.
    """
    selic_hist = get_bcb_hist(CODIGO_SELIC_BCB, start_date.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y'))
    ipca_hist = get_bcb_hist(CODIGO_IPCA_BCB, start_date.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y'))
    dolar_hist = get_bcb_hist(CODIGO_DOLAR_BCB, start_date.strftime('%d/%m/%Y'), end_date.strftime('%d/%m/%Y'))
    return selic_hist, ipca_hist, dolar_hist


