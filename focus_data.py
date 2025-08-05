import requests
import pandas as pd
import datetime
import logging
from tenacity import retry, wait_exponential, stop_after_attempt, RetriableError

from config.config import URL_OLINDA_API

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def buscar_projecoes_focus(indicador, ano=datetime.datetime.now().year):
    """
    Busca projeções de indicadores macroeconômicos do Boletim Focus do Banco Central.
    Implementa retentativas com backoff exponencial.

    Args:
        indicador (str): Nome do indicador (IPCA, Selic, PIB Total, Câmbio).
        ano (int): Ano de referência para as projeções.

    Returns:
        float: Mediana das projeções, ou None em caso de falha.
    """
    indicador_map = {
        "IPCA": "IPCA",
        "Selic": "Selic",
        "PIB Total": "PIB Total",
        "Câmbio": "Câmbio"
    }
    
    nome_indicador = indicador_map.get(indicador)
    if not nome_indicador:
        logging.error(f"Indicador '{indicador}' não reconhecido.")
        return None
    
    url = f"{URL_OLINDA_API}ExpectativasMercadoTop5Anuais?$top=100000&$filter=Indicador eq '{nome_indicador}'&$format=json&$select=Indicador,Data,DataReferencia,Mediana"
    
    try:
        logging.info(f"Buscando projeções do Focus para {indicador} em {ano}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        dados = response.json()["value"]
        
        if not dados:
            logging.warning(f"Nenhum dado retornado para {indicador} em {ano}")
            return None
        
        df = pd.DataFrame(dados)
        
        # Validação de colunas
        required_columns = ["DataReferencia", "Data", "Mediana"]
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Colunas necessárias não encontradas no retorno do Focus para {indicador}")
            return None
        
        df = df[df["DataReferencia"].str.contains(str(ano))]
        df = df.sort_values("Data", ascending=False)
        
        if df.empty:
            logging.warning(f"Nenhum dado encontrado para {indicador} em {ano}.")
            return None
        
        mediana = float(df.iloc[0]["Mediana"])
        logging.info(f"Projeção obtida para {indicador} em {ano}: {mediana}")
        return mediana
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro de requisição ao buscar {indicador} no Boletim Focus: {e}")
        raise RetriableError(f"Erro de requisição ao buscar {indicador}") from e
    except (ValueError, KeyError) as e:
        logging.error(f"Erro de parsing de dados para {indicador}: {e}")
        return None
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar {indicador} no Boletim Focus: {e}")
        return None

def obter_macro_focus():
    """
    Obtém todas as projeções macroeconômicas do Boletim Focus.
    """
    macro = {
        "ipca": buscar_projecoes_focus("IPCA"),
        "selic": buscar_projecoes_focus("Selic"),
        "pib": buscar_projecoes_focus("PIB Total"),
        "dolar": buscar_projecoes_focus("Câmbio")
    }
    return macro

