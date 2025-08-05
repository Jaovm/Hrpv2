import pandas as pd
import numpy as np
import logging
from datetime import datetime

from src.data.yfinance_data import obter_preco_atual, obter_preco_alvo
from src.models.macro_model import MacroEconomicModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AssetAnalyzer:
    """
    Classe para análise de ativos individuais e cálculo de scores.
    """
    
    def __init__(self):
        self.macro_model = MacroEconomicModel()
        self.setores_por_ticker = self._load_setores_por_ticker()
        
    def _load_setores_por_ticker(self):
        """
        Carrega o mapeamento de tickers para setores.
        Em um ambiente de produção, isso viria de um banco de dados ou arquivo de configuração.
        """
        return {
            # Bancos
            'ITUB4.SA': 'Bancos', 'BBDC4.SA': 'Bancos', 'SANB11.SA': 'Bancos',
            'BBAS3.SA': 'Bancos', 'ABCB4.SA': 'Bancos', 'BRSR6.SA': 'Bancos',
            'BMGB4.SA': 'Bancos', 'BPAC11.SA': 'Bancos', 'ITSA4.SA': 'Bancos',
            
            # Seguradoras
            'BBSE3.SA': 'Seguradoras', 'PSSA3.SA': 'Seguradoras',
            'SULA11.SA': 'Seguradoras', 'CXSE3.SA': 'Seguradoras',
            
            # Bolsas e Serviços Financeiros
            'B3SA3.SA': 'Bolsas e Serviços Financeiros',
            'XPBR31.SA': 'Bolsas e Serviços Financeiros',
            
            # Energia Elétrica
            'EGIE3.SA': 'Energia Elétrica', 'CPLE6.SA': 'Energia Elétrica',
            'TAEE11.SA': 'Energia Elétrica', 'CMIG4.SA': 'Energia Elétrica',
            'AURE3.SA': 'Energia Elétrica', 'CPFE3.SA': 'Energia Elétrica',
            'AESB3.SA': 'Energia Elétrica',
            
            # Petróleo, Gás e Biocombustíveis
            'PETR4.SA': 'Petróleo, Gás e Biocombustíveis',
            'PRIO3.SA': 'Petróleo, Gás e Biocombustíveis',
            'RECV3.SA': 'Petróleo, Gás e Biocombustíveis',
            'RRRP3.SA': 'Petróleo, Gás e Biocombustíveis',
            'UGPA3.SA': 'Petróleo, Gás e Biocombustíveis',
            'VBBR3.SA': 'Petróleo, Gás e Biocombustíveis',
            
            # Mineração e Siderurgia
            'VALE3.SA': 'Mineração e Siderurgia', 'CSNA3.SA': 'Mineração e Siderurgia',
            'GGBR4.SA': 'Mineração e Siderurgia', 'CMIN3.SA': 'Mineração e Siderurgia',
            'GOAU4.SA': 'Mineração e Siderurgia', 'BRAP4.SA': 'Mineração e Siderurgia',
            
            # Indústria e Bens de Capital
            'WEGE3.SA': 'Indústria e Bens de Capital', 'RANI3.SA': 'Indústria e Bens de Capital',
            'KLBN11.SA': 'Indústria e Bens de Capital', 'SUZB3.SA': 'Indústria e Bens de Capital',
            'UNIP6.SA': 'Indústria e Bens de Capital', 'KEPL3.SA': 'Indústria e Bens de Capital',
            'TUPY3.SA': 'Indústria e Bens de Capital',
            
            # Agronegócio
            'AGRO3.SA': 'Agronegócio', 'SLCE3.SA': 'Agronegócio',
            'SMTO3.SA': 'Agronegócio', 'CAML3.SA': 'Agronegócio',
            'RAIZ4.SA': 'Agronegócio',
            
            # Saúde
            'HAPV3.SA': 'Saúde', 'FLRY3.SA': 'Saúde', 'RDOR3.SA': 'Saúde',
            'QUAL3.SA': 'Saúde', 'RADL3.SA': 'Saúde', 'ANIM3.SA': 'Saúde',
            'AZEV4.SA': 'Saúde', 'PETZ3.SA': 'Saúde', 'SIMH3.SA': 'Saúde',
            'ALOS3.SA': 'Saúde', 'VIVA3.SA': 'Saúde', 'HYPE3.SA': 'Saúde',
            'PMAM3.SA': 'Saúde',
            
            # Tecnologia
            'TOTS3.SA': 'Tecnologia', 'POSI3.SA': 'Tecnologia',
            'LINX3.SA': 'Tecnologia', 'LWSA3.SA': 'Tecnologia',
            'COGN3.SA': 'Tecnologia', 'AMOB3.SA': 'Tecnologia',
            'IFCM3.SA': 'Tecnologia', 'SMFT3.SA': 'Tecnologia',
            'IGTI11.SA': 'Tecnologia', 'YDUQ3.SA': 'Tecnologia',
            'ECOR3.SA': 'Tecnologia', 'DXCO3.SA': 'Tecnologia',
            'LJQQ3.SA': 'Tecnologia', 'RCSL4.SA': 'Tecnologia',
            'IRBR3.SA': 'Tecnologia',
            
            # Consumo Discricionário
            'MGLU3.SA': 'Consumo Discricionário', 'LREN3.SA': 'Consumo Discricionário',
            'RENT3.SA': 'Consumo Discricionário', 'ARZZ3.SA': 'Consumo Discricionário',
            'ALPA4.SA': 'Consumo Discricionário', 'CRFB3.SA': 'Consumo Discricionário',
            'BEEF3.SA': 'Consumo Discricionário', 'AZUL4.SA': 'Consumo Discricionário',
            'CVCB3.SA': 'Consumo Discricionário', 'VAMO3.SA': 'Consumo Discricionário',
            'MRVE3.SA': 'Consumo Discricionário', 'RAPT4.SA': 'Consumo Discricionário',
            'MOVI3.SA': 'Consumo Discricionário', 'GFSA3.SA': 'Consumo Discricionário',
            'AMER3.SA': 'Consumo Discricionário', 'EZTC3.SA': 'Consumo Discricionário',
            'GOLL4.SA': 'Consumo Discricionário',
            
            # Consumo Básico
            'ABEV3.SA': 'Consumo Básico', 'NTCO3.SA': 'Consumo Básico',
            'PCAR3.SA': 'Consumo Básico', 'MDIA3.SA': 'Consumo Básico',
            'MRFG3.SA': 'Consumo Básico', 'JBSS3.SA': 'Consumo Básico',
            'BRFS3.SA': 'Consumo Básico', 'CBAV3.SA': 'Consumo Básico',
            
            # Comunicação
            'VIVT3.SA': 'Comunicação', 'TIMS3.SA': 'Comunicação',
            'OIBR3.SA': 'Comunicação',
            
            # Utilidades Públicas
            'SBSP3.SA': 'Utilidades Públicas', 'SAPR11.SA': 'Utilidades Públicas',
            'SAPR3.SA': 'Utilidades Públicas', 'SAPR4.SA': 'Utilidades Públicas',
            'CSMG3.SA': 'Utilidades Públicas', 'ALUP11.SA': 'Utilidades Públicas',
            'CCRO3.SA': 'Utilidades Públicas',
            
            # Outros
            'CSAN3.SA': 'Energia Elétrica', 'USIM5.SA': 'Mineração e Siderurgia',
            'ELET3.SA': 'Energia Elétrica', 'EQTL3.SA': 'Energia Elétrica',
            'POMO4.SA': 'Indústria e Bens de Capital', 'RAIL3.SA': 'Indústria e Bens de Capital',
            'BRAV3.SA': 'Bancos', 'PETR3.SA': 'Petróleo, Gás e Biocombustíveis',
            'ENEV3.SA': 'Energia Elétrica', 'CPLE3.SA': 'Energia Elétrica',
            'SRNA3.SA': 'Indústria e Bens de Capital', 'EMBR3.SA': 'Indústria e Bens de Capital',
            'MULT3.SA': 'Bancos', 'CYRE3.SA': 'Indústria e Bens de Capital',
            'STBP3.SA': 'Bancos', 'GMAT3.SA': 'Indústria e Bens de Capital',
            'CEAB3.SA': 'Indústria e Bens de Capital', 'ENGI11.SA': 'Energia Elétrica',
            'JHSF3.SA': 'Indústria e Bens de Capital', 'INTB3.SA': 'Indústria e Bens de Capital',
            'BRKM5.SA': 'Indústria e Bens de Capital', 'MMXM3.SA': 'Indústria e Bens de Capital',
            'BBDC3.SA': 'Bancos', 'BHIA3.SA': 'Bancos'
        }
    
    def calcular_score(self, preco_atual, preco_alvo, favorecimento_score, 
                      ticker, setor, macro_data, usar_pesos_macro=True, return_details=False):
        """
        Calcula o score final de um ativo baseado em preço-alvo e favorecimento macroeconômico.
        
        Args:
            preco_atual (float): Preço atual do ativo
            preco_alvo (float): Preço-alvo do ativo
            favorecimento_score (float): Score de favorecimento macroeconômico
            ticker (str): Ticker do ativo
            setor (str): Setor do ativo
            macro_data (dict): Dados macroeconômicos
            usar_pesos_macro (bool): Se deve usar pesos macroeconômicos
            return_details (bool): Se deve retornar detalhes do cálculo
            
        Returns:
            float or tuple: Score final ou (score, detalhes)
        """
        try:
            if preco_atual is None or preco_atual <= 0:
                logging.warning(f"Preço atual inválido para {ticker}")
                return 0 if not return_details else (0, "Preço atual inválido")
            
            if preco_alvo is None or preco_alvo <= 0:
                logging.warning(f"Preço-alvo inválido para {ticker}")
                return 0 if not return_details else (0, "Preço-alvo inválido")
            
            # Calcular potencial de valorização
            upside_potential = (preco_alvo - preco_atual) / preco_atual
            
            # Score base do potencial de valorização (0-10)
            if upside_potential > 0.5:  # >50% de upside
                score_upside = 10
            elif upside_potential > 0.3:  # 30-50% de upside
                score_upside = 8
            elif upside_potential > 0.15:  # 15-30% de upside
                score_upside = 6
            elif upside_potential > 0.05:  # 5-15% de upside
                score_upside = 4
            elif upside_potential > -0.05:  # -5% a 5%
                score_upside = 2
            else:  # <-5% (downside)
                score_upside = 0
            
            # Aplicar favorecimento macroeconômico se solicitado
            if usar_pesos_macro:
                # Normalizar favorecimento_score (-2 a 2) para multiplicador (0.5 a 1.5)
                macro_multiplier = 0.5 + (favorecimento_score + 2) / 4
                macro_multiplier = max(0.5, min(1.5, macro_multiplier))
                score_final = score_upside * macro_multiplier
            else:
                score_final = score_upside
                macro_multiplier = 1.0
            
            # Limitar score final entre 0 e 10
            score_final = max(0, min(10, score_final))
            
            if return_details:
                detalhes = {
                    "preco_atual": preco_atual,
                    "preco_alvo": preco_alvo,
                    "upside_potential": upside_potential,
                    "score_upside": score_upside,
                    "favorecimento_score": favorecimento_score,
                    "macro_multiplier": macro_multiplier,
                    "score_final": score_final
                }
                return score_final, detalhes
            
            return score_final
            
        except Exception as e:
            logging.error(f"Erro ao calcular score para {ticker}: {e}")
            return 0 if not return_details else (0, f"Erro: {str(e)}")
    
    def gerar_ranking_acoes(self, carteira, macro_data, usar_pesos_macro=True):
        """
        Gera ranking de ações baseado em scores de preço-alvo e favorecimento macro.
        
        Args:
            carteira (dict): Dicionário com tickers da carteira
            macro_data (dict): Dados macroeconômicos atuais
            usar_pesos_macro (bool): Se deve usar pesos macroeconômicos
            
        Returns:
            pd.DataFrame: DataFrame com ranking das ações
        """
        try:
            score_macro = self.macro_model.pontuar_macro(macro_data)
            resultados = []
            
            for ticker in carteira.keys():
                setor = self.setores_por_ticker.get(ticker)
                if setor is None:
                    logging.warning(f"Setor não encontrado para {ticker}. Ignorando.")
                    continue
                
                preco_atual = obter_preco_atual(ticker)
                preco_alvo = obter_preco_alvo(ticker)
                
                if preco_atual is None or preco_alvo is None or preco_atual == 0:
                    logging.warning(f"Dados insuficientes para {ticker}. Ignorando.")
                    continue
                
                favorecimento_score = self.macro_model.calcular_favorecimento_continuo(setor, score_macro)
                score, detalhe = self.calcular_score(
                    preco_atual, preco_alvo, favorecimento_score, 
                    ticker, setor, macro_data, usar_pesos_macro, return_details=True
                )
                
                resultados.append({
                    "ticker": ticker,
                    "setor": setor,
                    "preco_atual": preco_atual,
                    "preco_alvo": preco_alvo,
                    "upside_potential": detalhe.get("upside_potential", 0),
                    "favorecimento_macro": favorecimento_score,
                    "score": score,
                    "detalhe": detalhe
                })
            
            if not resultados:
                logging.warning("Nenhum resultado válido encontrado para o ranking")
                return pd.DataFrame()
            
            df = pd.DataFrame(resultados).sort_values(by="score", ascending=False)
            logging.info(f"Ranking gerado com {len(df)} ativos")
            return df
            
        except Exception as e:
            logging.error(f"Erro ao gerar ranking de ações: {e}")
            return pd.DataFrame()
    
    def filtrar_ativos_validos(self, carteira, macro_data, min_score=3.0):
        """
        Filtra ativos válidos baseado em critérios de qualidade.
        
        Args:
            carteira (dict): Dicionário com tickers da carteira
            macro_data (dict): Dados macroeconômicos atuais
            min_score (float): Score mínimo para considerar o ativo válido
            
        Returns:
            list: Lista de tickers válidos
        """
        try:
            ranking_df = self.gerar_ranking_acoes(carteira, macro_data)
            
            if ranking_df.empty:
                return []
            
            # Filtrar por score mínimo
            ativos_validos = ranking_df[ranking_df['score'] >= min_score]['ticker'].tolist()
            
            # Filtrar por dados válidos
            ativos_com_dados = []
            for ticker in ativos_validos:
                preco_atual = obter_preco_atual(ticker)
                preco_alvo = obter_preco_alvo(ticker)
                if preco_atual and preco_alvo and preco_atual > 0 and preco_alvo > 0:
                    ativos_com_dados.append(ticker)
            
            logging.info(f"Filtrados {len(ativos_com_dados)} ativos válidos de {len(carteira)} originais")
            return ativos_com_dados
            
        except Exception as e:
            logging.error(f"Erro ao filtrar ativos válidos: {e}")
            return []
    
    def calcular_scores_macro_por_ticker(self, tickers, macro_data):
        """
        Calcula scores macroeconômicos para uma lista de tickers.
        
        Args:
            tickers (list): Lista de tickers
            macro_data (dict): Dados macroeconômicos
            
        Returns:
            dict: Scores macro por ticker
        """
        try:
            score_macro = self.macro_model.pontuar_macro(macro_data)
            scores_por_ticker = {}
            
            for ticker in tickers:
                setor = self.setores_por_ticker.get(ticker)
                if setor:
                    favorecimento = self.macro_model.calcular_favorecimento_continuo(setor, score_macro)
                    scores_por_ticker[ticker] = favorecimento
                else:
                    scores_por_ticker[ticker] = 0.0
                    
            return scores_por_ticker
            
        except Exception as e:
            logging.error(f"Erro ao calcular scores macro por ticker: {e}")
            return {}
    
    def get_sector_distribution(self, tickers):
        """
        Retorna a distribuição setorial de uma lista de tickers.
        
        Args:
            tickers (list): Lista de tickers
            
        Returns:
            pd.Series: Contagem por setor
        """
        try:
            setores = [self.setores_por_ticker.get(ticker, 'Desconhecido') for ticker in tickers]
            return pd.Series(setores).value_counts()
        except Exception as e:
            logging.error(f"Erro ao calcular distribuição setorial: {e}")
            return pd.Series()

