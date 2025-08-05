import pandas as pd
import numpy as np
import logging
from datetime import datetime

from config.config import PARAMS
from src.data.yfinance_data import calcular_media_movel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MacroEconomicModel:
    def __init__(self):
        self.params = PARAMS.copy()
        self._update_commodity_params()
        self.sensibilidade_setorial = self._load_sensibilidade_setorial()
        self.setores_por_cenario = self._load_setores_por_cenario()

    def _update_commodity_params(self):
        """
        Atualiza os parâmetros de commodities com médias móveis dinâmicas.
        """
        logging.info("Atualizando parâmetros de commodities com médias móveis.")
        precos_ideais = {
            "soja_ideal": calcular_media_movel("ZS=F", periodo="12mo", intervalo="1mo"),
            "milho_ideal": calcular_media_movel("ZC=F", periodo="12mo", intervalo="1mo"),
            "minerio_ideal": calcular_media_movel("TIO=F", periodo="12mo", intervalo="1mo"),
            "petroleo_ideal": calcular_media_movel("BZ=F", periodo="12mo", intervalo="1mo")
        }
        self.params.update({k: v for k, v in precos_ideais.items() if v is not None})

    def _load_sensibilidade_setorial(self):
        """
        Define a sensibilidade de cada setor aos indicadores macroeconômicos.
        """
        return {
            # Setores pró-cíclicos (beneficiam muito de crescimento/expansão, sensíveis ao PIB e commodities)
            'Consumo Discricionário': {
                'juros': -2, 'inflação': -1, 'dolar': -1, 'pib': 2.5,
                'commodities_agro': -0.5, 'commodities_minerio': -0.5, 'commodities_petroleo': -0.2
            },
            'Tecnologia': {
                'juros': -1.5, 'inflação': 0, 'dolar': -1, 'pib': 2,
                'commodities_agro': -0.2, 'commodities_minerio': -0.2, 'commodities_petroleo': 0
            },
            'Indústria e Bens de Capital': {
                'juros': -1, 'inflação': -0.5, 'dolar': -0.5, 'pib': 2.2,
                'commodities_agro': 0, 'commodities_minerio': 0.2, 'commodities_petroleo': 0
            },
            'Mineração e Siderurgia': {
                'juros': 0, 'inflação': 0, 'dolar': 2, 'pib': 1.2,
                'commodities_agro': 0, 'commodities_minerio': 2.5, 'commodities_petroleo': 0.6
            },
            'Petróleo, Gás e Biocombustíveis': {
                'juros': 0, 'inflação': 0, 'dolar': 1.5, 'pib': 1,
                'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 2.7
            },
            'Agronegócio': {
                'juros': -0.5, 'inflação': -0.6, 'dolar': 1.7, 'pib': 1.1,
                'commodities_agro': 2.7, 'commodities_minerio': 0, 'commodities_petroleo': 0.4
            },

            # Setores defensivos (resilientes em qualquer ciclo, mas pouco sensíveis positivamente ao PIB)
            'Saúde': {
                'juros': 0, 'inflação': 0, 'dolar': 0, 'pib': 0.6,
                'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0
            },
            'Consumo Básico': {
                'juros': 0.7, 'inflação': -1.2, 'dolar': -0.7, 'pib': 0.6,
                'commodities_agro': -0.2, 'commodities_minerio': -0.2, 'commodities_petroleo': -0.1
            },
            'Utilidades Públicas': {
                'juros': 1.2, 'inflação': 0.7, 'dolar': -0.6, 'pib': -0.6,
                'commodities_agro': -0.2, 'commodities_minerio': -0.2, 'commodities_petroleo': 0
            },
            'Energia Elétrica': {
                'juros': 0.5, 'inflação': 0.5, 'dolar': -0.7, 'pib': -0.7,
                'commodities_agro': -0.3, 'commodities_minerio': -0.2, 'commodities_petroleo': 0.1
            },

            # Setores financeiros (Bancos, Seguradoras, Bolsas)
            'Bancos': {
                'juros': 1.6, 'inflação': -0.1, 'dolar': -0.3, 'pib': 1.1,
                'commodities_agro': 0.3, 'commodities_minerio': 0.2, 'commodities_petroleo': 0
            },
            'Seguradoras': {
                'juros': 2, 'inflação': 0.2, 'dolar': 0, 'pib': 0.7,
                'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0
            },
            'Bolsas e Serviços Financeiros': {
                'juros': 1, 'inflação': 0, 'dolar': 0, 'pib': 1.5,
                'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0
            },

            # Outros setores típicos
            'Comunicação': {
                'juros': 0, 'inflação': 0, 'dolar': -0.3, 'pib': 0.5,
                'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0
            }
        }

    def _load_setores_por_cenario(self):
        """
        Mapeamento dos setores mais favorecidos em cada fase do ciclo macroeconômico.
        """
        return {
            "Expansão Forte": [
                'Consumo Discricionário', 'Tecnologia', 'Indústria e Bens de Capital',
                'Agronegócio', 'Mineração e Siderurgia', 'Petróleo, Gás e Biocombustíveis'
            ],
            "Expansão Moderada": [
                'Consumo Discricionário', 'Tecnologia', 'Indústria e Bens de Capital',
                'Agronegócio', 'Mineração e Siderurgia', 'Petróleo, Gás e Biocombustíveis',
                'Saúde'
            ],
            "Estável": [
                'Saúde', 'Bancos', 'Seguradoras', 'Bolsas e Serviços Financeiros',
                'Consumo Básico', 'Utilidades Públicas', 'Comunicação'
            ],
            "Contração Moderada": [
                'Bancos', 'Seguradoras', 'Consumo Básico', 'Utilidades Públicas',
                'Saúde', 'Energia Elétrica', 'Comunicação'
            ],
            "Contração Forte": [
                'Utilidades Públicas', 'Consumo Básico', 'Energia Elétrica', 'Saúde'
            ]
        }

    def pontuar_ipca(self, ipca):
        if ipca is None or pd.isna(ipca):
            return 0
        meta = self.params["ipca_meta"]
        tolerancia = self.params["ipca_tolerancia"]
        if meta - tolerancia <= ipca <= meta + tolerancia:
            return 10
        elif ipca <= meta + tolerancia + 1:
            return 5
        elif ipca > meta + tolerancia + 1:
            return 0
        else:
            return 3

    def pontuar_selic(self, selic):
        if selic is None or pd.isna(selic):
            return 0
        neutra = self.params["selic_neutra"]
        if abs(selic - neutra) <= 0.5:
            return 10
        elif selic > neutra and selic <= neutra + 2:
            return 4
        elif selic > neutra + 2:
            return 0
        else:
            return 6

    def pontuar_dolar(self, dolar):
        if dolar is None or pd.isna(dolar):
            return 0
        ideal = self.params["dolar_ideal"]
        desvio = abs(dolar - ideal)
        return max(0, 10 - desvio * 2)

    def pontuar_pib(self, pib):
        if pib is None or pd.isna(pib):
            return 0
        ideal = self.params["pib_ideal"]
        if pib >= ideal:
            return min(10, 8 + (pib - ideal) * 2)
        else:
            return max(0, 8 - (ideal - pib) * 3)

    def pontuar_soja(self, soja):
        if soja is None or pd.isna(soja):
            return 0
        ideal = self.params.get("soja_ideal", 13.0)
        if ideal is None or pd.isna(ideal):
            return 0
        desvio = abs(soja - ideal)
        return max(0, 10 - desvio * 1.5)

    def pontuar_milho(self, milho):
        if milho is None or pd.isna(milho):
            return 0
        ideal = self.params.get("milho_ideal", 5.5)
        if ideal is None or pd.isna(ideal):
            return 0
        desvio = abs(milho - ideal)
        return max(0, 10 - desvio * 2)

    def pontuar_soja_milho(self, soja, milho):
        return (self.pontuar_soja(soja) + self.pontuar_milho(milho)) / 2

    def pontuar_minerio(self, minerio):
        if minerio is None or pd.isna(minerio):
            return 0
        ideal = self.params.get("minerio_ideal", 100.0) # Valor padrão
        if ideal is None or pd.isna(ideal):
            return 0
        desvio = abs(minerio - ideal)
        return max(0, 10 - desvio * 0.1)

    def pontuar_petroleo(self, petroleo):
        if petroleo is None or pd.isna(petroleo):
            return 0
        ideal = self.params.get("petroleo_ideal", 80.0) # Valor padrão
        if ideal is None or pd.isna(ideal):
            return 0
        desvio = abs(petroleo - ideal)
        return max(0, 10 - desvio * 0.2)

    def _validate_macro_data(self, macro_data):
        """
        Garante que todos os indicadores macro necessários estejam presentes e válidos.
        """
        required_indicators = ["selic", "ipca", "dolar", "pib", "soja", "milho", "minerio", "petroleo"]
        for k in required_indicators:
            if k not in macro_data or macro_data[k] is None or (isinstance(macro_data[k], float) and pd.isna(macro_data[k])):
                logging.warning(f"Indicador macroeconômico '{k}' ausente ou inválido. Definindo como 0.0.")
                macro_data[k] = 0.0
        return macro_data

    def pontuar_macro(self, macro_data, pesos=None):
        """
        Calcula scores macroeconômicos normalizados e média ponderada.
        """
        macro_data = self._validate_macro_data(macro_data)
        score = {
            "juros": self.pontuar_selic(macro_data["selic"]),
            "inflação": self.pontuar_ipca(macro_data["ipca"]),
            "dolar": self.pontuar_dolar(macro_data["dolar"]),
            "pib": self.pontuar_pib(macro_data["pib"]),
            "commodities_agro": self.pontuar_soja_milho(macro_data["soja"], macro_data["milho"]),
            "commodities_minerio": self.pontuar_minerio(macro_data["minerio"]),
            "commodities_petroleo": self.pontuar_petroleo(macro_data["petroleo"]),
        }
        pesos = pesos or {k: 1 for k in score}
        total_peso = sum(pesos.values())
        media_global = sum(score[k] * pesos.get(k, 1) for k in score) / total_peso if total_peso > 0 else 0
        score["media_global"] = media_global
        return score

    def classificar_cenario_macro(self, macro_data):
        """
        Classifica o cenário macroeconômico com base nos scores dos indicadores.
        """
        score_macro = self.pontuar_macro(macro_data)
        
        score_ipca = score_macro.get("inflação", 0)
        score_selic = score_macro.get("juros", 0)
        score_dolar = score_macro.get("dolar", 0)
        score_pib = score_macro.get("pib", 0)
        
        core_score = score_ipca + score_selic + score_dolar + score_pib

        commodities_score = (
            score_macro.get("commodities_agro", 0) * 0.1 +
            score_macro.get("commodities_minerio", 0) * 0.1 +
            score_macro.get("commodities_petroleo", 0) * 0.1
        )

        total_score = core_score + commodities_score

        if total_score >= 38:
            return "Expansão Forte"
        elif total_score >= 32:
            return "Expansão Moderada"
        elif total_score >= 26:
            return "Estável"
        elif total_score >= 14:
            return "Contração Moderada"
        else:
            return "Contração Forte"

    def calcular_favorecimento_continuo(self, setor, score_macro):
        """
        Calcula o favorecimento contínuo do setor com base na sensibilidade setorial e scores macro.
        """
        if setor not in self.sensibilidade_setorial:
            logging.warning(f"Setor '{setor}' não encontrado na sensibilidade setorial. Retornando 0.")
            return 0
        sens = self.sensibilidade_setorial[setor]
        bruto = sum(score_macro.get(k, 0) * peso for k, peso in sens.items())
        return np.tanh(bruto / 5) * 2

    def get_favored_sectors(self, current_macro_scenario):
        """
        Retorna a lista de setores favorecidos para um dado cenário macroeconômico.
        """
        return self.setores_por_cenario.get(current_macro_scenario, [])

    def montar_historico_macro_setorial(self, tickers, setores_por_ticker, start_date_str='2015-01-01'):
        """
        Gera histórico de cenários macro e favorecimento setorial.
        """
        hoje = datetime.today()
        inicio = pd.to_datetime(start_date_str)
        final = hoje
        datas = pd.date_range(inicio, final, freq='M').normalize()

        # Simulação de dados macro históricos (substituir por dados reais)
        historico_macro_simulado = []
        for data in datas:
            # Aqui você precisaria de dados macro históricos reais para cada data
            # Por simplicidade, usaremos valores fixos ou aleatórios para demonstração
            # Em um cenário real, você buscaria esses dados de um banco de dados histórico
            macro_data_sim = {
                "ipca": np.random.uniform(2, 6),
                "selic": np.random.uniform(5, 15),
                "dolar": np.random.uniform(4.5, 6.0),
                "pib": np.random.uniform(0.5, 3.0),
                "petroleo": np.random.uniform(50, 100),
                "soja": np.random.uniform(10, 15),
                "milho": np.random.uniform(4, 7),
                "minerio": np.random.uniform(80, 150)
            }
            historico_macro_simulado.append(macro_data_sim)

        df_macro_hist = pd.DataFrame(historico_macro_simulado, index=datas)

        historico_favorecimento = []
        for data_idx, row in df_macro_hist.iterrows():
            macro_data_for_date = row.to_dict()
            cenario = self.classificar_cenario_macro(macro_data_for_date)
            score_macro = self.pontuar_macro(macro_data_for_date)
            for ticker in tickers:
                setor = setores_por_ticker.get(ticker, None)
                if setor:
                    favorecido = self.calcular_favorecimento_continuo(setor, score_macro)
                    historico_favorecimento.append({
                        "data": str(data_idx.date()),
                        "cenario": cenario,
                        "ticker": ticker,
                        "setor": setor,
                        "favorecido": favorecido
                    })
        return pd.DataFrame(historico_favorecimento)


