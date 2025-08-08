import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from config import PARAMS # Alterado para importação direta
from yfinance_data import calcular_media_movel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MacroEconomicModel:
    def __init__(self):
        self.params = PARAMS.copy()
        self._update_commodity_params()
        self.sensibilidade_setorial = self._load_sensibilidade_setorial()
        self.setores_por_cenario = self._load_setores_por_cenario()
        self.regime_params = self._load_regime_parameters()

    def _update_commodity_params(self):
        logging.info("Atualizando parâmetros de commodities com médias móveis.")
        precos_ideais = {
            "soja_ideal": calcular_media_movel("ZS=F", periodo="5y", intervalo="1mo"),
            "milho_ideal": calcular_media_movel("ZC=F", periodo="5y", intervalo="1mo"),
            "minerio_ideal": calcular_media_movel("TIO=F", periodo="5y", intervalo="1mo"),
            "petroleo_ideal": calcular_media_movel("BZ=F", periodo="5y", intervalo="1mo")
        }
        self.params.update({k: v for k, v in precos_ideais.items() if v is not None})

    def _load_sensibilidade_setorial(self):
        return {
            'Consumo Discricionário': {'juros': -2, 'inflação': -1, 'dolar': -1, 'pib': 2.5,
                                     'commodities_agro': -0.5, 'commodities_minerio': -0.5, 'commodities_petroleo': -0.2},
            'Tecnologia': {'juros': -1.5, 'inflação': 0, 'dolar': -1, 'pib': 2,
                           'commodities_agro': -0.2, 'commodities_minerio': -0.2, 'commodities_petroleo': 0},
            'Indústria e Bens de Capital': {'juros': -1, 'inflação': -0.5, 'dolar': -0.5, 'pib': 2.2,
                                            'commodities_agro': 0, 'commodities_minerio': 0.2, 'commodities_petroleo': 0},
            'Mineração e Siderurgia': {'juros': 0, 'inflação': 0, 'dolar': 2, 'pib': 1.2,
                                       'commodities_agro': 0, 'commodities_minerio': 2.5, 'commodities_petroleo': 0.6},
            'Petróleo, Gás e Biocombustíveis': {'juros': 0, 'inflação': 0, 'dolar': 1.5, 'pib': 1,
                                                'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 2.7},
            'Agronegócio': {'juros': -0.5, 'inflação': -0.6, 'dolar': 1.7, 'pib': 1.1,
                            'commodities_agro': 2.7, 'commodities_minerio': 0, 'commodities_petroleo': 0.4},

            'Saúde': {'juros': 0, 'inflação': 0, 'dolar': 0, 'pib': 0.6,
                      'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0},
            'Consumo Básico': {'juros': 0.7, 'inflação': -1.2, 'dolar': -0.7, 'pib': 0.6,
                               'commodities_agro': -0.2, 'commodities_minerio': -0.2, 'commodities_petroleo': -0.1},
            'Utilidades Públicas': {'juros': 1.2, 'inflação': 0.7, 'dolar': -0.6, 'pib': -0.6,
                                    'commodities_agro': -0.2, 'commodities_minerio': -0.2, 'commodities_petroleo': 0},
            'Energia Elétrica': {'juros': 0.5, 'inflação': 0.5, 'dolar': -0.7, 'pib': -0.7,
                                 'commodities_agro': -0.3, 'commodities_minerio': -0.2, 'commodities_petroleo': 0.1},

            'Bancos': {'juros': 1.6, 'inflação': -0.1, 'dolar': -0.3, 'pib': 1.1,
                       'commodities_agro': 0.3, 'commodities_minerio': 0.2, 'commodities_petroleo': 0},
            'Seguradoras': {'juros': 2, 'inflação': 0.2, 'dolar': 0, 'pib': 0.7,
                            'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0},
            'Bolsas e Serviços Financeiros': {'juros': 1, 'inflação': 0, 'dolar': 0, 'pib': 1.5,
                                             'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0},

            'Comunicação': {'juros': 0, 'inflação': 0, 'dolar': -0.3, 'pib': 0.5,
                            'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0}
        }

    def _load_setores_por_cenario(self):
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

    def _load_regime_parameters(self):
        return {
            "Juros Altos": {
                "juros": 1.5, "inflação": 0.8, "dolar": 1.2, "pib": 0.7, 
                "commodities_agro": 0.9, "commodities_minerio": 0.9, "commodities_petroleo": 0.9
            },
            "Inflação Alta": {
                "juros": 1.2, "inflação": 1.5, "dolar": 1.1, "pib": 0.8, 
                "commodities_agro": 1.1, "commodities_minerio": 1.1, "commodities_petroleo": 1.1
            },
            "Crescimento Forte": {
                "juros": 0.8, "inflação": 0.9, "dolar": 0.8, "pib": 1.5, 
                "commodities_agro": 1.0, "commodities_minerio": 1.0, "commodities_petroleo": 1.0
            },
            "Estabilidade": {
                "juros": 1.0, "inflação": 1.0, "dolar": 1.0, "pib": 1.0, 
                "commodities_agro": 1.0, "commodities_minerio": 1.0, "commodities_petroleo": 1.0
            },
            "Recessão": {
                "juros": 0.7, "inflação": 0.7, "dolar": 1.3, "pib": 0.5, 
                "commodities_agro": 0.8, "commodities_minerio": 0.8, "commodities_petroleo": 0.8
            }
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
                logging.warning(f"Indicador macroeconômico \'{k}\' ausente ou inválido. Definindo como 0.0.")
                macro_data[k] = 0.0
        return macro_data

    def identify_macro_regime(self, macro_data):
        """
        Identifica o regime macroeconômico atual com base nos indicadores.
        Pode ser aprimorado com modelos de clustering (KMeans) ou Markov-Switching.
        """
        score_macro = self.pontuar_macro(macro_data)

        ipca_score = score_macro.get("inflação", 0)
        selic_score = score_macro.get("juros", 0)
        pib_score = score_macro.get("pib", 0)
        dolar_score = score_macro.get("dolar", 0)

        if ipca_score <= 7 and selic_score >= 7 and pib_score >= 7:
            return "Crescimento Forte"
        elif ipca_score <= 5 and selic_score <= 5 and pib_score <= 5:
            return "Recessão"
        elif ipca_score <= 5 and selic_score >= 7:
            return "Juros Altos"
        elif ipca_score >= 7 and selic_score <= 5:
            return "Inflação Alta"
        else:
            return "Estabilidade"

    def pontuar_macro(self, macro_data, pesos=None):
        """
        Calcula scores macroeconômicos normalizados e média ponderada.
        Aplica pesos de regime se um regime for identificado.
        """
        macro_data = self._validate_macro_data(macro_data)
        
        current_regime = self.identify_macro_regime(macro_data)
        logging.info(f"Regime macroeconômico identificado: {current_regime}")
        
        regime_pesos = self.regime_params.get(current_regime, {})

        score = {
            "juros": self.pontuar_selic(macro_data["selic"]),
            "inflação": self.pontuar_ipca(macro_data["ipca"]),
            "dolar": self.pontuar_dolar(macro_data["dolar"]),
            "pib": self.pontuar_pib(macro_data["pib"]),
            "commodities_agro": self.pontuar_soja_milho(macro_data["soja"], macro_data["milho"]),
            "commodities_minerio": self.pontuar_minerio(macro_data["minerio"]),
            "commodities_petroleo": self.pontuar_petroleo(macro_data["petroleo"]),
        }
        
        adjusted_score = {k: v * regime_pesos.get(k, 1.0) for k, v in score.items()}

        pesos = pesos or {k: 1 for k in adjusted_score}
        total_peso = sum(pesos.values())
        media_global = sum(adjusted_score[k] * pesos.get(k, 1) for k in adjusted_score) / total_peso if total_peso > 0 else 0
        adjusted_score["media_global"] = media_global
        return adjusted_score

    def classificar_cenario_macro(self, macro_data):
        """
        Classifica o cenário macroeconômico com base nos scores dos indicadores.
        Esta função agora pode ser mais influenciada pelo regime identificado.
        """
        score_macro = self.pontuar_macro(macro_data)
        
        score_ipca = score_macro.get("inflação", 0)
        score_selic = score_macro.get("juros", 0)
        score_dolar = score_macro.get("dolar", 0)
        score_pib = score_macro.get("pib", 0)
        
        current_regime = self.identify_macro_regime(macro_data)
        
        core_score = score_ipca + score_selic + score_dolar + score_pib

        commodities_score = (
            score_macro.get("commodities_agro", 0) * 0.1 +
            score_macro.get("commodities_minerio", 0) * 0.1 +
            score_macro.get("commodities_petroleo", 0) * 0.1
        )

        total_score = core_score + commodities_score

        if current_regime == "Recessão":
            if total_score >= 35:
                return "Expansão Forte"
            elif total_score >= 29:
                return "Expansão Moderada"
            elif total_score >= 23:
                return "Estável"
            elif total_score >= 12:
                return "Contração Moderada"
            else:
                return "Contração Forte"
        elif current_regime == "Crescimento Forte":
            if total_score >= 40:
                return "Expansão Forte"
            elif total_score >= 34:
                return "Expansão Moderada"
            elif total_score >= 28:
                return "Estável"
            elif total_score >= 16:
                return "Contração Moderada"
            else:
                return "Contração Forte"
        else:
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
        if setor not in self.sensibilidade_setorial:
            logging.warning(f"Setor \'{setor}\' não encontrado na sensibilidade setorial. Retornando 0.")
            return 0
        sens = self.sensibilidade_setorial[setor]
        bruto = sum(score_macro.get(k, 0) * peso for k, peso in sens.items())
        return np.tanh(bruto / 5) * 2

    def get_favored_sectors(self, current_macro_scenario):
        return self.setores_por_cenario.get(current_macro_scenario, [])

    def montar_historico_macro_setorial(self, tickers, setores_por_ticker, start_date_str='2015-01-01'):
        hoje = datetime.today()
        inicio = pd.to_datetime(start_date_str)
        final = hoje
        datas = pd.date_range(inicio, final, freq='M').normalize()

        historico_macro_simulado = []
        for data in datas:
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

    def predict_macro_trend(self, macro_data_history, periods=3):
        if not macro_data_history or len(macro_data_history) < 3:
            logging.warning("Histórico insuficiente para predição de tendência.")
            return None
        
        df_hist = pd.DataFrame(macro_data_history)
        
        trends = {}
        for indicator in ["selic", "ipca", "dolar", "pib"]:
            if indicator in df_hist.columns:
                ma_3 = df_hist[indicator].rolling(window=3).mean().iloc[-1]
                slope = df_hist[indicator].iloc[-1] - df_hist[indicator].iloc[-2] if len(df_hist) >= 2 else 0
                
                trends[indicator] = {
                    "current": df_hist[indicator].iloc[-1],
                    "ma_3": ma_3,
                    "slope": slope,
                    "trend": "up" if slope > 0.1 else "down" if slope < -0.1 else "stable"
                }
        
        return trends

    def adjust_scores_by_trend(self, macro_data, trend_data):
        if not trend_data:
            return self.pontuar_macro(macro_data)
        
        base_scores = self.pontuar_macro(macro_data)
        
        trend_adjustments = {
            "juros": 0,
            "inflação": 0,
            "dolar": 0,
            "pib": 0
        }
        
        for indicator, trend_info in trend_data.items():
            if indicator == "selic" and trend_info["trend"] == "up":
                trend_adjustments["juros"] += 0.5
            elif indicator == "selic" and trend_info["trend"] == "down":
                trend_adjustments["juros"] -= 0.5
            
            if indicator == "ipca" and trend_info["trend"] == "up":
                trend_adjustments["inflação"] -= 0.5
            elif indicator == "ipca" and trend_info["trend"] == "down":
                trend_adjustments["inflação"] += 0.5
            
            if indicator == "dolar" and trend_info["trend"] == "up":
                trend_adjustments["dolar"] -= 0.3
            elif indicator == "dolar" and trend_info["trend"] == "down":
                trend_adjustments["dolar"] += 0.3
            
            if indicator == "pib" and trend_info["trend"] == "up":
                trend_adjustments["pib"] += 0.5
            elif indicator == "pib" and trend_info["trend"] == "down":
                trend_adjustments["pib"] -= 0.5
        
        adjusted_scores = base_scores.copy()
        for key, adjustment in trend_adjustments.items():
            if key in adjusted_scores:
                adjusted_scores[key] = max(0, min(10, adjusted_scores[key] + adjustment))
        
        score_keys = [k for k in adjusted_scores.keys() if k != "media_global"]
        adjusted_scores["media_global"] = sum(adjusted_scores[k] for k in score_keys) / len(score_keys)
        
        return adjusted_scores

    def calculate_volatility_adjustment(self, macro_data_history):
        if not macro_data_history or len(macro_data_history) < 5:
            return 1.0
        
        df_hist = pd.DataFrame(macro_data_history)
        
        volatilities = {}
        for indicator in ["selic", "ipca", "dolar", "pib"]:
            if indicator in df_hist.columns:
                vol = df_hist[indicator].tail(5).std()
                volatilities[indicator] = vol
        
        if volatilities:
            avg_vol = np.mean(list(volatilities.values()))
            vol_factor = max(0.7, 1 - (avg_vol / 10))
            return vol_factor
        
        return 1.0

    def enhanced_pontuar_macro(self, macro_data, macro_data_history=None):
        base_scores = self.pontuar_macro(macro_data)
        
        if not macro_data_history:
            return base_scores
        
        trends = self.predict_macro_trend(macro_data_history)
        
        trend_adjusted_scores = self.adjust_scores_by_trend(macro_data, trends)
        
        vol_factor = self.calculate_volatility_adjustment(macro_data_history)
        
        final_scores = {}
        for key, score in trend_adjusted_scores.items():
            if key != "media_global":
                final_scores[key] = score * vol_factor
            else:
                final_scores[key] = score * vol_factor
        
        logging.info(f"Scores ajustados por tendência e volatilidade. Fator de volatilidade: {vol_factor:.3f}")
        
        return final_scores

