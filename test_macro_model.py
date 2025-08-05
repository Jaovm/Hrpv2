import unittest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.models.macro_model import MacroEconomicModel

class TestMacroEconomicModel(unittest.TestCase):

    def setUp(self):
        self.model = MacroEconomicModel()
        self.mock_macro_data = {
            "selic": 10.0,
            "ipca": 4.0,
            "dolar": 5.0,
            "pib": 2.5,
            "soja": 12.0,
            "milho": 6.0,
            "minerio": 120.0,
            "petroleo": 85.0
        }

    def test_pontuar_ipca(self):
        self.assertEqual(self.model.pontuar_ipca(3.0), 10) # Dentro da meta
        self.assertEqual(self.model.pontuar_ipca(4.0), 5)  # Acima da meta, mas perto
        self.assertEqual(self.model.pontuar_ipca(6.0), 0)  # Muito acima da meta
        self.assertEqual(self.model.pontuar_ipca(1.0), 3)  # Abaixo da meta
        self.assertEqual(self.model.pontuar_ipca(None), 0)

    def test_pontuar_selic(self):
        self.assertEqual(self.model.pontuar_selic(7.0), 10) # Neutra
        self.assertEqual(self.model.pontuar_selic(9.0), 4)  # Acima da neutra
        self.assertEqual(self.model.pontuar_selic(12.0), 0) # Muito acima
        self.assertEqual(self.model.pontuar_selic(5.0), 6)  # Abaixo
        self.assertEqual(self.model.pontuar_selic(None), 0)

    def test_pontuar_dolar(self):
        self.assertEqual(self.model.pontuar_dolar(5.30), 10) # Ideal
        self.assertEqual(self.model.pontuar_dolar(5.80), 9)  # Um pouco acima
        self.assertEqual(self.model.pontuar_dolar(6.30), 7)  # Mais acima
        self.assertEqual(self.model.pontuar_dolar(None), 0)

    def test_pontuar_pib(self):
        self.assertEqual(self.model.pontuar_pib(2.0), 8)   # Ideal
        self.assertEqual(self.model.pontuar_pib(3.0), 10)  # Acima do ideal
        self.assertEqual(self.model.pontuar_pib(0.5), 3.5) # Abaixo do ideal
        self.assertEqual(self.model.pontuar_pib(None), 0)

    def test_pontuar_soja(self):
        with patch("src.models.macro_model.calcular_media_movel", return_value=12.0):
            self.model._update_commodity_params()
            self.assertEqual(self.model.pontuar_soja(12.0), 10)
            self.assertEqual(self.model.pontuar_soja(13.0), 8.5)
            self.assertEqual(self.model.pontuar_soja(None), 0)

    def test_pontuar_milho(self):
        with patch("src.models.macro_model.calcular_media_movel", return_value=5.5):
            self.model._update_commodity_params()
            self.assertEqual(self.model.pontuar_milho(5.5), 10)
            self.assertEqual(self.model.pontuar_milho(6.0), 9)
            self.assertEqual(self.model.pontuar_milho(None), 0)

    def test_pontuar_macro(self):
        scores = self.model.pontuar_macro(self.mock_macro_data)
        self.assertIn("media_global", scores)
        self.assertGreater(scores["media_global"], 0)

    def test_classificar_cenario_macro(self):
        cenario = self.model.classificar_cenario_macro(self.mock_macro_data)
        self.assertIn(cenario, ["Expansão Forte", "Expansão Moderada", "Estável", "Contração Moderada", "Contração Forte"])

    def test_calcular_favorecimento_continuo(self):
        score_macro = self.model.pontuar_macro(self.mock_macro_data)
        favorecimento = self.model.calcular_favorecimento_continuo("Agronegócio", score_macro)
        self.assertIsInstance(favorecimento, float)

    @patch("src.models.macro_model.pd.to_datetime", return_value=pd.Timestamp("2023-01-01"))
    @patch("src.models.macro_model.pd.date_range", return_value=pd.to_datetime(["2023-01-31", "2023-02-28"]))
    def test_montar_historico_macro_setorial(self, mock_date_range, mock_to_datetime):
        tickers = ["AGRO3.SA"]
        setores_por_ticker = {"AGRO3.SA": "Agronegócio"}
        df_hist = self.model.montar_historico_macro_setorial(tickers, setores_por_ticker)
        self.assertFalse(df_hist.empty)
        self.assertIn("data", df_hist.columns)
        self.assertIn("cenario", df_hist.columns)
        self.assertIn("ticker", df_hist.columns)
        self.assertIn("setor", df_hist.columns)
        self.assertIn("favorecido", df_hist.columns)

if __name__ == "__main__":
    unittest.main()


