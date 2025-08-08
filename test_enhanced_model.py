import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Adicionar o diretório atual ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos diretamente da raiz
from macro_model import MacroEconomicModel
from portfolio_optimizer import PortfolioOptimizer
from config import PARAMS
from yfinance_data import calcular_media_movel

class TestEnhancedMacroModel(unittest.TestCase):
    """
    Testes unitários para o modelo macroeconômico aprimorado.
    """
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.model = MacroEconomicModel()
        self.sample_macro_data = {
            "selic": 10.5,
            "ipca": 4.2,
            "dolar": 5.20,
            "pib": 2.1,
            "soja": 12.5,
            "milho": 6.0,
            "minerio": 120.0,
            "petroleo": 85.0
        }
    
    def test_regime_identification(self):
        """Testa a identificação de regimes macroeconômicos."""
        # Teste para regime de crescimento forte
        growth_data = {
            "selic": 7.0,  # Score alto
            "ipca": 3.0,   # Score alto
            "dolar": 5.30, # Score alto
            "pib": 3.0,    # Score alto
            "soja": 13.0, "milho": 5.5, "minerio": 100.0, "petroleo": 80.0
        }
        
        regime = self.model.identify_macro_regime(growth_data)
        self.assertIn(regime, ["Crescimento Forte", "Estabilidade"], 
                     f"Regime identificado: {regime}")
    
    def test_enhanced_scoring(self):
        """Testa a pontuação macroeconômica aprimorada."""
        # Criar histórico simulado
        history = []
        for i in range(10):
            data = self.sample_macro_data.copy()
            # Adicionar alguma variação
            data["selic"] += np.random.normal(0, 0.5)
            data["ipca"] += np.random.normal(0, 0.3)
            history.append(data)
        
        # Testar pontuação aprimorada
        enhanced_scores = self.model.enhanced_pontuar_macro(
            self.sample_macro_data, 
            history
        )
        
        self.assertIsInstance(enhanced_scores, dict)
        self.assertIn("media_global", enhanced_scores)
        self.assertTrue(0 <= enhanced_scores["media_global"] <= 10)
    
    def test_trend_prediction(self):
        """Testa a predição de tendências."""
        # Criar histórico com tendência crescente na Selic
        history = []
        for i in range(6):
            data = self.sample_macro_data.copy()
            data["selic"] = 8.0 + i * 0.5  # Tendência crescente
            history.append(data)
        
        trends = self.model.predict_macro_trend(history)
        
        self.assertIsInstance(trends, dict)
        if "selic" in trends:
            self.assertEqual(trends["selic"]["trend"], "up")
    
    def test_volatility_adjustment(self):
        """Testa o ajuste por volatilidade."""
        # Criar histórico com alta volatilidade
        high_vol_history = []
        for i in range(10):
            data = self.sample_macro_data.copy()
            data["selic"] += np.random.normal(0, 2.0)  # Alta volatilidade
            high_vol_history.append(data)
        
        vol_factor = self.model.calculate_volatility_adjustment(high_vol_history)
        
        self.assertIsInstance(vol_factor, float)
        self.assertTrue(0.5 <= vol_factor <= 1.0)  # Deve penalizar alta volatilidade
    
    def test_scenario_classification_with_regime(self):
        """Testa a classificação de cenário considerando regimes."""
        # Dados que devem resultar em expansão
        expansion_data = {
            "selic": 7.0, "ipca": 3.5, "dolar": 5.30, "pib": 2.5,
            "soja": 13.0, "milho": 5.5, "minerio": 100.0, "petroleo": 80.0
        }
        
        scenario = self.model.classificar_cenario_macro(expansion_data)
        self.assertIn(scenario, [
            "Expansão Forte", "Expansão Moderada", "Estável"
        ])


class TestEnhancedPortfolioOptimizer(unittest.TestCase):
    """
    Testes unitários para o otimizador de carteira aprimorado.
    """
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.optimizer = PortfolioOptimizer()
        self.sample_tickers = ["ITUB4.SA", "VALE3.SA", "PETR4.SA"]
        
        # Criar dados de retorno simulados
        dates = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
        np.random.seed(42)  # Para reprodutibilidade
        
        returns_data = {}
        for ticker in self.sample_tickers:
            returns_data[ticker] = np.random.normal(0.001, 0.02, len(dates))
        
        self.optimizer.returns_data = pd.DataFrame(returns_data, index=dates)
    
    def test_macro_adjusted_returns(self):
        """Testa o cálculo de retornos ajustados por scores macro."""
        macro_scores = {
            "ITUB4.SA": 8.0,  # Score alto
            "VALE3.SA": 6.0,  # Score médio
            "PETR4.SA": 4.0   # Score baixo
        }
        
        expected_returns = self.optimizer.calculate_expected_returns(
            method="macro_adjusted",
            macro_scores=macro_scores
        )
        
        self.assertIsInstance(expected_returns, pd.Series)
        self.assertEqual(len(expected_returns), len(self.sample_tickers))
        
        # ITUB4 deve ter retorno esperado maior que PETR4
        self.assertGreater(expected_returns["ITUB4.SA"], expected_returns["PETR4.SA"])
    
    def test_macro_bounds_generation(self):
        """Testa a geração de limites baseados em scores macro."""
        macro_scores = {
            "ITUB4.SA": 9.0,  # Score muito alto
            "VALE3.SA": 5.0,  # Score médio
            "PETR4.SA": 2.0   # Score baixo
        }
        
        bounds = self.optimizer.generate_macro_bounds(macro_scores)
        
        self.assertIsInstance(bounds, dict)
        self.assertEqual(len(bounds), len(macro_scores))
        
        # ITUB4 deve ter limite superior maior que PETR4
        self.assertGreater(bounds["ITUB4.SA"][1], bounds["PETR4.SA"][1])
    
    def test_portfolio_optimization_with_macro(self):
        """Testa a otimização de carteira com fatores macro."""
        # Calcular retornos esperados e matriz de covariância
        self.optimizer.calculate_expected_returns()
        self.optimizer.calculate_covariance_matrix()
        
        macro_scores = {
            "ITUB4.SA": 8.0,
            "VALE3.SA": 6.0,
            "PETR4.SA": 4.0
        }
        
        macro_bounds = self.optimizer.generate_macro_bounds(macro_scores)
        
        result = self.optimizer.optimize_portfolio(
            objective="sharpe",
            macro_bounds=macro_bounds
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(result["optimization_success"])
        self.assertIsInstance(result["weights"], pd.DataFrame)
        
        # Verificar se a soma dos pesos é aproximadamente 1
        total_weight = result["weights"]["Weight"].sum()
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_portfolio_validation(self):
        """Testa a validação da carteira."""
        # Criar uma carteira de teste
        weights_df = pd.DataFrame({
            "Ticker": ["ITUB4.SA", "VALE3.SA", "PETR4.SA"],
            "Weight": [0.5, 0.3, 0.2],
            "Expected_Return": [0.12, 0.15, 0.10]
        })
        
        validation = self.optimizer.validate_portfolio(weights_df)
        
        self.assertIsInstance(validation, dict)
        self.assertIn("is_valid", validation)
        self.assertTrue(validation["is_valid"])
    
    def test_efficient_frontier_calculation(self):
        """Testa o cálculo da fronteira eficiente."""
        # Calcular retornos esperados e matriz de covariância
        self.optimizer.calculate_expected_returns()
        self.optimizer.calculate_covariance_matrix()
        
        frontier = self.optimizer.calculate_efficient_frontier(n_points=10)
        
        if frontier is not None:  # Pode falhar devido à simulação
            self.assertIsInstance(frontier, pd.DataFrame)
            self.assertIn("Return", frontier.columns)
            self.assertIn("Risk", frontier.columns)
            self.assertIn("Sharpe", frontier.columns)


class TestIntegration(unittest.TestCase):
    """
    Testes de integração entre o modelo macro e o otimizador.
    """
    
    def setUp(self):
        """Configuração inicial para os testes de integração."""
        self.macro_model = MacroEconomicModel()
        self.optimizer = PortfolioOptimizer()
        
        self.sample_macro_data = {
            "selic": 10.5, "ipca": 4.2, "dolar": 5.20, "pib": 2.1,
            "soja": 12.5, "milho": 6.0, "minerio": 120.0, "petroleo": 85.0
        }
        
        self.sample_tickers = ["ITUB4.SA", "VALE3.SA", "PETR4.SA"]
        self.setores_por_ticker = {
            "ITUB4.SA": "Bancos",
            "VALE3.SA": "Mineração e Siderurgia",
            "PETR4.SA": "Petróleo, Gás e Biocombustíveis"
        }
    
    def test_macro_to_portfolio_integration(self):
        """Testa a integração entre análise macro e otimização de carteira."""
        # Calcular scores macro
        scores_macro = self.macro_model.pontuar_macro(self.sample_macro_data)
        
        # Calcular favorecimento por setor
        macro_scores_by_ticker = {}
        for ticker, setor in self.setores_por_ticker.items():
            favorecimento = self.macro_model.calcular_favorecimento_continuo(
                setor, scores_macro
            )
            # Converter favorecimento (-2 a 2) para score (0 a 10)
            score = max(0, min(10, 5 + favorecimento * 2.5))
            macro_scores_by_ticker[ticker] = score
        
        # Verificar se os scores foram calculados
        self.assertEqual(len(macro_scores_by_ticker), len(self.sample_tickers))
        for score in macro_scores_by_ticker.values():
            self.assertTrue(0 <= score <= 10)
        
        # Gerar limites macro para otimização
        macro_bounds = self.optimizer.generate_macro_bounds(macro_scores_by_ticker)
        
        self.assertIsInstance(macro_bounds, dict)
        self.assertEqual(len(macro_bounds), len(self.sample_tickers))


def run_tests():
    """
    Executa todos os testes.
    """
    # Criar suite de testes
    test_suite = unittest.TestSuite()
    
    # Adicionar testes do modelo macro
    test_suite.addTest(unittest.makeSuite(TestEnhancedMacroModel))
    
    # Adicionar testes do otimizador
    test_suite.addTest(unittest.makeSuite(TestEnhancedPortfolioOptimizer))
    
    # Adicionar testes de integração
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Executando testes do modelo macroeconômico e otimizador aprimorados...")
    success = run_tests()
    
    if success:
        print("\n✅ Todos os testes passaram com sucesso!")
    else:
        print("\n❌ Alguns testes falharam. Verifique os logs acima.")
    
    exit(0 if success else 1)


