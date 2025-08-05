import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import yfinance as yf
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PortfolioOptimizer:
    """
    Classe para otimização de carteira usando métodos modernos de teoria de portfólio.
    Implementa otimização de Markowitz com melhorias para uso institucional.
    """
    
    def __init__(self, risk_free_rate=0.05):
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.cov_matrix = None
        self.expected_returns = None
        
    def fetch_returns_data(self, tickers, period="2y"):
        """
        Busca dados de retorno para os tickers especificados.
        
        Args:
            tickers (list): Lista de tickers para buscar dados
            period (str): Período de dados históricos
            
        Returns:
            pd.DataFrame: DataFrame com retornos diários
        """
        try:
            logging.info(f"Buscando dados de retorno para {len(tickers)} ativos")
            data = yf.download(tickers, period=period, progress=False)['Adj Close']
            
            if data.empty:
                logging.error("Nenhum dado de preço foi obtido")
                return None
                
            # Calcular retornos diários
            returns = data.pct_change().dropna()
            
            # Remover colunas com muitos valores NaN (>50%)
            threshold = len(returns) * 0.5
            returns = returns.dropna(axis=1, thresh=threshold)
            
            # Preencher valores NaN restantes com 0
            returns = returns.fillna(0)
            
            logging.info(f"Dados de retorno obtidos para {len(returns.columns)} ativos")
            self.returns_data = returns
            return returns
            
        except Exception as e:
            logging.error(f"Erro ao buscar dados de retorno: {e}")
            return None
    
    def calculate_expected_returns(self, method="mean", macro_scores=None):
        """
        Calcula retornos esperados usando diferentes métodos.
        
        Args:
            method (str): Método de cálculo ('mean', 'capm', 'macro_adjusted')
            macro_scores (dict): Scores macro para ajuste de retornos
            
        Returns:
            pd.Series: Retornos esperados anualizados
        """
        if self.returns_data is None:
            logging.error("Dados de retorno não disponíveis")
            return None
            
        try:
            if method == "mean":
                # Média histórica anualizada
                expected_returns = self.returns_data.mean() * 252
                
            elif method == "macro_adjusted" and macro_scores:
                # Ajuste baseado em scores macroeconômicos
                base_returns = self.returns_data.mean() * 252
                
                # Normalizar scores para multiplicadores entre 0.5 e 1.5
                if macro_scores:
                    min_score = min(macro_scores.values())
                    max_score = max(macro_scores.values())
                    
                    adjustments = {}
                    for ticker in base_returns.index:
                        if ticker in macro_scores:
                            normalized_score = (macro_scores[ticker] - min_score) / (max_score - min_score + 1e-9)
                            adjustment = 0.5 + normalized_score
                            adjustments[ticker] = adjustment
                        else:
                            adjustments[ticker] = 1.0
                    
                    expected_returns = base_returns * pd.Series(adjustments)
                else:
                    expected_returns = base_returns
                    
            else:
                expected_returns = self.returns_data.mean() * 252
                
            self.expected_returns = expected_returns
            logging.info(f"Retornos esperados calculados usando método: {method}")
            return expected_returns
            
        except Exception as e:
            logging.error(f"Erro ao calcular retornos esperados: {e}")
            return None
    
    def calculate_covariance_matrix(self, method="ledoit_wolf"):
        """
        Calcula matriz de covariância usando diferentes estimadores.
        
        Args:
            method (str): Método de estimação ('sample', 'ledoit_wolf')
            
        Returns:
            pd.DataFrame: Matriz de covariância anualizada
        """
        if self.returns_data is None:
            logging.error("Dados de retorno não disponíveis")
            return None
            
        try:
            if method == "ledoit_wolf":
                # Estimador de Ledoit-Wolf (mais robusto para amostras pequenas)
                lw = LedoitWolf()
                cov_matrix = lw.fit(self.returns_data).covariance_
                cov_matrix = pd.DataFrame(cov_matrix, 
                                        index=self.returns_data.columns,
                                        columns=self.returns_data.columns)
            else:
                # Matriz de covariância amostral
                cov_matrix = self.returns_data.cov()
            
            # Anualizar
            cov_matrix = cov_matrix * 252
            
            self.cov_matrix = cov_matrix
            logging.info(f"Matriz de covariância calculada usando método: {method}")
            return cov_matrix
            
        except Exception as e:
            logging.error(f"Erro ao calcular matriz de covariância: {e}")
            return None
    
    def optimize_portfolio(self, objective="sharpe", target_return=None, 
                          max_weight=0.3, min_weight=0.0, macro_bounds=None):
        """
        Otimiza a carteira usando diferentes objetivos.
        
        Args:
            objective (str): Objetivo da otimização ('sharpe', 'min_variance', 'target_return')
            target_return (float): Retorno alvo (para objective='target_return')
            max_weight (float): Peso máximo por ativo
            min_weight (float): Peso mínimo por ativo
            macro_bounds (dict): Limites específicos baseados em scores macro
            
        Returns:
            dict: Resultado da otimização com pesos e métricas
        """
        if self.expected_returns is None or self.cov_matrix is None:
            logging.error("Retornos esperados ou matriz de covariância não disponíveis")
            return None
            
        try:
            n_assets = len(self.expected_returns)
            
            # Definir bounds
            if macro_bounds:
                bounds = []
                for ticker in self.expected_returns.index:
                    if ticker in macro_bounds:
                        bounds.append(macro_bounds[ticker])
                    else:
                        bounds.append((min_weight, max_weight))
                bounds = tuple(bounds)
            else:
                bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
            
            # Restrições
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Soma dos pesos = 1
            ]
            
            if objective == "target_return" and target_return:
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: np.dot(x, self.expected_returns) - target_return
                })
            
            # Função objetivo
            if objective == "sharpe":
                def objective_function(weights):
                    portfolio_return = np.dot(weights, self.expected_returns)
                    portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
                    portfolio_std = np.sqrt(portfolio_variance)
                    sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
                    return -sharpe_ratio  # Minimizar o negativo = maximizar
                    
            elif objective == "min_variance":
                def objective_function(weights):
                    return np.dot(weights, np.dot(self.cov_matrix, weights))
                    
            elif objective == "target_return":
                def objective_function(weights):
                    return np.dot(weights, np.dot(self.cov_matrix, weights))
            
            # Pesos iniciais (igualmente distribuídos)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Otimização
            result = minimize(
                objective_function,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if not result.success:
                logging.warning(f"Otimização não convergiu: {result.message}")
                return None
            
            # Calcular métricas da carteira otimizada
            optimal_weights = result.x
            portfolio_return = np.dot(optimal_weights, self.expected_returns)
            portfolio_variance = np.dot(optimal_weights, np.dot(self.cov_matrix, optimal_weights))
            portfolio_std = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
            
            # Criar DataFrame com pesos
            weights_df = pd.DataFrame({
                'Ticker': self.expected_returns.index,
                'Weight': optimal_weights,
                'Expected_Return': self.expected_returns.values
            }).sort_values('Weight', ascending=False)
            
            # Filtrar pesos muito pequenos
            weights_df = weights_df[weights_df['Weight'] > 0.001]
            
            optimization_result = {
                'weights': weights_df,
                'portfolio_return': portfolio_return,
                'portfolio_std': portfolio_std,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': result.success,
                'optimization_message': result.message
            }
            
            logging.info(f"Otimização concluída com sucesso. Sharpe Ratio: {sharpe_ratio:.4f}")
            return optimization_result
            
        except Exception as e:
            logging.error(f"Erro na otimização da carteira: {e}")
            return None
    
    def calculate_efficient_frontier(self, n_points=50):
        """
        Calcula a fronteira eficiente de Markowitz.
        
        Args:
            n_points (int): Número de pontos na fronteira
            
        Returns:
            pd.DataFrame: DataFrame com retornos, riscos e pesos da fronteira eficiente
        """
        if self.expected_returns is None or self.cov_matrix is None:
            logging.error("Dados necessários não disponíveis para calcular fronteira eficiente")
            return None
            
        try:
            min_ret = self.expected_returns.min()
            max_ret = self.expected_returns.max()
            target_returns = np.linspace(min_ret, max_ret, n_points)
            
            frontier_results = []
            
            for target_ret in target_returns:
                result = self.optimize_portfolio(
                    objective="target_return",
                    target_return=target_ret,
                    max_weight=1.0,  # Sem restrições para fronteira eficiente
                    min_weight=0.0
                )
                
                if result and result['optimization_success']:
                    frontier_results.append({
                        'Return': result['portfolio_return'],
                        'Risk': result['portfolio_std'],
                        'Sharpe': result['sharpe_ratio']
                    })
            
            if frontier_results:
                frontier_df = pd.DataFrame(frontier_results)
                logging.info(f"Fronteira eficiente calculada com {len(frontier_df)} pontos")
                return frontier_df
            else:
                logging.warning("Nenhum ponto válido encontrado para a fronteira eficiente")
                return None
                
        except Exception as e:
            logging.error(f"Erro ao calcular fronteira eficiente: {e}")
            return None
    
    def generate_macro_bounds(self, macro_scores, base_limit=0.20, bonus=0.10):
        """
        Gera limites de peso baseados em scores macroeconômicos.
        
        Args:
            macro_scores (dict): Scores macro por ticker
            base_limit (float): Limite base para todos os ativos
            bonus (float): Bônus máximo para ativos favorecidos
            
        Returns:
            dict: Limites (min, max) por ticker
        """
        if not macro_scores:
            return None
            
        max_score = max(macro_scores.values()) if macro_scores else 1
        bounds = {}
        
        for ticker, score in macro_scores.items():
            bonus_pct = bonus * (score / max_score) if max_score > 0 else 0
            upper_bound = min(1.0, base_limit + bonus_pct)
            bounds[ticker] = (0.0, upper_bound)
            
        return bounds
    
    def validate_portfolio(self, weights_df, min_diversification=5):
        """
        Valida se a carteira atende aos critérios mínimos.
        
        Args:
            weights_df (pd.DataFrame): DataFrame com pesos da carteira
            min_diversification (int): Número mínimo de ativos
            
        Returns:
            dict: Resultado da validação
        """
        try:
            validation_result = {
                'is_valid': True,
                'warnings': [],
                'errors': []
            }
            
            # Verificar diversificação mínima
            n_assets = len(weights_df[weights_df['Weight'] > 0.01])  # Pesos > 1%
            if n_assets < min_diversification:
                validation_result['warnings'].append(
                    f"Carteira pouco diversificada: apenas {n_assets} ativos com peso > 1%"
                )
            
            # Verificar concentração máxima
            max_weight = weights_df['Weight'].max()
            if max_weight > 0.4:
                validation_result['warnings'].append(
                    f"Alta concentração: ativo com {max_weight:.1%} da carteira"
                )
            
            # Verificar soma dos pesos
            total_weight = weights_df['Weight'].sum()
            if abs(total_weight - 1.0) > 0.01:
                validation_result['errors'].append(
                    f"Soma dos pesos incorreta: {total_weight:.3f}"
                )
                validation_result['is_valid'] = False
            
            return validation_result
            
        except Exception as e:
            logging.error(f"Erro na validação da carteira: {e}")
            return {'is_valid': False, 'errors': [str(e)], 'warnings': []}

