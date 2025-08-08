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
    Implementa otimização de Markowitz com melhorias para uso institucional, incluindo
    ajustes baseados em fatores macroeconômicos.
    """
    
    def __init__(self, risk_free_rate=0.05):
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.cov_matrix = None
        self.expected_returns = None
        
    def fetch_returns_data(self, tickers, period="2y", interval="1d"):
        """
        Busca dados de retorno para os tickers especificados.
        
        Args:
            tickers (list): Lista de tickers para buscar dados.
            period (str): Período de dados históricos (ex: "2y").
            interval (str): Intervalo dos dados (ex: "1d", "1wk", "1mo").
            
        Returns:
            pd.DataFrame: DataFrame com retornos diários.
        """
        try:
            logging.info(f"Buscando dados de retorno para {len(tickers)} ativos no período {period} com intervalo {interval}")
            # Ajuste para garantir que todos os tickers sejam tratados como lista
            if isinstance(tickers, str):
                tickers = [tickers]
            
            data = yf.download(tickers, period=period, interval=interval, progress=False)["Adj Close"]
            
            if data.empty:
                logging.error("Nenhum dado de preço foi obtido ou dados vazios.")
                return None
            
            # Se apenas um ticker, garantir que seja um DataFrame
            if len(tickers) == 1:
                data = pd.DataFrame(data)

            # Calcular retornos diários/periódicos
            returns = data.pct_change().dropna()
            
            # Remover colunas com muitos valores NaN (>50%)
            threshold = len(returns) * 0.5
            returns = returns.dropna(axis=1, thresh=threshold)
            
            # Preencher valores NaN restantes com 0 (ou outro método de imputação)
            returns = returns.fillna(0)
            
            logging.info(f"Dados de retorno obtidos para {len(returns.columns)} ativos. Total de {len(returns)} observações.")
            self.returns_data = returns
            return returns
            
        except Exception as e:
            logging.error(f"Erro ao buscar dados de retorno: {e}")
            return None
    
    def calculate_expected_returns(self, method="mean", macro_scores=None, annualization_factor=252):
        """
        Calcula retornos esperados usando diferentes métodos.
        
        Args:
            method (str): Método de cálculo (
                'mean': média histórica simples,
                'macro_adjusted': média histórica ajustada por scores macroeconômicos
            ).
            macro_scores (dict): Scores macro para ajuste de retornos (espera-se valores entre 0 e 10).
            annualization_factor (int): Fator de anualização (252 para dias úteis, 12 para meses, etc.).
            
        Returns:
            pd.Series: Retornos esperados anualizados.
        """
        if self.returns_data is None:
            logging.error("Dados de retorno não disponíveis para calcular retornos esperados.")
            return None
            
        try:
            if method == "mean":
                # Média histórica anualizada
                expected_returns = self.returns_data.mean() * annualization_factor
                
            elif method == "macro_adjusted" and macro_scores:
                base_returns = self.returns_data.mean() * annualization_factor
                
                # Normalizar scores macro para um multiplicador de ajuste
                # Assumindo scores entre 0 e 10, mapear para um range de 0.8 a 1.2
                # Onde 5 é neutro (multiplicador 1.0)
                adjustments = {}
                for ticker in base_returns.index:
                    score = macro_scores.get(ticker, 5.0) # 5.0 como score neutro se não encontrado
                    # Mapeamento linear: score 0 -> 0.8, score 10 -> 1.2
                    # Multiplicador = 0.8 + (score / 10) * (1.2 - 0.8)
                    multiplier = 0.8 + (score / 10.0) * 0.4
                    adjustments[ticker] = multiplier
                
                expected_returns = base_returns * pd.Series(adjustments)
                
            else:
                logging.warning(f"Método de cálculo de retorno esperado '{method}' não reconhecido ou dados macro ausentes. Usando média simples.")
                expected_returns = self.returns_data.mean() * annualization_factor
                
            self.expected_returns = expected_returns
            logging.info(f"Retornos esperados calculados usando método: {method}")
            return expected_returns
            
        except Exception as e:
            logging.error(f"Erro ao calcular retornos esperados: {e}")
            return None
    
    def calculate_covariance_matrix(self, method="ledoit_wolf", annualization_factor=252):
        """
        Calcula matriz de covariância usando diferentes estimadores.
        
        Args:
            method (str): Método de estimação (
                'sample': matriz de covariância amostral simples,
                'ledoit_wolf': estimador de Ledoit-Wolf (mais robusto para amostras pequenas e ruidosas)
            ).
            annualization_factor (int): Fator de anualização.
            
        Returns:
            pd.DataFrame: Matriz de covariância anualizada.
        """
        if self.returns_data is None:
            logging.error("Dados de retorno não disponíveis para calcular matriz de covariância.")
            return None
            
        try:
            if method == "ledoit_wolf":
                # Estimador de Ledoit-Wolf (mais robusto para amostras pequenas e ruidosas)
                lw = LedoitWolf()
                cov_matrix = lw.fit(self.returns_data).covariance_
                cov_matrix = pd.DataFrame(cov_matrix, 
                                        index=self.returns_data.columns,
                                        columns=self.returns_data.columns)
            elif method == "sample":
                # Matriz de covariância amostral
                cov_matrix = self.returns_data.cov()
            else:
                logging.warning(f"Método de estimação de covariância '{method}' não reconhecido. Usando Ledoit-Wolf.")
                lw = LedoitWolf()
                cov_matrix = lw.fit(self.returns_data).covariance_
                cov_matrix = pd.DataFrame(cov_matrix, 
                                        index=self.returns_data.columns,
                                        columns=self.returns_data.columns)
            
            # Anualizar a matriz de covariância
            cov_matrix = cov_matrix * annualization_factor
            
            self.cov_matrix = cov_matrix
            logging.info(f"Matriz de covariância calculada usando método: {method}")
            return cov_matrix
            
        except Exception as e:
            logging.error(f"Erro ao calcular matriz de covariância: {e}")
            return None
    
    def optimize_portfolio(self, objective="sharpe", target_return=None, 
                          max_weight=0.3, min_weight=0.0, macro_bounds=None):
        """
        Otimiza a carteira usando diferentes objetivos (Markowitz).
        
        Args:
            objective (str): Objetivo da otimização (
                'sharpe': maximizar Sharpe Ratio,
                'min_variance': minimizar variância,
                'target_return': minimizar variância para um retorno alvo específico
            ).
            target_return (float): Retorno alvo (para objective='target_return').
            max_weight (float): Peso máximo permitido para qualquer ativo individual na carteira.
            min_weight (float): Peso mínimo permitido para qualquer ativo individual na carteira.
            macro_bounds (dict): Dicionário de limites de peso específicos por ticker,
                                 gerados com base em scores macroeconômicos.
            
        Returns:
            dict: Resultado da otimização com pesos e métricas da carteira otimizada.
        """
        if self.expected_returns is None or self.cov_matrix is None:
            logging.error("Retornos esperados ou matriz de covariância não disponíveis para otimização.")
            return None
            
        try:
            n_assets = len(self.expected_returns)
            
            # Definir bounds (limites) para os pesos de cada ativo
            # Se macro_bounds for fornecido, ele tem precedência
            if macro_bounds:
                bounds = []
                for ticker in self.expected_returns.index:
                    if ticker in macro_bounds:
                        bounds.append(macro_bounds[ticker])
                    else:
                        # Se o ticker não está nos macro_bounds, usa os limites gerais
                        bounds.append((min_weight, max_weight))
                bounds = tuple(bounds)
            else:
                # Limites gerais para todos os ativos
                bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
            
            # Restrições da otimização
            constraints = [
                # Restrição: a soma de todos os pesos deve ser igual a 1 (100% da carteira)
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # Se o objetivo é um retorno alvo, adicionar essa restrição
            if objective == "target_return" and target_return is not None:
                constraints.append({
                    'type': 'eq', 
                    'fun': lambda x: np.dot(x, self.expected_returns) - target_return
                })
            
            # Função objetivo a ser minimizada pela otimização
            if objective == "sharpe":
                def objective_function(weights):
                    portfolio_return = np.dot(weights, self.expected_returns)
                    portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
                    portfolio_std = np.sqrt(portfolio_variance)
                    
                    # Evitar divisão por zero ou Sharpe negativo em caso de risco zero
                    if portfolio_std == 0:
                        return -np.inf # Retorno negativo infinito para evitar essa solução
                    
                    sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
                    return -sharpe_ratio  # Minimizar o negativo do Sharpe para maximizá-lo
                    
            elif objective == "min_variance":
                def objective_function(weights):
                    # Minimizar a variância da carteira
                    return np.dot(weights, np.dot(self.cov_matrix, weights))
                    
            elif objective == "target_return":
                def objective_function(weights):
                    # Minimizar a variância para um retorno alvo (já garantido por restrição)
                    return np.dot(weights, np.dot(self.cov_matrix, weights))
            
            else:
                logging.error(f"Objetivo de otimização '{objective}' não suportado.")
                return None

            # Pesos iniciais para o algoritmo de otimização (igualmente distribuídos)
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Realizar a otimização usando o método Sequential Least Squares Programming (SLSQP)
            result = minimize(
                objective_function,
                initial_weights,
                method=\'SLSQP\',
                bounds=bounds,
                constraints=constraints,
                options={\'maxiter\': 1000, \'ftol\': 1e-9} # Aumentar maxiter para maior chance de convergência
            )
            
            if not result.success:
                logging.warning(f"Otimização não convergiu: {result.message}. Tentando com pesos iniciais aleatórios...")
                # Tentar novamente com pesos iniciais aleatórios se não convergir
                initial_weights_rand = np.random.random(n_assets)
                initial_weights_rand /= np.sum(initial_weights_rand)
                result = minimize(
                    objective_function,
                    initial_weights_rand,
                    method=\'SLSQP\',
                    bounds=bounds,
                    constraints=constraints,
                    options={\'maxiter\': 1000, \'ftol\': 1e-9}
                )
                if not result.success:
                    logging.error(f"Otimização falhou mesmo com pesos aleatórios: {result.message}")
                    return None
            
            # Calcular métricas da carteira otimizada com os pesos resultantes
            optimal_weights = result.x
            portfolio_return = np.dot(optimal_weights, self.expected_returns)
            portfolio_variance = np.dot(optimal_weights, np.dot(self.cov_matrix, optimal_weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            # Recalcular Sharpe Ratio para garantir que não haja problemas com o sinal
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std != 0 else 0
            
            # Criar DataFrame com os pesos otimizados e retornos esperados
            weights_df = pd.DataFrame({
                \'Ticker\': self.expected_returns.index,
                \'Weight\': optimal_weights,
                \'Expected_Return\': self.expected_returns.values
            }).sort_values(\'Weight\', ascending=False)
            
            # Filtrar ativos com pesos muito pequenos (considerados insignificantes)
            weights_df = weights_df[weights_df[\'Weight\'] > 0.0001] # Ajustado para 0.01% para maior precisão
            
            optimization_result = {
                \'weights\': weights_df,
                \'portfolio_return\': portfolio_return,
                \'portfolio_std\': portfolio_std,
                \'sharpe_ratio\': sharpe_ratio,
                \'optimization_success\': result.success,
                \'optimization_message\': result.message
            }
            
            logging.info(f"Otimização concluída com sucesso. Sharpe Ratio: {sharpe_ratio:.4f}")
            return optimization_result
            
        except Exception as e:
            logging.error(f"Erro inesperado na otimização da carteira: {e}")
            return None
    
    def calculate_efficient_frontier(self, n_points=50, max_weight_per_asset=1.0):
        """
        Calcula a fronteira eficiente de Markowitz, gerando uma série de carteiras
        com diferentes níveis de risco e retorno.
        
        Args:
            n_points (int): Número de pontos a serem calculados na fronteira.
            max_weight_per_asset (float): Peso máximo permitido para um único ativo na fronteira.
            
        Returns:
            pd.DataFrame: DataFrame com retornos, riscos e Sharpe Ratios para cada ponto da fronteira.
        """
        if self.expected_returns is None or self.cov_matrix is None:
            logging.error("Dados necessários não disponíveis para calcular fronteira eficiente.")
            return None
            
        try:
            # Definir o range de retornos alvo para a fronteira
            min_ret = self.expected_returns.min() * 0.8 # Começar um pouco abaixo do mínimo
            max_ret = self.expected_returns.max() * 1.2 # Ir um pouco acima do máximo
            target_returns = np.linspace(min_ret, max_ret, n_points)
            
            frontier_results = []
            
            for target_ret in target_returns:
                # Otimizar para cada retorno alvo, minimizando a variância
                result = self.optimize_portfolio(
                    objective="target_return",
                    target_return=target_ret,
                    max_weight=max_weight_per_asset, 
                    min_weight=0.0 # Permitir pesos zero
                )
                
                if result and result["optimization_success"]:
                    frontier_results.append({
                        "Return": result["portfolio_return"],
                        "Risk": result["portfolio_std"],
                        "Sharpe": result["sharpe_ratio"]
                    })
            
            if frontier_results:
                frontier_df = pd.DataFrame(frontier_results)
                # Remover duplicatas e ordenar por risco
                frontier_df = frontier_df.drop_duplicates(subset=["Return", "Risk"]).sort_values(by="Risk")
                logging.info(f"Fronteira eficiente calculada com {len(frontier_df)} pontos válidos.")
                return frontier_df
            else:
                logging.warning("Nenhum ponto válido encontrado para a fronteira eficiente.")
                return None
                
        except Exception as e:
            logging.error(f"Erro ao calcular fronteira eficiente: {e}")
            return None
    
    def generate_macro_bounds(self, macro_scores, base_limit=0.20, bonus_factor=0.02, max_overall_weight=0.35):
        """
        Gera limites de peso para a otimização baseados em scores macroeconômicos.
        Ativos com scores macro mais altos podem ter um limite de peso máximo maior.
        
        Args:
            macro_scores (dict): Scores macro por ticker (espera-se valores entre 0 e 10).
            base_limit (float): Limite de peso base para todos os ativos (ex: 0.20 = 20%).
            bonus_factor (float): Fator de bônus aplicado ao score macro para aumentar o limite.
                                  (ex: 0.02 significa que um score de 10 adiciona 0.20 ao limite base).
            max_overall_weight (float): Limite máximo absoluto para qualquer peso, mesmo com bônus.
            
        Returns:
            dict: Dicionário de tuplas (min_weight, max_weight) por ticker.
        """
        if not macro_scores:
            logging.warning("Nenhum score macro fornecido para gerar limites de peso. Usando limites gerais.")
            return None
            
        bounds = {}
        for ticker, score in macro_scores.items():
            # Calcular o bônus com base no score macro (score 0 = 0 bônus, score 10 = max_bonus)
            calculated_bonus = (score / 10.0) * (bonus_factor * 10) # Multiplicar por 10 para usar o bonus_factor diretamente
            
            # O limite superior é o base_limit mais o bônus, limitado pelo max_overall_weight
            upper_bound = min(max_overall_weight, base_limit + calculated_bonus)
            
            # O limite inferior pode ser 0 ou um valor mínimo institucional
            bounds[ticker] = (0.0, upper_bound)
            
        logging.info(f"Limites de peso gerados com base em scores macro para {len(bounds)} ativos.")
        return bounds
    
    def validate_portfolio(self, weights_df, min_diversification=5, max_concentration_single_asset=0.40):
        """
        Valida se a carteira otimizada atende aos critérios mínimos de diversificação e concentração.
        
        Args:
            weights_df (pd.DataFrame): DataFrame com pesos da carteira otimizada.
            min_diversification (int): Número mínimo de ativos com peso significativo (ex: >1%).
            max_concentration_single_asset (float): Concentração máxima permitida para um único ativo (ex: 0.40 = 40%).
            
        Returns:
            dict: Resultado da validação, incluindo status, avisos e erros.
        """
        try:
            validation_result = {
                'is_valid': True,
                'warnings': [],
                'errors': []
            }
            
            # Verificar diversificação mínima (ativos com peso > 1%)
            significant_assets = weights_df[weights_df['Weight'] > 0.01]
            n_significant_assets = len(significant_assets)
            if n_significant_assets < min_diversification:
                validation_result['warnings'].append(
                    f"Carteira pouco diversificada: apenas {n_significant_assets} ativos com peso > 1%. Mínimo recomendado: {min_diversification}."
                )
            
            # Verificar concentração máxima em um único ativo
            if not weights_df.empty:
                max_weight = weights_df['Weight'].max()
                if max_weight > max_concentration_single_asset:
                    most_concentrated_asset = weights_df.loc[weights_df['Weight'].idxmax(), 'Ticker']
                    validation_result['warnings'].append(
                        f"Alta concentração: {most_concentrated_asset} com {max_weight:.1%} da carteira. Máximo recomendado: {max_concentration_single_asset:.1%}."
                    )
            
            # Verificar soma dos pesos (deve ser aproximadamente 1.0)
            total_weight = weights_df['Weight'].sum()
            if abs(total_weight - 1.0) > 0.001: # Tolerância de 0.1%
                validation_result['errors'].append(
                    f"Soma dos pesos incorreta: {total_weight:.3f}. Deveria ser 1.0."
                )
                validation_result['is_valid'] = False
            
            logging.info(f"Validação da carteira concluída. Válida: {validation_result['is_valid']}. Erros: {len(validation_result['errors'])}. Avisos: {len(validation_result['warnings'])}.")
            return validation_result
            
        except Exception as e:
            logging.error(f"Erro na validação da carteira: {e}")
            return {'is_valid': False, 'errors': [str(e)], 'warnings': []}



