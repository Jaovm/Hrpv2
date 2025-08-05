import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# Configuração da página
st.set_page_config(
    page_title="HBP Macro - Análise de Carteira",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Importações dos módulos locais
try:
    from src.data.bcb_data import fetch_macro_bcb_data
    from src.data.yfinance_data import obter_preco_petroleo_hist, calcular_media_movel
    from src.data.focus_data import obter_macro_focus
    from src.models.macro_model import MacroEconomicModel
    from src.models.portfolio_optimizer import PortfolioOptimizer
    from src.utils.asset_analyzer import AssetAnalyzer
    from config.config import PARAMS
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

# Cache das funções principais
@st.cache_data(ttl=3600)  # Cache por 1 hora
def obter_dados_macro():
    """
    Obtém dados macroeconômicos de todas as fontes.
    """
    try:
        # Dados do Boletim Focus
        macro_focus = obter_macro_focus()
        
        # Dados de commodities
        petroleo_atual = calcular_media_movel("BZ=F", periodo="1mo", intervalo="1d")
        soja_atual = calcular_media_movel("ZS=F", periodo="1mo", intervalo="1d")
        milho_atual = calcular_media_movel("ZC=F", periodo="1mo", intervalo="1d")
        minerio_atual = calcular_media_movel("TIO=F", periodo="1mo", intervalo="1d")
        
        macro_data = {
            "ipca": macro_focus.get("ipca", 4.0),
            "selic": macro_focus.get("selic", 10.0),
            "pib": macro_focus.get("pib", 2.0),
            "dolar": macro_focus.get("dolar", 5.0),
            "petroleo": petroleo_atual or 80.0,
            "soja": soja_atual or 13.0,
            "milho": milho_atual or 5.5,
            "minerio": minerio_atual or 100.0
        }
        
        return macro_data
    except Exception as e:
        st.error(f"Erro ao obter dados macroeconômicos: {e}")
        return None

@st.cache_data(ttl=1800)  # Cache por 30 minutos
def gerar_ranking_completo(carteira_selecionada, macro_data):
    """
    Gera ranking completo de ações.
    """
    try:
        analyzer = AssetAnalyzer()
        return analyzer.gerar_ranking_acoes(carteira_selecionada, macro_data)
    except Exception as e:
        st.error(f"Erro ao gerar ranking: {e}")
        return pd.DataFrame()

def main():
    """
    Função principal da aplicação Streamlit.
    """
    st.title("📈 HBP Macro - Análise de Carteira de Investimentos")
    st.markdown("### Sistema de Sugestão de Carteira baseado em Análise Macroeconômica")
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Seleção de carteira
        carteiras_disponiveis = {
            "Carteira Conservadora": {
                'ITUB4.SA': 'Itaú Unibanco', 'BBDC4.SA': 'Bradesco', 'BBAS3.SA': 'Banco do Brasil',
                'EGIE3.SA': 'Engie Brasil', 'CPLE6.SA': 'Copel', 'SBSP3.SA': 'Sabesp',
                'ABEV3.SA': 'Ambev', 'NTCO3.SA': 'Natura', 'HAPV3.SA': 'Hapvida'
            },
            "Carteira Moderada": {
                'VALE3.SA': 'Vale', 'PETR4.SA': 'Petrobras', 'ITUB4.SA': 'Itaú Unibanco',
                'WEGE3.SA': 'WEG', 'MGLU3.SA': 'Magazine Luiza', 'LREN3.SA': 'Lojas Renner',
                'AGRO3.SA': 'BrasilAgro', 'FLRY3.SA': 'Fleury', 'B3SA3.SA': 'B3'
            },
            "Carteira Agressiva": {
                'MGLU3.SA': 'Magazine Luiza', 'TOTS3.SA': 'Totvs', 'RENT3.SA': 'Localiza',
                'AZUL4.SA': 'Azul', 'COGN3.SA': 'Cogna', 'PETZ3.SA': 'Petz',
                'LWSA3.SA': 'Locaweb', 'MOVI3.SA': 'Movida', 'AMER3.SA': 'Americanas'
            }
        }
        
        carteira_escolhida = st.selectbox(
            "Escolha uma carteira:",
            list(carteiras_disponiveis.keys())
        )
        
        carteira_selecionada = carteiras_disponiveis[carteira_escolhida]
        
        # Configurações de otimização
        st.subheader("🎯 Otimização de Carteira")
        usar_macro_weights = st.checkbox("Usar pesos macroeconômicos", value=True)
        objetivo_otimizacao = st.selectbox(
            "Objetivo da otimização:",
            ["sharpe", "min_variance", "target_return"]
        )
        
        if objetivo_otimizacao == "target_return":
            target_return = st.slider("Retorno alvo anual:", 0.05, 0.30, 0.15, 0.01)
        else:
            target_return = None
        
        max_weight_per_asset = st.slider("Peso máximo por ativo:", 0.1, 0.5, 0.2, 0.05)
    
    # Obter dados macroeconômicos
    with st.spinner("Carregando dados macroeconômicos..."):
        macro_data = obter_dados_macro()
    
    if macro_data is None:
        st.error("Não foi possível obter dados macroeconômicos. Verifique a conexão.")
        st.stop()
    
    # Inicializar modelos
    macro_model = MacroEconomicModel()
    asset_analyzer = AssetAnalyzer()
    
    # Análise macroeconômica
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cenário Macroeconômico Atual")
        
        cenario_atual = macro_model.classificar_cenario_macro(macro_data)
        score_macro = macro_model.pontuar_macro(macro_data)
        
        # Exibir cenário com cor
        if "Expansão" in cenario_atual:
            st.success(f"**Cenário:** {cenario_atual}")
        elif "Estável" in cenario_atual:
            st.info(f"**Cenário:** {cenario_atual}")
        else:
            st.warning(f"**Cenário:** {cenario_atual}")
        
        # Métricas macroeconômicas
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("IPCA (%)", f"{macro_data['ipca']:.1f}", 
                     delta=f"Score: {score_macro['inflação']:.1f}")
            st.metric("Selic (%)", f"{macro_data['selic']:.1f}", 
                     delta=f"Score: {score_macro['juros']:.1f}")
        
        with metrics_col2:
            st.metric("Dólar (R$)", f"{macro_data['dolar']:.2f}", 
                     delta=f"Score: {score_macro['dolar']:.1f}")
            st.metric("PIB (%)", f"{macro_data['pib']:.1f}", 
                     delta=f"Score: {score_macro['pib']:.1f}")
    
    with col2:
        st.subheader("🏭 Setores Favorecidos")
        
        setores_favorecidos = macro_model.get_favored_sectors(cenario_atual)
        
        if setores_favorecidos:
            for i, setor in enumerate(setores_favorecidos[:5]):  # Top 5
                st.write(f"{i+1}. {setor}")
        else:
            st.write("Nenhum setor especificamente favorecido")
        
        # Gráfico de scores macro
        fig_scores = go.Figure(data=[
            go.Bar(
                x=list(score_macro.keys())[:-1],  # Excluir 'media_global'
                y=list(score_macro.values())[:-1],
                marker_color=['green' if v >= 7 else 'orange' if v >= 4 else 'red' 
                             for v in list(score_macro.values())[:-1]]
            )
        ])
        fig_scores.update_layout(
            title="Scores dos Indicadores Macro",
            xaxis_title="Indicadores",
            yaxis_title="Score (0-10)",
            height=300
        )
        st.plotly_chart(fig_scores, use_container_width=True)
    
    # Análise da carteira
    st.subheader("💼 Análise da Carteira Selecionada")
    
    with st.spinner("Analisando ativos da carteira..."):
        ranking_df = gerar_ranking_completo(carteira_selecionada, macro_data)
    
    if not ranking_df.empty:
        # Exibir ranking
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏆 Ranking de Ações")
            
            # Preparar dados para exibição
            display_df = ranking_df.copy()
            display_df['upside_potential'] = display_df['upside_potential'].apply(lambda x: f"{x:.1%}")
            display_df['preco_atual'] = display_df['preco_atual'].apply(lambda x: f"R$ {x:.2f}")
            display_df['preco_alvo'] = display_df['preco_alvo'].apply(lambda x: f"R$ {x:.2f}")
            display_df['score'] = display_df['score'].apply(lambda x: f"{x:.2f}")
            display_df['favorecimento_macro'] = display_df['favorecimento_macro'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(
                display_df[['ticker', 'setor', 'preco_atual', 'preco_alvo', 
                           'upside_potential', 'favorecimento_macro', 'score']],
                column_config={
                    "ticker": "Ticker",
                    "setor": "Setor",
                    "preco_atual": "Preço Atual",
                    "preco_alvo": "Preço Alvo",
                    "upside_potential": "Potencial (%)",
                    "favorecimento_macro": "Fav. Macro",
                    "score": "Score Final"
                },
                use_container_width=True
            )
        
        with col2:
            st.subheader("📈 Top 5 Ações")
            
            top_5 = ranking_df.head(5)
            for idx, row in top_5.iterrows():
                with st.container():
                    st.write(f"**{row['ticker']}** - {row['setor']}")
                    st.write(f"Score: {row['score']:.2f} | Upside: {row['upside_potential']:.1%}")
                    st.write("---")
    
    # Otimização de carteira
    st.subheader("🎯 Otimização de Carteira")
    
    if st.button("Otimizar Carteira", type="primary"):
        with st.spinner("Otimizando carteira..."):
            try:
                # Filtrar ativos válidos
                tickers_validos = asset_analyzer.filtrar_ativos_validos(carteira_selecionada, macro_data)
                
                if len(tickers_validos) < 3:
                    st.error("Poucos ativos válidos para otimização. Necessário pelo menos 3 ativos.")
                else:
                    # Inicializar otimizador
                    optimizer = PortfolioOptimizer()
                    
                    # Buscar dados de retorno
                    returns_data = optimizer.fetch_returns_data(tickers_validos)
                    
                    if returns_data is not None and not returns_data.empty:
                        # Calcular retornos esperados
                        if usar_macro_weights:
                            macro_scores = asset_analyzer.calcular_scores_macro_por_ticker(tickers_validos, macro_data)
                            expected_returns = optimizer.calculate_expected_returns("macro_adjusted", macro_scores)
                        else:
                            expected_returns = optimizer.calculate_expected_returns("mean")
                        
                        # Calcular matriz de covariância
                        cov_matrix = optimizer.calculate_covariance_matrix("ledoit_wolf")
                        
                        if expected_returns is not None and cov_matrix is not None:
                            # Gerar bounds baseados em scores macro
                            if usar_macro_weights:
                                macro_bounds = optimizer.generate_macro_bounds(macro_scores, max_weight_per_asset)
                            else:
                                macro_bounds = None
                            
                            # Otimizar
                            result = optimizer.optimize_portfolio(
                                objective=objetivo_otimizacao,
                                target_return=target_return,
                                max_weight=max_weight_per_asset,
                                macro_bounds=macro_bounds
                            )
                            
                            if result and result['optimization_success']:
                                # Exibir resultados
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("📊 Carteira Otimizada")
                                    
                                    weights_df = result['weights']
                                    
                                    # Métricas da carteira
                                    st.metric("Retorno Esperado", f"{result['portfolio_return']:.1%}")
                                    st.metric("Risco (Volatilidade)", f"{result['portfolio_std']:.1%}")
                                    st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
                                    
                                    # Validação da carteira
                                    validation = optimizer.validate_portfolio(weights_df)
                                    if validation['warnings']:
                                        for warning in validation['warnings']:
                                            st.warning(warning)
                                    if validation['errors']:
                                        for error in validation['errors']:
                                            st.error(error)
                                
                                with col2:
                                    st.subheader("🥧 Alocação da Carteira")
                                    
                                    # Gráfico de pizza
                                    fig_pie = px.pie(
                                        weights_df,
                                        values='Weight',
                                        names='Ticker',
                                        title="Distribuição dos Pesos"
                                    )
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                
                                # Tabela detalhada
                                st.subheader("📋 Detalhes da Alocação")
                                
                                detailed_df = weights_df.copy()
                                detailed_df['Weight'] = detailed_df['Weight'].apply(lambda x: f"{x:.1%}")
                                detailed_df['Expected_Return'] = detailed_df['Expected_Return'].apply(lambda x: f"{x:.1%}")
                                
                                st.dataframe(
                                    detailed_df,
                                    column_config={
                                        "Ticker": "Ticker",
                                        "Weight": "Peso (%)",
                                        "Expected_Return": "Retorno Esperado (%)"
                                    },
                                    use_container_width=True
                                )
                                
                                # Distribuição setorial
                                st.subheader("🏭 Distribuição Setorial da Carteira")
                                sector_dist = asset_analyzer.get_sector_distribution(weights_df['Ticker'].tolist())
                                
                                if not sector_dist.empty:
                                    fig_sectors = px.bar(
                                        x=sector_dist.index,
                                        y=sector_dist.values,
                                        title="Número de Ativos por Setor"
                                    )
                                    fig_sectors.update_layout(
                                        xaxis_title="Setores",
                                        yaxis_title="Número de Ativos"
                                    )
                                    st.plotly_chart(fig_sectors, use_container_width=True)
                            else:
                                st.error("Falha na otimização da carteira. Tente ajustar os parâmetros.")
                        else:
                            st.error("Erro ao calcular retornos esperados ou matriz de covariância.")
                    else:
                        st.error("Erro ao obter dados históricos dos ativos.")
            except Exception as e:
                st.error(f"Erro durante a otimização: {e}")
                logging.error(f"Erro na otimização: {e}")
    
    # Informações adicionais
    with st.expander("ℹ️ Informações sobre a Metodologia"):
        st.markdown("""
        ### Metodologia de Análise
        
        **Análise Macroeconômica:**
        - Coleta dados do BCB, Yahoo Finance e Boletim Focus
        - Classifica cenários macro em 5 categorias
        - Calcula scores para cada indicador (0-10)
        
        **Favorecimento Setorial:**
        - Mapeia sensibilidade de cada setor aos indicadores macro
        - Calcula favorecimento contínuo baseado no cenário atual
        - Ajusta scores de ativos conforme favorecimento setorial
        
        **Otimização de Carteira:**
        - Utiliza Teoria Moderna de Portfólio (Markowitz)
        - Estimador Ledoit-Wolf para matriz de covariância
        - Incorpora scores macroeconômicos nos retornos esperados
        - Aplica restrições de peso baseadas em favorecimento macro
        
        **Validação:**
        - Verifica diversificação mínima
        - Controla concentração máxima por ativo
        - Valida consistência dos pesos
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**HBP Macro** - Sistema de Análise de Carteira | Desenvolvido com Streamlit")

if __name__ == "__main__":
    main()

