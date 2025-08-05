# HBP Macro - Sistema de Análise de Carteira de Investimentos

## Visão Geral

O HBP Macro é um sistema avançado de análise e otimização de carteiras de investimentos que combina análise macroeconômica com teoria moderna de portfólio. O sistema foi desenvolvido para uso institucional, oferecendo robustez, confiabilidade e transparência nas decisões de investimento.

## Características Principais

### 🔍 Análise Macroeconômica Abrangente
- Coleta automática de dados do Banco Central do Brasil (BCB)
- Integração com Yahoo Finance para commodities
- Projeções do Boletim Focus do BCB
- Classificação automática de cenários macroeconômicos

### 📊 Favorecimento Setorial Inteligente
- Mapeamento de sensibilidade setorial a indicadores macro
- Cálculo dinâmico de favorecimento por setor
- Ajuste de scores baseado no cenário macroeconômico atual

### 🎯 Otimização de Carteira Avançada
- Implementação da Teoria Moderna de Portfólio (Markowitz)
- Estimador Ledoit-Wolf para matriz de covariância
- Múltiplos objetivos de otimização (Sharpe, mínima variância, retorno alvo)
- Incorporação de scores macroeconômicos nos retornos esperados

### 🛡️ Robustez e Confiabilidade
- Tratamento abrangente de erros
- Sistema de logging detalhado
- Retentativas automáticas com backoff exponencial
- Validação rigorosa de dados de entrada e saída

## Estrutura do Projeto

```
hbpmacro_app/
├── config/
│   └── config.py                 # Configurações e parâmetros
├── src/
│   ├── data/
│   │   ├── bcb_data.py          # Ingestão de dados do BCB
│   │   ├── yfinance_data.py     # Ingestão de dados do Yahoo Finance
│   │   └── focus_data.py        # Ingestão de dados do Boletim Focus
│   ├── models/
│   │   ├── macro_model.py       # Modelo macroeconômico
│   │   └── portfolio_optimizer.py # Otimizador de carteira
│   └── utils/
│       └── asset_analyzer.py    # Analisador de ativos
├── tests/
│   └── test_macro_model.py      # Testes unitários
├── streamlit_app.py             # Aplicação principal
├── requirements.txt             # Dependências
└── README.md                    # Documentação
```

## Instalação e Configuração

### Pré-requisitos
- Python 3.8 ou superior
- Conexão com internet para APIs externas

### Instalação

1. Clone ou baixe o projeto:
```bash
git clone <repository-url>
cd hbpmacro_app
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute a aplicação:
```bash
streamlit run streamlit_app.py
```

## Uso da Aplicação

### Interface Principal

A aplicação oferece uma interface intuitiva dividida em seções:

1. **Configurações (Sidebar)**
   - Seleção de carteira (Conservadora, Moderada, Agressiva)
   - Configurações de otimização
   - Parâmetros de peso máximo por ativo

2. **Cenário Macroeconômico**
   - Classificação automática do cenário atual
   - Scores dos indicadores macroeconômicos
   - Setores favorecidos no cenário atual

3. **Análise da Carteira**
   - Ranking de ações baseado em scores
   - Potencial de valorização (upside)
   - Favorecimento macroeconômico por ativo

4. **Otimização de Carteira**
   - Otimização automática baseada em objetivos
   - Visualização da alocação otimizada
   - Métricas de risco e retorno

### Carteiras Disponíveis

#### Carteira Conservadora
Focada em ativos de baixo risco com dividendos consistentes:
- Bancos grandes (ITUB4, BBDC4, BBAS3)
- Utilities (EGIE3, CPLE6, SBSP3)
- Consumo básico (ABEV3, NTCO3)
- Saúde (HAPV3)

#### Carteira Moderada
Balanceamento entre crescimento e estabilidade:
- Commodities (VALE3, PETR4)
- Indústria (WEGE3)
- Varejo (MGLU3, LREN3)
- Agronegócio (AGRO3)
- Serviços financeiros (B3SA3)

#### Carteira Agressiva
Focada em crescimento e inovação:
- Tecnologia (TOTS3, COGN3, LWSA3)
- Varejo digital (MGLU3)
- Mobilidade (RENT3, MOVI3)
- Aviação (AZUL4)
- Pets (PETZ3)

## Metodologia

### Análise Macroeconômica

O sistema coleta e analisa os seguintes indicadores:

1. **IPCA** - Índice de inflação
2. **Selic** - Taxa básica de juros
3. **PIB** - Crescimento econômico
4. **Dólar** - Taxa de câmbio
5. **Commodities** - Petróleo, soja, milho, minério de ferro

Cada indicador recebe uma pontuação de 0 a 10 baseada em parâmetros ótimos predefinidos.

### Classificação de Cenários

O sistema classifica o ambiente macroeconômico em 5 cenários:

1. **Expansão Forte** - Crescimento robusto, inflação controlada
2. **Expansão Moderada** - Crescimento moderado, alguns desafios
3. **Estável** - Economia equilibrada, sem grandes movimentos
4. **Contração Moderada** - Desaceleração, pressões inflacionárias
5. **Contração Forte** - Recessão, alta volatilidade

### Favorecimento Setorial

Cada setor possui sensibilidades específicas aos indicadores macro:

- **Setores Pró-cíclicos**: Beneficiam-se de crescimento (Consumo Discricionário, Tecnologia)
- **Setores Defensivos**: Resilientes em qualquer ciclo (Saúde, Utilities)
- **Setores de Commodities**: Sensíveis a preços de matérias-primas
- **Setores Financeiros**: Beneficiam-se de juros altos

### Otimização de Carteira

A otimização utiliza:

1. **Teoria de Markowitz** - Maximização do Sharpe Ratio
2. **Estimador Ledoit-Wolf** - Matriz de covariância robusta
3. **Restrições Macro** - Limites baseados em favorecimento setorial
4. **Validação** - Verificação de diversificação e concentração

## Configuração Avançada

### Parâmetros Macroeconômicos

Os parâmetros podem ser ajustados em `config/config.py`:

```python
PARAMS = {
    "selic_neutra": 7.0,      # Taxa Selic neutra
    "ipca_meta": 3.0,         # Meta de inflação
    "ipca_tolerancia": 1.5,   # Tolerância da meta
    "dolar_ideal": 5.30,      # Dólar ideal
    "pib_ideal": 2.0          # PIB ideal
}
```

### Sensibilidade Setorial

A sensibilidade pode ser customizada no arquivo `src/models/macro_model.py` na função `_load_sensibilidade_setorial()`.

### Mapeamento de Setores

Novos tickers podem ser adicionados no arquivo `src/utils/asset_analyzer.py` na função `_load_setores_por_ticker()`.

## Testes

Execute os testes unitários:

```bash
python -m pytest tests/ -v
```

## Deploy no Streamlit Cloud

1. Faça upload do projeto para um repositório GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositório GitHub
4. Configure o arquivo principal como `streamlit_app.py`
5. Deploy automático será realizado

## Limitações e Considerações

### Dados Externos
- Dependência de APIs externas (BCB, Yahoo Finance)
- Possíveis falhas temporárias de conectividade
- Limites de taxa das APIs

### Modelo Financeiro
- Baseado em dados históricos
- Não garante performance futura
- Requer validação contínua por especialistas

### Uso Recomendado
- Ferramenta de suporte à decisão
- Não substitui análise fundamentalista
- Sempre considerar fatores qualitativos

## Suporte e Manutenção

### Logs
Os logs são gerados automaticamente e incluem:
- Timestamps de operações
- Erros e avisos
- Informações de depuração

### Monitoramento
Recomenda-se monitorar:
- Sucesso das chamadas de API
- Tempo de resposta da aplicação
- Qualidade dos dados obtidos

### Atualizações
- Revisar parâmetros macroeconômicos trimestralmente
- Atualizar mapeamento setorial conforme necessário
- Validar modelo com dados históricos periodicamente

## Contribuição

Para contribuir com o projeto:

1. Faça fork do repositório
2. Crie uma branch para sua feature
3. Implemente testes para novas funcionalidades
4. Submeta um pull request

## Licença

Este projeto é desenvolvido para uso institucional. Consulte os termos de uso específicos da sua organização.

---

**Desenvolvido com ❤️ para análise financeira institucional**

