# Arquivo de Configuração para a Aplicação HBPMacro

# URLs de APIs
URL_BCB_API = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"
URL_OLINDA_API = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"

# Códigos de Indicadores do BCB
CODIGO_SELIC_BCB = 432
CODIGO_IPCA_BCB = 433
CODIGO_DOLAR_BCB = 1

# Tickers de Commodities
TICKER_PETROLEO = "BZ=F"
TICKER_SOJA = "ZS=F"
TICKER_MILHO = "ZC=F"
TICKER_MINERIO_FERRO = "TIO=F" # Exemplo, verificar o ticker correto

# Parâmetros de Pontuação Macroeconômica
PARAMS = {
    "selic_neutra": 7.0,
    "ipca_meta": 3.0,
    "ipca_tolerancia": 1.5,
    "dolar_ideal": 5.30,
    "pib_ideal": 2.0
}


