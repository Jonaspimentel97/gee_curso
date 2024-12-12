import ee
import streamlit as st
import geemap.foliumap as geemap
from google.oauth2 import service_account
from ee import oauth

# Função para autenticar no Google Earth Engine
def get_auth():
    try:
        # Acessa as credenciais a partir dos segredos configurados no Streamlit
        service_account_keys = st.secrets["[google]service_account_json"]
        credentials = service_account.Credentials.from_service_account_info(service_account_keys, scopes=oauth.SCOPES)
        
        # Inicializa o Google Earth Engine com as credenciais
        ee.Initialize(credentials)
        
        # Retorna sucesso após autenticação
        return 'GEE initialized successfully'
    except Exception as e:
        # Caso haja erro na autenticação, exibe a mensagem de erro
        return f"Error initializing GEE: {e}"

# Inicialize o GEE antes de qualquer outra operação
auth_status = get_auth()
st.write(auth_status)

if "successfully" in auth_status:  # Continua com o processamento somente se o GEE for inicializado com sucesso
    # Configuração do Streamlit
    st.title('Classificação de Uso e Cobertura do Solo com Landsat 8')
    st.sidebar.header('Parâmetros de Entrada')

    # Função para mascarar nuvens usando a banda QA_PIXEL do Landsat 8
    def mask_l8(image):
        qa = image.select('QA_PIXEL')
        # Verifica se o bit de nuvem está desligado
        mask = qa.bitwiseAnd(1 << 3).eq(0)
        return image.updateMask(mask)

    # Definir a área de interesse
    area = ee.Geometry.Polygon([[
        [-43.003569900550836, -22.631163343162356],
        [-43.003569900550836, -22.71289591702512],
        [-42.814055740394586, -22.71289591702512],
        [-42.814055740394586, -22.631163343162356]
    ]])

    # Inputs do usuário
    start_date = st.sidebar.text_input('Data inicial (YYYY-MM-DD):', '2023-01-01')
    end_date = st.sidebar.text_input('Data final (YYYY-MM-DD):', '2023-12-25')

    # Criar a composição com mediana de um ano de dados Landsat 8 TOA
    composite = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')\
        .filterDate(start_date, end_date)\
        .map(mask_l8)\
        .median()\
        .clip(area)

    # Calcular índices espectrais
    ndvi = composite.normalizedDifference(['B5', 'B4']).rename('ndvi')
    ndbi = composite.expression(
        '(SWIR - NIR) / (SWIR + NIR)', {
            'SWIR': composite.select('B6'),
            'NIR': composite.select('B5')
        }).rename('ndbi')
    ndwi = composite.normalizedDifference(['B3', 'B5']).rename('ndwi')

    # Adicionar bandas NDVI, NDBI e NDWI à composição
    composite2 = composite.addBands(ndvi).addBands(ndwi).addBands(ndbi)

    # Seleção de bandas
    composite_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'ndvi', 'ndwi', 'ndbi']

    # Classes de treinamento
    urb = ee.FeatureCollection("projects/ee-jonasrpx2/assets/urb")
    veg = ee.FeatureCollection("projects/ee-jonasrpx2/assets/veg")
    agua = ee.FeatureCollection("projects/ee-jonasrpx2/assets/agua")
    agr = ee.FeatureCollection("projects/ee-jonasrpx2/assets/agr")

    classes = urb.merge(veg).merge(agua).merge(agr)

    # Amostragem de regiões
    uso_classes = composite2.select(composite_bands).sampleRegions(
        collection=classes,
        properties=['uso'],
        scale=30,
        projection='EPSG:4326',
        tileScale=2,
        geometries=False
    )

    # Treinamento do classificador Random Forest
    classificador = ee.Classifier.smileRandomForest(
        numberOfTrees=50,
        variablesPerSplit=2,
        minLeafPopulation=2,
        bagFraction=0.8,
        maxNodes=None,
        seed=12345
    ).train(
        features=uso_classes,
        classProperty='uso',
        inputProperties=composite_bands
    )

    # Classificação
    classificacao = composite2.select(composite_bands).classify(classificador)

    # Configurar o mapa com geemap
    Map = geemap.Map(center=[-22.67, -42.91], zoom=10)
    Map.addLayer(classificacao, {'min': 0, 'max': 3, 'palette': ['red', 'green', 'blue', 'yellow']}, "Classificação")
    Map.addLayer(area, {}, "Área de Interesse")
    Map.to_streamlit(height=600)
else:
    st.error("Erro na inicialização do GEE. Verifique as credenciais.")
