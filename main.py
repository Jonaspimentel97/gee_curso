import ee
import streamlit as st
import geemap.foliumap as geemap
from google.oauth2 import service_account
from ee import oauth

# Função para autenticar no Google Earth Engine
def get_auth():
    try:
        # Acessa as credenciais a partir dos segredos configurados no Streamlit
        service_account_keys = st.secrets["service_account_json"]
        credentials = service_account.Credentials.from_service_account_info(service_account_keys, scopes=oauth.SCOPES)
        
        # Inicializa o Google Earth Engine com as credenciais
        ee.Initialize(credentials)
        

# Inicialize o GEE antes de qualquer outra operação
auth_status = get_auth()
st.write(auth_status)

# Configuração do Streamlit
st.title('Classificação de Uso e Cobertura do Solo')

# Função para mascarar nuvens usando a banda QA_PIXEL do Landsat 8
def mask_l8(image):
    qa = image.select('QA_PIXEL')
    # Verifica se o bit de nuvem está desligado
    mask = qa.bitwiseAnd(1 << 3).eq(0)
    return image.updateMask(mask)

# Definir a área de interesse
area = ee.Geometry.Polygon([
    [[-43.003569900550836, -22.631163343162356],
     [-43.003569900550836, -22.71289591702512],
     [-42.814055740394586, -22.71289591702512],
     [-42.814055740394586, -22.631163343162356]]
])

# Criar a composição com mediana de um ano de dados Landsat 8 TOA
composite = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')\
    .filterDate('2023-01-01', '2023-12-25')\
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
Map = geemap.Map(center=[-22.67, -42.91], zoom=12)
Map.addLayer(classificacao, {'min': 0, 'max': 3, 'palette': ['red', 'green', 'blue', 'yellow']}, "Classificação")
Map.to_streamlit(height=300)

# Layout com colunas
col1, col2 = st.columns([1, 4])  # Ajuste a proporção das colunas para 1:4

# Legenda no lado esquerdo
with col1:
    legend_html = """
    <div style="background-color: white; padding: 1px; border: 2px solid grey; border-radius: 5px; width: 250px;">
        <b>Classes de Uso e Cobertura do Solo</b><br>
        <i style="background:red; width:15px; height:20px; display:inline-block; margin-right:1px;"></i>Área Construída<br>
        <i style="background:green; width:15px; height:20px; display:inline-block; margin-right:1px;"></i>Vegetação<br>
        <i style="background:blue; width:15px; height:20px; display:inline-block; margin-right:1px;"></i>Corpos d’água<br>
        <i style="background:yellow; width:15px; height:20px; display:inline-block; margin-right:1px;"></i>Áreas Agropastoris<br>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
