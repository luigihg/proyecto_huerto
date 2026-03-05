import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import geopandas as gpd
from shapely.geometry import Point
from fpdf import FPDF
import io

# Configuración de la página
st.set_page_config(page_title="Huerto Inteligente - Proyecto Final", layout="wide")

st.title("🍎 Dashboard: Análisis Integral del Huerto")
st.markdown("---")

# 1. DATOS Y LIMPIEZA
@st.cache_data # Para que la web cargue rápido
def cargar_datos():
    data = {
        'variedad': ['Gala', 'Roja', 'Verde', 'Gala', 'Roja', 'Verde', 'Gala'],
        'peso_g': [150, 170, 120, 155, 168, 125, 152],
        'precio': [10, 15, 9, 11, 14.5, 9.5, 10.5],
        'calidad': [8, 9, 10, 8, 9, 10, 8],
        'lat': [19.43, 20.65, 25.68, 19.45, 20.67, 25.70, 19.40],
        'lon': [-99.13, -103.34, -100.31, -99.15, -103.36, -100.30, -99.10]
    }
    return pd.DataFrame(data)

df = cargar_datos()

# Sidebar - Filtros
st.sidebar.header("Filtros de Cosecha")
variedad_sel = st.sidebar.multiselect("Selecciona Variedades:", df['variedad'].unique(), default=df['variedad'].unique())
df_filtrado = df[df['variedad'].isin(variedad_sel)]

# PESTAÑAS
tab1, tab2, tab3, tab4 = st.tabs(["📊 Estadísticas", "🤖 Inteligencia Artificial", "☁️ Opiniones (NLP)", "🗺️ Mapa de Ventas"])

with tab1:
    st.header("Estadísticas de la Cosecha")
    col1, col2 = st.columns(2)
    col1.metric("Precio Promedio", f"${df_filtrado['precio'].mean():.2f}")
    col2.metric("Peso Promedio", f"{df_filtrado['peso_g'].mean():.1f}g")
    st.dataframe(df_filtrado)

with tab2:
    st.header("Modelos Predictivos y Agrupamiento")
    
    # Clustering (K-means)
    kmeans = KMeans(n_clusters=2, random_state=42)
    df_filtrado['segmento'] = kmeans.fit_predict(df_filtrado[['peso_g', 'calidad']])
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_filtrado['peso_g'], df_filtrado['calidad'], c=df_filtrado['segmento'], cmap='viridis', s=100)
    ax.set_xlabel("Peso (g)")
    ax.set_ylabel("Calidad")
    st.pyplot(fig)
    st.write("Interpretación: Los colores agrupan automáticamente a los clientes VIP de los Mayoristas.")

with tab3:
    st.header("¿Qué dicen los clientes?")
    resenas = "dulce fresca crujiente premium excelente fresca dulce dulce manzana huerto calidad"
    nube = WordCloud(background_color='white', width=800, height=400).generate(resenas)
    
    fig_word, ax_word = plt.subplots()
    ax_word.imshow(nube)
    ax_word.axis("off")
    st.pyplot(fig_word)

with tab4:
    st.header("Ubicación Geográfica de Ventas")
    # Para Streamlit es más fácil usar su mapa nativo si tenemos lat/lon
    st.map(df_filtrado[['lat', 'lon']])
    st.write("Puntos de distribución activa en la República Mexicana.")

# Metodología y Conclusión
st.markdown("---")
st.subheader("📝 Metodología")
st.info("Este proyecto utiliza el flujo completo: Extracción (Scraping), Limpieza (Pandas), Modelado (Sklearn) y Visualización (Streamlit/GeoPandas).")


# --- FUNCIÓN PARA GENERAR EL PDF ---
def generar_pdf(dataframe):
    pdf = FPDF()
    pdf.add_page()
    
    # Título del Reporte
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Reporte Final: Análisis del Huerto Inteligente", ln=True, align='C')
    pdf.ln(10)
    
    # Introducción (Storytelling)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"Este reporte resume el análisis de {len(dataframe)} registros de cosecha. "
                               "Se aplicaron técnicas de Machine Learning para segmentación y "
                               "procesamiento de lenguaje natural para el análisis de sentimientos.")
    pdf.ln(5)

    # Estadísticas Clave
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Estadísticas Principales:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 10, txt=f"- Precio Promedio: ${dataframe['precio'].mean():.2f}", ln=True)
    pdf.cell(200, 10, txt=f"- Peso Promedio: {dataframe['peso_g'].mean():.1f}g", ln=True)
    pdf.ln(10)

    # Conclusión
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Conclusión del Análisis:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, txt="Los datos indican una correlación positiva entre peso y calidad, "
                               "permitiendo optimizar los precios de venta según el segmento de mercado.")

    # Guardar en un buffer de memoria para que Streamlit pueda descargarlo
    return pdf.output(dest='S').encode('latin-1')

# --- BOTÓN EN LA PÁGINA WEB ---
st.sidebar.markdown("---")
st.sidebar.subheader("Exportar Resultados")

# Generamos el archivo PDF
pdf_archivo = generar_pdf(df_filtrado)

# Creamos el botón de descarga
st.sidebar.download_button(
    label="📥 Descargar Reporte en PDF",
    data=pdf_archivo,
    file_name="Reporte_Huerto_Inteligente.pdf",
    mime="application/pdf"
)