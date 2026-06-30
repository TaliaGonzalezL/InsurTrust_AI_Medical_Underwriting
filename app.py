import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Configuración de la página y Estética "Lux"
st.set_page_config(page_title="LuxLogistics DaaS", page_icon="💡", layout="wide")

st.title("💡 LuxLogistics DaaS")
st.subheader("Sistema de Inteligencia Logística y Predicción de Riesgo")
st.markdown("---")

# 2. Cargar el "Cerebro" de Predicción (Machine Learning)
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_luxlogistics.joblib')

model = cargar_modelo()

# 3. Cargar los Datos para el Agente de IA (Pandas)
@st.cache_data
def cargar_datos_csv():
    ruta_carpeta = os.path.dirname(__file__)
    ruta_al_archivo = os.path.join(ruta_carpeta, 'data_logistica_limpia.csv')
    # Usamos latin-1 por si el archivo viene de Excel/Windows con acentos
    return pd.read_csv(ruta_al_archivo, encoding='latin-1') 

df_ia = cargar_datos_csv()

# --- DICCIONARIO DE JERARQUÍA GEOGRÁFICA ---
regiones_por_mercado = {
    'LATAM': ['Central America', 'South America', 'Caribbean'],
    'USCA': ['USCA'],
    'Europe': ['Western Europe', 'Southern Europe', 'Northern Europe', 'Eastern Europe'],
    'Africa': ['North Africa', 'East Africa', 'West Africa', 'Central Africa', 'Southern Africa'],
    'Pacific Asia': ['Southeast Asia', 'South Asia', 'Oceania', 'Eastern Asia', 'Western Asia', 'Central Asia']
}

# --- SIDEBAR (Global para la predicción) ---
st.sidebar.header("Configuración del Envío")
market = st.sidebar.selectbox("Mercado de Destino", list(regiones_por_mercado.keys()))
lista_regiones_filtrada = regiones_por_mercado[market]
order_region = st.sidebar.selectbox("Región del Orden", lista_regiones_filtrada)
type_envio = st.sidebar.selectbox("Tipo de Pago", ['CASH', 'PAYMENT', 'DEBIT', 'TRANSFER'])
shipping_mode = st.sidebar.selectbox("Modo de Envío", ['Standard Class', 'First Class', 'Second Class', 'Same Day'])
customer_segment = st.sidebar.selectbox("Segmento de Cliente", ['Consumer', 'Corporate', 'Home Office'])
sales = st.sidebar.number_input("Valor de la Venta (USD)", min_value=1.0, value=150.0)
order_month = st.sidebar.slider("Mes del Pedido", 1, 12, 6)
order_day = st.sidebar.slider("Día de la Semana (0=Lun, 6=Dom)", 0, 6, 2)

# --- INTERFAZ: PESTAÑAS (TABS) ---
tab_prediccion, tab_chat = st.tabs(["🔮 Simulador de Riesgo", "🤖 LuxLogistics Smart Chat"])

with tab_prediccion:
    st.header("Simulador de Riesgo de Envío")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Utilice el panel de la izquierda para configurar los parámetros del envío.")
        
    if st.button("Evaluar Riesgo de Retraso"):
        input_data = pd.DataFrame([[
            type_envio, market, shipping_mode, customer_segment, 
            order_region, sales, order_month, order_day
        ]], columns=['Type', 'Market', 'Shipping Mode', 'Customer Segment', 'Order Region', 'Sales', 'order_month', 'order_day_of_week'])
        
        try:
            prediccion = model.predict(input_data)[0]
            probabilidad = model.predict_proba(input_data)[0][1]

            st.markdown("### Resultado del Análisis Predictivo")
            if prediccion == 1:
                st.error(f"⚠️ ALTO RIESGO DE RETRASO (Probabilidad: {probabilidad:.2%})")
                st.info("💡 Sugerencia: Considere cambiar el modo de envío o revisar la prioridad de despacho.")
            else:
                st.success(f"✅ ENVÍO SEGURO (Probabilidad de retraso: {probabilidad:.2%})")
        except Exception as e:
            st.error(f"Error en el motor: {e}")

with tab_chat:
    st.header("🤖 Chat con tus Datos (Powered by Gemini)")
    st.markdown("Esta sección utiliza Agentes de IA para analizar tus bases de datos en tiempo real.")
    
    if df_ia is not None:
        user_api_key = st.text_input("Introduce tu Google API Key (Gemini):", type="password")
        
        if user_api_key:
            pregunta = st.text_input("Hazle una pregunta a la base de datos de LuxLogistics:")
            
            if pregunta:
                try: # --- INICIO BLOQUE IA LIMPIO ---
                    # 1. Configuramos el modelo sin parámetros obsoletos
                    llm = ChatGoogleGenerativeAI(
                         model="gemini-flash-latest", 
                         google_api_key=user_api_key,
                         temperature=0
                         # Quitamos 'transport' porque ya vimos que causa advertencia
                    )   
                    # 2. Instrucciones ultra-directas
                    prefix = "You are a Python expert. Work with the dataframe 'df'. Always provide a 'Final Answer:' in Spanish."

                    # 3. Creamos el agente (Sin el parámetro que dio error en la terminal)
                    agent = create_pandas_dataframe_agent(
                        llm, 
                        df_ia, 
                        verbose=True, 
                        allow_dangerous_code=True,
                        max_iterations=5,
                        agent_type="zero-shot-react-description",
                        prefix=prefix
                    )

                    with st.spinner("LuxLogistics AI está analizando tu consulta..."):
                        try:
                            # INTENTO DE EJECUCIÓN
                            resultado = agent.invoke(pregunta)
                            st.success("✅ Análisis Completado")
                            st.markdown(f"### {resultado['output']}")
                        except Exception as parse_err:
                            # ESCUDO DE RECUPERACIÓN MEJORADO
                            error_str = str(parse_err)
                            # Si la respuesta está ahí pero LangChain no la entiende:
                            if "Final Answer:" in error_str:
                                respuesta_final = error_str.split("Final Answer:")[-1].split("`")[0].strip()
                                st.success("✅ Análisis Completado")
                                st.markdown(f"### {respuesta_final}")
                            # Si Gemini respondió pero sin el formato Final Answer
                            elif "Could not parse LLM output: `" in error_str:
                                respuesta_raw = error_str.split("Could not parse LLM output: `")[-1].rstrip("` ")
                                st.success("✅ Análisis Completado")
                                st.markdown(f"### {respuesta_raw}")
                            else:
                                # Si de plano no hay nada, mostramos el error técnico para saber qué pasó
                                st.error(f"La IA se confundió. Intenta preguntar de otra forma. (Detalle: {error_str[:100]}...)")
                except Exception as e: # --- ESTE EXCEPT AHORA SÍ ESTÁ BIEN ALINEADO ---
                    st.error(f"Error de conexión: {e}")
        else:
            st.info("💡 Introduce tu API Key de Google AI Studio para comenzar.")
    else:
        st.error("❌ El chat no está disponible porque falta el archivo 'data_logistica_limpia.csv'.")

st.markdown("---")
st.caption("LuxLogistics DaaS - Inteligencia Artificial para la cadena de suministro | Toluca, México.")