import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai
import re # <---Se agrega esto para búsqueda avanzada de texto
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Configuración de la página y Estética "InsurTrust"
st.set_page_config(page_title="InsurTrust AI", page_icon="🛡️", layout="wide")

st.title("🛡️ InsurTrust AI: Inteligencia en Suscripción Médica")
st.subheader("Sistema Agéntico de Evaluación de Riesgo para Vida y Salud")
st.markdown("---")

# 2. Cargar el "Cerebro" de Predicción (Machine Learning)
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_insurtrust.joblib')

model = cargar_modelo()

# 3. Cargar los Datos para el Agente de IA (Pandas)
@st.cache_data
def cargar_datos_csv():
    ruta_carpeta = os.path.dirname(__file__)
    ruta_al_archivo = os.path.join(ruta_carpeta, 'medical_insurance_data.csv')
    # Usamos latin-1 por si el archivo viene de Excel/Windows con acentos
    return pd.read_csv(ruta_al_archivo, encoding='latin-1') 

df_ia = cargar_datos_csv()

# --- SIDEBAR (Entradas para el suscritor) ---
st.sidebar.header("Perfil del Asegurado")
age = st.sidebar.number_input("Edad", min_value=18, max_value=100, value=30)
sex = st.sidebar.selectbox("Sexo", ['female', 'male'])
bmi = st.sidebar.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.selectbox("Número de Hijos", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("¿Es fumador?", ['yes', 'no'])
region = st.sidebar.selectbox("Región", ['southwest', 'southeast', 'northwest', 'northeast'])

# --- INTERFAZ: PESTAÑAS (TABS) ---
tab_prediccion, tab_chat = st.tabs(["🔮 Simulador de Riesgo", "🤖 InsurTrust Smart Chat"])

with tab_prediccion:
    st.header("🔍 Evaluación de Riesgo de Suscripción Médica")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Utilice el panel de la izquierda para configurar los parámetros del solicitante.")
        
    if st.button("Ejecutar Dictaminación de Riesgo"):
        input_data = pd.DataFrame([[ # Creamos el DataFrame con las columnas exactas del dataset
            age, sex, bmi, children, smoker, region 
        ]], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        # Aquí irá la lógica de predicción cuando tengamos el .joblib listo
        try:
            prediccion = model.predict(input_data)[0]
            probabilidad = model.predict_proba(input_data)[0][1]

            st.markdown("### 📊 Diagnóstico Actuarial Inteligente")

            if prediccion == 1:
                st.error(f"⚠️ ALTO RIESGO DE SINIESTRALIDAD (Probabilidad: {probabilidad:.2%})")
                # Sugerencia para Riesgo Alto
                st.info("📋 **Sugerencia de InsurTrust:** Se recomienda solicitar exámenes médicos de laboratorio o aplicar una extraprima por factores de riesgo detectados.")
            else:
                st.success(f"✅ RIESGO ESTÁNDAR DETECTADO (Confianza: {(1-probabilidad):.2%})")
                # Fusionamos las dos sugerencias en una sola más elegante
                st.info("💡 **Sugerencia de InsurTrust:** El perfil es elegible para **emisión inmediata** con tarifa preferente. No se requieren requisitos médicos adicionales ni revisiones manuales.")
                st.write("El perfil cumple con los parámetros de suscripción estándar.")

        except Exception as e:
            st.error(f"Error en el motor: {e}")
            
with tab_chat:
    st.header("🤖 Chat con tus Datos (Powered by Gemini)")
    st.markdown("Esta sección utiliza Agentes de IA para analizar tus bases de datos en tiempo real.")
    
    if df_ia is not None:
        user_api_key = st.text_input("Introduce tu Google API Key (Gemini):", type="password")
        
        if user_api_key:
            pregunta = st.text_input("Hazle una pregunta a la base de datos de InsurTrust (AXA):")
            
            if pregunta:
                res_final = ""
                try:
                    # 1. Configuración del LLM
                    llm = ChatGoogleGenerativeAI(
                         model="gemini-flash-latest", 
                         google_api_key=user_api_key,
                         temperature=0
                    )   
                    
                    # 2. Prefix optimizado
                    prefix = """
                    You are a Senior Medical Underwriter at AXA. You work with a dataframe 'df'.
                    You must answer questions by writing Python code. 

                    RULES:
                    1. Always start with 'Thought:'.
                    2. Then use 'Action: python_repl_ast'.
                    3. Then 'Action Input:' followed by the code.
                    4. After seeing the 'Observation', conclude with 'Final Answer:' in Spanish.

                    EXAMPLE:
                    Question: ¿Cuántos asegurados NO fuman, tienen BMI < 25 y cargos > 13270?
                    Thought: I need to filter the dataframe for non-smokers with BMI below 25 and charges above the average.
                    Action: python_repl_ast
                    Action Input: len(df[(df['smoker'] == 'no') & (df['bmi'] < 25) & (df['charges'] > 13270)])
                    Observation: 19
                    Final Answer: He encontrado 19 registros que cumplen con estos criterios de siniestralidad atípica.

                    DATA CONTEXT:
                    - Average charges: $13,270.42.
                    - Target Segment: (smoker == 'no') & (bmi < 25) & (charges > 13270).
                    """

                    # 3. Creación del Agente
                    agent = create_pandas_dataframe_agent(
                        llm, 
                        df_ia, 
                        verbose=True, 
                        allow_dangerous_code=True,
                        handle_parsing_errors="Check your output! You must use the format: Thought: <your thought>, Action: python_repl_ast, Action Input: <code to run>. If you have the answer, use Final Answer: <your answer in Spanish>",# <--- ESTO ES LO QUE PIDIÓ EL ERROR
                        max_iterations=15, # Subimos de 10 a 15 para darle más tiempo de corregirse
                        agent_type="zero-shot-react-description",
                        prefix=prefix
                    )

                    with st.spinner("LuxLogistics AI está analizando tu consulta..."):
                        try:
                            # Intento de ejecución normal
                            resultado = agent.invoke(pregunta)
                            res_final = resultado['output']
                        except Exception as parse_err:
                            # --- ESCUDO DE RESCATE SI HAY ERROR DE FORMATO ---
                            error_str = str(parse_err)
                            res_final=""
                            #1. EL FILTRO DE SEGURIDAD (PRIORIDAD ALTA)
                            if "429" in error_str or "quota" in error_str.lower():
                                res_final = "⚠️ Límite de Google alcanzado. Espera 1 minuto."
                                st.stop() # <--- CAMBIO CLAVE: Detiene la App aquí limpiamente
                            #2.eL ESCUDO DE EXTRACCIÓN (SI HAY RESPUESTA PERO ESTÁ SUCIA)
                            elif "Final Answer:" in error_str:
                                res_final = error_str.split("Final Answer:")[-1]
                            elif "Could not parse LLM output: `" in error_str:
                                res_final = error_str.split("Could not parse LLM output: `")[-1]
                            else:
                                res_final = error_str


                        # --- LA GUILLOTINA DEFINITIVA (Limpieza de Links y basura técnica) ---
                        # Cortamos en seco si aparece cualquiera de estas frases técnicas
                        for basura in ["For troubleshooting", "visit:", "https://", "Agent stopped"]:
                            res_final = res_final.split(basura)[0]
                        
                        # Limpieza final de caracteres de código
                        res_final = res_final.replace("`", "").strip()

                        # --- MOSTRAR RESULTADO ---
                        if len(res_final) > 0:# Bajamos de 5 a 0,Si hay CUALQUIER cosa, la mostramos
                            st.success("✅ Análisis Completado(Recuperado)")
                            st.markdown(f"## {res_final}")
                        else:
                            st.error("La IA se quedó en blanco. Intenta reformular.")
                            with st.expander("Ver detalle técnico"):
                                st.write(error_str)

                except Exception as e:
                    st.error(f"Error crítico en el motor: {e}")
        else:
            st.info("💡 Introduce tu API Key para activar el Oráculo de AXA.")
    else:
        st.error("❌ El archivo 'medical_insurance_data.csv' no está en la carpeta.")