import streamlit as st
import pandas as pd
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import geopandas as gpd
from unidecode import unidecode
import datetime
import numpy as np
import seaborn as sns
import config
import sqlite3
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from sqlalchemy import create_engine
import sqlalchemy.pool

APP_TITLE = 'Ocupación hotelera en España'
APP_SUB_TITLE = 'Fuente: Ine'
if 'desplegable' not in st.session_state:
    st.session_state['desplegable'] = None

@st.cache_data
def get_keys_with_value(dic, value):
    
    return [key for key in dic if dic[key][3:] == value][0]

def display_filtros(df, origen):
    year_list = list(df['año'].unique())
    year_list.sort(reverse = True )
    year = st.sidebar.selectbox('año', year_list, 0)
    df_anyo= df[df['año']==year]
    month_list = list(df_anyo['mes'].unique())
    # month_list = list(df['mes'].unique())

    # if year == '2023':
    #     month_list = ['01', '02' ,'03']
    #     # mes = st.sidebar.selectbox('mes', ['01', '02' ,'03'])
    # else:
    #     month_list = list(df['mes'].unique())
    mes = st.sidebar.selectbox('mes', month_list, 0)

    # if year == 2023:
    #     mes = st.sidebar.radio('trimestre', [1])
    # else:
    #     quarter = st.sidebar.radio('trimestre', [1, 2, 3, 4])


    st.header(f'{year}/{mes} - {origen}' )
    return year, mes

def display_provincia(df, prov):
    prov_name = st.sidebar.selectbox('Provincia', prov_list)
    return prov_name

def display_origen_filter():
    return st.sidebar.radio('origen', ['Ambos_origenes', 'Nacional', 'Internacional'])

def display_map(df, year, month, origen):
    
   # df = df[(df['año'] == year) & (df['trimestre'] == quarter) & (df['origeno'] == origen)]
    df = df[(df['año'] == year) & (df['mes'] == month)]
    m = folium.Map(location=[40.42,  -3.7], zoom_start=5)
    print(df)

    #creo una nueva columna y aplico una escala logaritmica con los valores transformados
    df['log_' + origen] = np.log1p(df[origen])

    # Crear los 10 intervalos para el mapa de coropletas
    bins = np.linspace(df['log_' + origen].min(), df['log_' + origen].max(), num=12)
    coropletas = folium.Choropleth(
        geo_data=provincias,
        name="choropleth",
        data=df,
        columns=["codProv", 'log_' + origen],  # Usar la nueva columna con los valores transformados
        key_on="properties.codProv",
        bins=bins,
        fill_color="Blues",
        fill_opacity=0.7,
        line_opacity=1.0,
        legend_name="Tasa de ocupacion"
    )
    coropletas.add_to(m)

    # st.markdown(m._repr_html_(), unsafe_allow_html=True)

    # valor_min = df[origen].min()
    # valor_max = df[origen].max()

    # # Crear los 10 intervalos para el mapa de coropletas
    # bins = np.linspace(valor_min, valor_max, num=8)
    # coropletas = folium.Choropleth(geo_data=provincias,name="choropleth",data=df,columns=["codProv", origen],key_on="properties.codProv", bins = bins,fill_color="Blues",fill_opacity=0.7,line_opacity=1.0,legend_name="Tasa de ocupacion")
    # coropletas.add_to(m)
    for feature in coropletas.geojson.data['features']:
       code = feature['properties']['codProv']
       #feature['properties']['Provincia'] = prov_dict[code]
    coropletas.geojson.add_child(folium.features.GeoJsonTooltip(['provincia'], labels=False))
    
    folium.LayerControl().add_to(m)
    st_map = st_folium(m, width=700, height=450)
    codigo = '00'
    if st_map['last_active_drawing']:
        codigo = st_map['last_active_drawing']['properties']['codProv']
    return codigo

def display_datos_ocup(df, year, month, origen, prov_code):
    df = df[(df['año'] == year) & (df['mes'] == month) & (df['codProv'] == prov_code)]    
   # st.metric(origen, str(df.Paro.iat[0])+' %')
    print(df[origen])
    print(origen)
   # print(year,month,prov_name,df)
    
    st.subheader(f'Ocupación de tipo {origen} en {prov_name}:')   
    st.metric('valor', df[origen])
    
def display_grafica(df, year, month, origen, codProvin, provincia):
    df = df[(df['año'] == year) & (df['mes'] == month)]    
   # st.metric(origen, str(df.Paro.iat[0])+' %')
    print(df[origen])
    df_top = df.nlargest(10, origen)
    provincias_top = df_top['codProv']
    data_top = df_top[origen]

    provincias = list(provincias_top)
    if codProvin not in provincias:
        provincias.append(codProvin)
        data_top = list(data_top)
        data_top.append(df[df['codProv'] == codProvin][origen].values[0])

    chart_data = pd.DataFrame({'Provincias': provincias, 'Datos': data_top})

    sns.set(style="whitegrid")

    fig, ax = plt.subplots()
    colors = sns.color_palette("viridis", len(chart_data))
    sns.barplot(x='Provincias', y='Datos', data=chart_data, ax=ax, palette=colors)

    ax.set_xlabel('Provincias')
    ax.set_ylabel(origen)
    ax.set_title(f'Provincias en {year}/{month} con mayor ocupación "{format(origen)}" comparadas con {provincia}')
    ax.tick_params(axis='x', rotation=45)
    sns.despine()

    st.pyplot(fig)

    

# st.set_page_config(page_title='Ocupación_hotelera', page_icon = ":hotel:", layout = 'wide', initial_sidebar_state = 'auto')
st.set_page_config(page_title= APP_TITLE, page_icon = ":hotel:", layout = 'wide', initial_sidebar_state = 'auto')
st.title(APP_TITLE)
st.caption(APP_SUB_TITLE)

# df2 = pd.read_csv(r'2066.csv',sep=';',encoding="utf-8",on_bad_lines='skip')
df3 = pd.read_csv(r'2074.csv',sep=';',encoding="utf-8",on_bad_lines='skip')


provincias = gpd.read_file('provincias.geojson')

# df_OcupHabitProv = df2.pivot(index=['Periodo','Comunidades y Ciudades Autónomas', 'Provincias'], columns='Establecimientos y personal empleado (plazas)', values='Total').reset_index()
# df_OcupHabitProv['fecha'] = pd.to_datetime(df_OcupHabitProv['Periodo'].str.extract('(\d{4})M(\d{2})').apply(lambda x: '-'.join(x), axis=1), format='%Y-%m')
# df_OcupHabitProv['trimestre'] = df_OcupHabitProv['fecha'].dt.quarter
# df_OcupHabitProv['año'] = df_OcupHabitProv['Periodo'].str[:4]

# df_OcupHabitProv['mes'] = df_OcupHabitProv['Periodo'].str[5:]
# ####evolución de los datos de 2022
# #ocupacion_22 = df_OcupHabitProv[df_OcupHabitProv['año']=='2022']
# df_OcupHabitProv['Provincias'].fillna(df_OcupHabitProv['Comunidades y Ciudades Autónomas'], inplace=True)
# #si provincia es nan copiar comunidad autonoma
# df_ocupacion = df_OcupHabitProv.dropna(subset=['Provincias', 'Comunidades y Ciudades Autónomas'])

# #elimino de las provincias la media por comunidad para que salga bien el mapa

# eliminar = ['180Ceuta','19 Melilla', '01 Andalucía', '02 Aragón', '05 Canarias', '07 Castilla y León', '08 Castilla - La Mancha', '09 Cataluña','10 Comunitat Valenciana','11 Extremadura','12 Galicia', '16 País Vasco', ]
# #eliminar = ['01 Andalucía', '02 Aragón', '05 Canarias', '07 Castilla y León', '08 Castilla - La Mancha', '09 Cataluña','10 Comunitat Valenciana','11 Extremadura','12 Galicia', '16 País Vasco', ]
# df_ocupacion = df_ocupacion[~df_ocupacion.Provincias.isin(eliminar)]

# #ocupacion_20 = df_OcupHabitProv[df_OcupHabitProv['año']=='2020']
# #ocupacion_20['Provincias'].fillna(ocupacion_20['Comunidades y Ciudades Autónomas'], inplace=True)
# #si provincia es nan copiar comunidad autonoma
# #ocupacion_2020 = ocupacion_20.dropna(subset=['Provincias', 'Comunidades y Ciudades Autónomas'])



# df_ocupacion['codProv'] = df_ocupacion['Provincias'].str.upper()
# df_ocupacion['codProv'] = df_ocupacion['codProv'].str.strip()
# df_ocupacion['codProv'] = df_ocupacion['codProv'].str[3:7].apply(quitar_acentos)


# provincias['codProv'] = provincias['pName'].str.upper()
# provincias['codProv'] = provincias['codProv'].str.strip()
# provincias['codProv'] = provincias['codProv'].str[:4].apply(quitar_acentos)

# #df_ocupacion['Grado de ocupación por plazas'] = pd.to_numeric(df_ocupacion['Grado de ocupación por plazas'], errors='coerce')
# #df_ocupacion.dropna(subset=['Grado de ocupación por plazas'], inplace=True)
# df_ocupacion['Grado de ocupación por plazas']=df_ocupacion['Grado de ocupación por plazas'].str.replace(',', '').str.replace('.', '')
# df_ocupacion['Grado de ocupación por plazas'] = pd.to_numeric(df_ocupacion['Grado de ocupación por plazas'], errors='coerce')
# df_ocupacion['Grado de ocupación por plazas en fin de semana']=df_ocupacion['Grado de ocupación por plazas en fin de semana'].str.replace(',', '').replace('.', '')
# df_ocupacion['Grado de ocupación por plazas'] = pd.to_numeric(df_ocupacion['Grado de ocupación por plazas en fin de semana'], errors='coerce')
# #df_ocupacion['Número de plazas estimadas']=df_ocupacion['Número de plazas estimadas'].str.replace(',', '').replace('.', '').astype(float)
# #df_ocupacion['Personal empleado']=df_ocupacion['Personal empleado'].str.replace(',', '').astype(float)
# #df_ocupacion['Número de establecimientos abiertos estimados']=df_ocupacion['Número de establecimientos abiertos estimados'].str.replace(',', '').replace('.', '').astype(float)

# media_trimestral = df_ocupacion.groupby(['año', 'trimestre'])['Grado de ocupación por plazas'].mean().reset_index()




provincias['pName'] = provincias['provincia']
provincias.loc[provincias['pName'] == 'La Rioja', 'pName'] = 'Rioja, La'
provincias.loc[provincias['pName'] == 'A Coruña', 'pName'] = 'Coruña, A '
provincias.loc[provincias['pName'] == 'Las Palmas', 'pName'] = 'Palmas, Las'
provincias.loc[provincias['pName'] == 'Illes Balears', 'pName'] = 'Balears, Illes'
provincias.loc[provincias['pName'] == 'Alacant', 'pName'] = 'Alicante/Alacant'

def quitar_acentos(texto):
    return unidecode(texto)

###MAPA turismo español vs extranjero
pernoct= df3[df3['Viajeros y pernoctaciones']=='Pernoctaciones']
df_nacional_inter = pernoct.pivot(index=['Periodo','Comunidades y Ciudades Autónomas', 'Provincias'], columns='Residencia: Nivel 2', values='Total').reset_index()
df_nacional_inter['año'] = df_nacional_inter['Periodo'].str[:4]
df_nacional_inter['mes'] = df_nacional_inter['Periodo'].str[5:]
df_nacional_inter = df_nacional_inter.rename(columns={np.nan: 'Ambos_origenes'})
#resi_22 = df_nacional_inter[df_nacional_inter['año']=='2022']
#df_nacional_inter['Provincias'].fillna(df_nacional_inter['Comunidades y Ciudades Autónomas'], inplace=True)
#si provincia es nan copiar comunidad autonoma
df_nacional_inter = df_nacional_inter.dropna(subset=['Provincias', 'Comunidades y Ciudades Autónomas'])

eliminar = ['18 Ceuta','19 Melilla', '01 Andalucía', '02 Aragón', '05 Canarias', '07 Castilla y León', '08 Castilla - La Mancha', '09 Cataluña','10 Comunitat Valenciana','11 Extremadura','12 Galicia', '16 País Vasco', ]
# eliminar = ['01 Andalucía', '02 Aragón', '05 Canarias', '07 Castilla y León', '08 Castilla - La Mancha', '09 Cataluña','10 Comunitat Valenciana','11 Extremadura','12 Galicia', '16 País Vasco', ]
df_nacional_inter = df_nacional_inter[~df_nacional_inter.Provincias.isin(eliminar)]

df_nacional_inter['codProv'] = df_nacional_inter['Provincias'].str.upper()
df_nacional_inter['codProv'] = df_nacional_inter['codProv'].str.strip()
df_nacional_inter['codProv'] = df_nacional_inter['codProv'].str[3:7].apply(quitar_acentos)

provincias['codProv'] = provincias['pName'].str.upper()
provincias['codProv'] = provincias['codProv'].str.strip()
provincias['codProv'] = provincias['codProv'].str[:4].apply(quitar_acentos)

df_nacional_inter['Ambos_origenes']=df_nacional_inter['Ambos_origenes'].str.replace(',', '').str.replace('.', '')
df_nacional_inter['Ambos_origenes']=pd.to_numeric(df_nacional_inter['Ambos_origenes'], errors='coerce')
df_nacional_inter['Residentes en España']=df_nacional_inter['Residentes en España'].str.replace(',', '').str.replace('.', '')
df_nacional_inter['Residentes en España']=pd.to_numeric(df_nacional_inter['Residentes en España'], errors='coerce')
df_nacional_inter['Residentes en el Extranjero']=df_nacional_inter['Residentes en el Extranjero'].str.replace(',', '').str.replace('.', '')
df_nacional_inter['Residentes en el Extranjero']=pd.to_numeric(df_nacional_inter['Residentes en el Extranjero'], errors='coerce')
df_nacional_inter = df_nacional_inter.rename(columns={'Residentes en España': 'Nacional'})
df_nacional_inter = df_nacional_inter.rename(columns={'Residentes en el Extranjero': 'Internacional'})
#resi_22['Residentes en España']=resi_22['Residentes en España'].str.replace(',', '').astype(float)
#resi_22['Residentes en el extranjero']=resi_22['Residentes en el Extranjero'].str.replace(',', '').astype(float)
#media_resi_esp = resi_22.groupby(['codProv', 'año'])['Residentes en España'].mean()
#media_resi_ext = resi_22.groupby(['codProv', 'año'])['Residentes en el Extranjero'].mean()






#prov_paro = 'TasaParoProvSeTr.csv'
#prov_data = pd.read_csv(prov_paro, encoding='utf-8')
#origeno	Codigo	Provincia	Trimestre	Paro
#El código de provincia en el geojson es str y con cero a la izquierda
#prov_data['codigo'] = prov_data['codigo'].astype(str).str.zfill(2)



prov_list = list(df_nacional_inter['Provincias'].str[3:].unique())

#code3 = prov_code
#cod4 = '00'
#prov_dict = pd.Series(prov_data.Provincia.values,index=prov_data.codigo).to_dict()
prov_dict = pd.Series(df_nacional_inter[df_nacional_inter["mes"]=="12"].Provincias.values,index=df_nacional_inter[df_nacional_inter["mes"]=="12"].codProv).to_dict()
origen = display_origen_filter()
#st.write('Opción seleccionada:', origen)
year, month = display_filtros(df_nacional_inter, origen)
prov_name = display_provincia(df_nacional_inter, '')
prov_code= display_map(df_nacional_inter, year, month, origen)


#if (prov_code2 == '00'):
 #       cod3 = prov_code2
  #      cod4 = prov_code2
        
st.write(st.session_state['desplegable'], prov_name, st.session_state['desplegable'] != prov_name, prov_code)

if st.session_state['desplegable'] != prov_name or prov_code == '00':
        # prov_name = prov_dict[]
        prov_code = get_keys_with_value(prov_dict, prov_name)
        st.session_state['desplegable'] = prov_name
else: 
     prov_name = prov_dict[prov_code][3:]


display_datos_ocup(df_nacional_inter, year, month, origen, prov_code)

# col1, col2, col3 = st.columns(3)
# with col1:
#     display_datos_ocup(df_nacional_inter, year, month, 'Ambos_origenes', prov_code)
# with col2:
#     display_datos_ocup(df_nacional_inter, year, month, 'Nacional', prov_code)
# with col3:
#     display_datos_ocup(df_nacional_inter, year, month, 'Internacional', prov_code)


display_grafica(df_nacional_inter, year, month, origen, prov_code,prov_dict[prov_code][3:])





def answer():
    uri = "file::memory:?cache=shared"
    table_name = 'prov_datadb'

    # commit data to sql

    conn = sqlite3.connect(':memory:')

    df_nacional_inter.to_sql(table_name, conn, if_exists='replace', index=False)

    # create db engine
    eng = create_engine(
        url='sqlite:///file:memdb1?mode=memory&cache=shared', 
        poolclass=sqlalchemy.pool.StaticPool, # single connection for requests
        creator=lambda: conn)
    db = SQLDatabase(engine=eng)

    # create open AI conn and db chain
    if config.openai_api_key:
      llm = OpenAI(
          openai_api_key=config.openai_api_key, # st.secrets
          temperature=0, # creative scale
          max_tokens=300)
      db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

    # run query and display result
    if config.openai_api_key:
        config.answer_text = db_chain.run(st.session_state.question)
        st.write(config.answer_text)

#Query IA
user_q = st.text_input(
    "Pregunta: ", 
    help="Haz una pregunta relativa al Dataset", key='question', on_change=answer)

st.write(config.answer_text)

