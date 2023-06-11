
import streamlit as st
import pandas as pd
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import geopandas as gpd
from unidecode import unidecode
import pydeck as pdk
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import networkx as nx


st.set_page_config(page_title= "Puntos turísticos", page_icon = ":hotel:", layout = 'wide', initial_sidebar_state = 'auto')
st.title("mapa de histótico de puntos turísticos")
@st.cache_data
def get_keys_with_value(dic, value):
    
    return [key for key in dic if dic[key][3:] == value][0]


def display_year_month(df):
    year_list = list(df['año'].unique())
    year_list.sort(reverse = True )
    year = st.sidebar.selectbox('año', year_list, 0)
    df_anyo= df[df['año']==year]
    month_list = list(df_anyo['mes'].unique())
    mes = st.sidebar.selectbox('mes', month_list, 0)

    return year, mes

def display_ciudad(df):
    city_name = st.sidebar.selectbox('Punto Turístico', ciudad_list)
    return city_name

def display_origen_filter():
    return st.sidebar.radio('origen', ['Residentes en España', 'Residentes en el Extranjero'])



ciudades = pd.read_excel("listado-longitud-latitud-municipios-espana.xls", header=2)
df4 = pd.read_csv(r'2078.csv',sep=';',encoding="utf-8",on_bad_lines='skip')
print(ciudades)
print('dd--', df4)

df4 = df4.rename(columns={"Puntos turísticos": 'Población'})
df4['Población'] = df4['Población'].str[6:]
print('dd33--', df4)
# df_ciu= ciudades['Población', 'Latitud', 'Longitud']
df_ciudades = pd.merge(df4, ciudades[['Población', 'Latitud', 'Longitud']], on='Población')
df_ciudades['año'] = df_ciudades['Periodo'].str[:4]
df_ciudades['mes'] = df_ciudades['Periodo'].str[5:]
df_ciudades =df_ciudades[df_ciudades['Viajeros y pernoctaciones']== "Pernoctaciones"]
df_ciudades_data = df_ciudades.pivot(index=['Periodo','Población', 'Latitud', 'Longitud', 'año', 'mes'], columns='Residencia', values='Total').reset_index()
df_ciudades_data['Residentes en España']=df_ciudades_data['Residentes en España'].str.replace(',', '').str.replace('.', '')
df_ciudades_data['Residentes en España']=pd.to_numeric(df_ciudades_data['Residentes en España'], errors='coerce')
df_ciudades_data['Residentes en el Extranjero']=df_ciudades_data['Residentes en el Extranjero'].str.replace(',', '').str.replace('.', '')
df_ciudades_data['Residentes en el Extranjero']=pd.to_numeric(df_ciudades_data['Residentes en el Extranjero'], errors='coerce')
df_ciudades_data = df_ciudades_data.dropna(subset=['Residentes en el Extranjero'])
df_ciudades_data = df_ciudades_data.dropna(subset=['Residentes en España'])
print(df_ciudades_data)

# df_ciu['año'] = df_ciu['Periodo'].str[:4]
# df_ciu['mes'] = df_ciu['Periodo'].str[5:]
# df_ciudades =df_ciu[df_ciu['Viajeros y pernoctaciones']== "Pernoctaciones"]
# df_ciudades_data = df_ciudades.pivot(index=['Periodo','Población', 'Latitud', 'Longitud', 'año', 'mes'], columns='Residencia', values='Total').reset_index()
# df_ciudades_data['Residentes en España']=df_ciudades_data['Residentes en España'].str.replace(',', '').str.replace('.', '')
# df_ciudades_data['Residentes en España']=pd.to_numeric(df_ciudades_data['Residentes en España'], errors='coerce')
# df_ciudades_data['Residentes en el Extranjero']=df_ciudades_data['Residentes en el Extranjero'].str.replace(',', '').str.replace('.', '')
# df_ciudades_data['Residentes en el Extranjero']=pd.to_numeric(df_ciudades_data['Residentes en el Extranjero'], errors='coerce')
# df_ciudades_data = df_ciudades_data.dropna(subset=['Residentes en el Extranjero'])
# df_ciudades_data = df_ciudades_data.dropna(subset=['Residentes en España'])
# print(df_ciudades_data)

ciudad_list = list(df_ciudades_data['Población'].unique())

year, month = display_year_month(df_ciudades_data)   
ciudad = display_ciudad(df_ciudades_data)
origen = display_origen_filter()

# df_ciudades_data.to_csv('ARCHIVOTRATADO.csv')

# hex_map2('ARCHIVOTRATADO.csv', '2022', '02','Residentes en España')

# origen = 'Residentes en España'

# hex_map2(df_ciudades_data, '2017', '07','Residentes en España')
elevation_scale = st.slider("Escala de elevación", 0, 20, 65)
layer = pdk.Layer(
    "HexagonLayer",
    df_ciudades_data,
    get_position=['Longitud', 'Latitud'],
    get_elevation='log_' + origen,
    auto_highlight=True,
    elevation_scale=elevation_scale,
    pickable=True,
    elevation_range=[0, 3000],
    extruded=True,
    coverage=1,
  
)

# Set the viewport location
view_state = pdk.ViewState(
    longitude=df_ciudades_data['Longitud'].mean(), latitude=df_ciudades_data['Latitud'].mean(), zoom=5, min_zoom=2, max_zoom=15, pitch=40.5, bearing=0,
)
# r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": f"{df_ciudades_data[origen]}"})
r = pdk.Deck(layers=[layer], initial_view_state=view_state)

# r.to_html("hexagon_layer.html")

st.pydeck_chart(r)
df_ciu = df_ciudades_data[(df_ciudades_data['año'] == year) & (df_ciudades_data['mes'] == month)&(df_ciudades_data['Población'] == ciudad)]

# st.subheader(f'{origen} que visitan {ciudad} en {year}/{month}:')   
if df_ciu.empty:
        valor = 0
else:
    valor = df_ciu[origen] # Ejemplo de acceso al valor en el dataset
    

st.subheader(f'Visitantes {ciudad} en {year}/{month}:')   
st.metric(f'{origen}:', valor)










# def handle_hover(x, y, hovered_object):
#     if hovered_object is not None:
#         hexagon_data = hovered_object.get("object")
#         # Display detailed information about the selected hexagon
#         st.json(hexagon_data)
#     else:
#         # Clear the hover information when not hovering over a hexagon
#         st.write("Hover over a hexagon to see detailed information.")

# # Display the Deck visualization with hover interactions
# st.pydeck_chart(r, use_container_width=True, on_mousemove=handle_hover)





# provincias = gpd.read_file('provincias.geojson')

# provincias['pName'] = provincias['provincia']
# provincias.loc[provincias['pName'] == 'La Rioja', 'pName'] = 'Rioja, La'
# provincias.loc[provincias['pName'] == 'A Coruña', 'pName'] = 'Coruña, A '
# provincias.loc[provincias['pName'] == 'Las Palmas', 'pName'] = 'Palmas, Las'
# provincias.loc[provincias['pName'] == 'Illes Balears', 'pName'] = 'Balears, Illes'
# provincias.loc[provincias['pName'] == 'Alacant', 'pName'] = 'Alicante/Alacant'

# def quitar_acentos(texto):
#     return unidecode(texto)


# ###MAPA turismo español vs extranjero
# pernoct= df3[df3['Viajeros y pernoctaciones']=='Pernoctaciones']
# df_nacional_inter = pernoct.pivot(index=['Periodo','Comunidades y Ciudades Autónomas', 'Provincias'], columns='Residencia: Nivel 2', values='Total').reset_index()
# df_nacional_inter['año'] = df_nacional_inter['Periodo'].str[:4]
# df_nacional_inter['mes'] = df_nacional_inter['Periodo'].str[5:]
# df_nacional_inter = df_nacional_inter.rename(columns={np.nan: 'Ambos_origenes'})

# df_nacional_inter = df_nacional_inter.dropna(subset=['Provincias', 'Comunidades y Ciudades Autónomas'])

# eliminar = ['18 Ceuta','19 Melilla', '01 Andalucía', '02 Aragón', '05 Canarias', '07 Castilla y León', '08 Castilla - La Mancha', '09 Cataluña','10 Comunitat Valenciana','11 Extremadura','12 Galicia', '16 País Vasco', ]
# #eliminar = ['01 Andalucía', '02 Aragón', '05 Canarias', '07 Castilla y León', '08 Castilla - La Mancha', '09 Cataluña','10 Comunitat Valenciana','11 Extremadura','12 Galicia', '16 País Vasco', ]
# df_nacional_inter = df_nacional_inter[~df_nacional_inter.Provincias.isin(eliminar)]

# df_nacional_inter['codProv'] = df_nacional_inter['Provincias'].str.upper()
# df_nacional_inter['codProv'] = df_nacional_inter['codProv'].str.strip()
# df_nacional_inter['codProv'] = df_nacional_inter['codProv'].str[3:7].apply(quitar_acentos)

# provincias['codProv'] = provincias['pName'].str.upper()
# provincias['codProv'] = provincias['codProv'].str.strip()
# provincias['codProv'] = provincias['codProv'].str[:4].apply(quitar_acentos)

# df_nacional_inter['Ambos_origenes']=df_nacional_inter['Ambos_origenes'].str.replace(',', '').str.replace('.', '')
# df_nacional_inter['Ambos_origenes']=pd.to_numeric(df_nacional_inter['Ambos_origenes'], errors='coerce')
# df_nacional_inter['Residentes en España']=df_nacional_inter['Residentes en España'].str.replace(',', '').str.replace('.', '')
# df_nacional_inter['Residentes en España']=pd.to_numeric(df_nacional_inter['Residentes en España'], errors='coerce')
# df_nacional_inter['Residentes en el Extranjero']=df_nacional_inter['Residentes en el Extranjero'].str.replace(',', '').str.replace('.', '')
# df_nacional_inter['Residentes en el Extranjero']=pd.to_numeric(df_nacional_inter['Residentes en el Extranjero'], errors='coerce')
# df_nacional_inter = df_nacional_inter.rename(columns={'Residentes en España': 'Nacional'})
# df_nacional_inter = df_nacional_inter.rename(columns={'Residentes en el Extranjero': 'Internacional'})
# prov_list = list(df_nacional_inter['Provincias'].str[3:].unique())
# prov_dict = pd.Series(df_nacional_inter[df_nacional_inter["mes"]=="12"].Provincias.values,index=df_nacional_inter[df_nacional_inter["mes"]=="12"].codProv).to_dict()

# #dataframeEmpleo
# df2 = pd.read_csv(r'2066.csv',sep=';',encoding="utf-8",on_bad_lines='skip')

# df_OcupHabitProv = df2.pivot(index=['Periodo','Comunidades y Ciudades Autónomas', 'Provincias'], columns='Establecimientos y personal empleado (plazas)', values='Total').reset_index()
# df_toMerge = df_OcupHabitProv[['Periodo', 'Personal empleado']]
# df_nacional_inter_empleo = pd.merge(df_nacional_inter, df_toMerge, on='Periodo', how='left')

# df_nacional_inter_empleo['Personal empleado']=df_nacional_inter_empleo['Personal empleado'].str.replace(',', '').str.replace('.', '')
# df_nacional_inter_empleo['Personal empleado'] = pd.to_numeric(df_nacional_inter_empleo['Personal empleado'], errors='coerce')
# df_nacional_inter_empleo = df_nacional_inter_empleo.dropna(subset=['Personal empleado'])
# # df_OcupHabitProv['fecha'] = pd.to_datetime(df_OcupHabitProv['Periodo'].str.extract('(\d{4})M(\d{2})').apply(lambda x: '-'.join(x), axis=1), format='%Y-%m')
# # df_OcupHabitProv['trimestre'] = df_OcupHabitProv['fecha'].dt.quarter
# # df_OcupHabitProv['año'] = df_OcupHabitProv['Periodo'].str[:4]

# # df_OcupHabitProv['mes'] = df_OcupHabitProv['Periodo'].str[5:]
# # ####evolución de los datos de 2022
# # #ocupacion_22 = df_OcupHabitProv[df_OcupHabitProv['año']=='2022']
# # df_OcupHabitProv['Provincias'].fillna(df_OcupHabitProv['Comunidades y Ciudades Autónomas'], inplace=True)
# # #si provincia es nan copiar comunidad autonoma
# # df_ocupacion = df_OcupHabitProv.dropna(subset=['Provincias', 'Comunidades y Ciudades Autónomas'])

# # #elimino de las provincias la media por comunidad para que salga bien el mapa

# # eliminar = ['180Ceuta','19 Melilla', '01 Andalucía', '02 Aragón', '05 Canarias', '07 Castilla y León', '08 Castilla - La Mancha', '09 Cataluña','10 Comunitat Valenciana','11 Extremadura','12 Galicia', '16 País Vasco', ]
# # #eliminar = ['01 Andalucía', '02 Aragón', '05 Canarias', '07 Castilla y León', '08 Castilla - La Mancha', '09 Cataluña','10 Comunitat Valenciana','11 Extremadura','12 Galicia', '16 País Vasco', ]
# # df_ocupacion = df_ocupacion[~df_ocupacion.Provincias.isin(eliminar)]

# # #ocupacion_20 = df_OcupHabitProv[df_OcupHabitProv['año']=='2020']
# # #ocupacion_20['Provincias'].fillna(ocupacion_20['Comunidades y Ciudades Autónomas'], inplace=True)
# # #si provincia es nan copiar comunidad autonoma
# # #ocupacion_2020 = ocupacion_20.dropna(subset=['Provincias', 'Comunidades y Ciudades Autónomas'])



# # df_ocupacion['codProv'] = df_ocupacion['Provincias'].str.upper()
# # df_ocupacion['codProv'] = df_ocupacion['codProv'].str.strip()
# # df_ocupacion['codProv'] = df_ocupacion['codProv'].str[3:7].apply(quitar_acentos)


# # provincias['codProv'] = provincias['pName'].str.upper()
# # provincias['codProv'] = provincias['codProv'].str.strip()
# # provincias['codProv'] = provincias['codProv'].str[:4].apply(quitar_acentos)

# # #df_ocupacion['Grado de ocupación por plazas'] = pd.to_numeric(df_ocupacion['Grado de ocupación por plazas'], errors='coerce')
# # #df_ocupacion.dropna(subset=['Grado de ocupación por plazas'], inplace=True)
# # df_ocupacion['Grado de ocupación por plazas']=df_ocupacion['Grado de ocupación por plazas'].str.replace(',', '').str.replace('.', '')
# # df_ocupacion['Grado de ocupación por plazas'] = pd.to_numeric(df_ocupacion['Grado de ocupación por plazas'], errors='coerce')
# # df_ocupacion['Grado de ocupación por plazas en fin de semana']=df_ocupacion['Grado de ocupación por plazas en fin de semana'].str.replace(',', '').replace('.', '')
# # df_ocupacion['Grado de ocupación por plazas'] = pd.to_numeric(df_ocupacion['Grado de ocupación por plazas en fin de semana'], errors='coerce')
# # #df_ocupacion['Número de plazas estimadas']=df_ocupacion['Número de plazas estimadas'].str.replace(',', '').replace('.', '').astype(float)
# # df_ocupacion['Personal empleado']=df_ocupacion['Personal empleado'].str.replace(',', '').str.replace('.', '').astype(float)
# # df_ocupacion['Personal empleado'] = pd.to_numeric(df_ocupacion['Grado de ocupación por plazas'], errors='coerce')
# # # df_ocupacion['Número de establecimientos abiertos estimados']=df_ocupacion['Número de establecimientos abiertos estimados'].str.replace(',', '').replace('.', '').astype(float)




# tipo = display_origen_filter()

# if tipo == "Origen_turismo":
#     st.header('Graficas Ocupación')
#     year = display_year(df_nacional_inter_empleo)
#     provincia = display_provincia(df_nacional_inter_empleo, '')
#     codProvin = get_keys_with_value(prov_dict, provincia)

#     display_pyramid_top15(df_nacional_inter_empleo, year)
#     display_evolucion_turismo(df_nacional_inter_empleo, codProvin)
# else:
#      st.header('Graficas Empleo')
#      year = display_year(df_nacional_inter_empleo)
#      provincia = display_provincia(df_nacional_inter_empleo, '')
#     #  circular_packing_plot(df_nacional_inter_empleo)
#     #  bubble(df_nacional_inter_empleo, year)
#     #  bubble_plot(df_nacional_inter_empleo, year)
#     #  barra(df_nacional_inter_empleo,year)
#     #  plot_province_map(df_nacional_inter_empleo,year)
#      grafica_donut(df_nacional_inter_empleo, year, '08')