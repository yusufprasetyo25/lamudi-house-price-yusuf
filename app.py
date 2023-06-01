import streamlit as st
import numpy as np
import pandas as pd
import geopy.distance
from PIL import Image
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from feature_engine.encoding import OrdinalEncoder
from xgboost import XGBRegressor
import folium
from streamlit_folium import folium_static

def get_average_njop_near(lat, long, df_njop):
    """
    Dependencies: numpy as np, pandas as pd, geopy.distance
    Input: Latitude (float), Longitude (float), Radius in km (float)
    Output: Average NJOP value based on df_njop in the radius (float)
    """
    d_lat = 2/110 # let's use upper bound of 110000
    d_long = 2/111 # let's assume upper bound of cos(0) and 111000
    df = df_njop.copy(deep=True)
    df = df[ (df['latitude'].between(lat-d_lat, lat+d_lat)) & (df['longitude'].between(long-d_long, long+d_long)) ] # bbox
    try:
        df['distance'] = df.apply(lambda x: geopy.distance.distance((lat, long) , (x['latitude'], x['longitude'])).km, axis=1)
    except:
        df['distance'] = np.nan
    df = df[ df['distance'] <= 2 ]
    return [df['ave_njb'].mean(), df['ave_njb'].count(), df['ave_njb'].min(), df['ave_njb'].max(), df['ave_njb'].std()]

def get_pixel(lat, long):
    return [ int(round(160+(478-160)*(long-(106.7351))/(106.91427-((106.7351))))), int(round(145+(654-145)*(lat-(-6.0808))/(-6.36549-((-6.0808))))) ]

def get_flood_scenario(lat, long):
    colors_list = []
    pixel = get_pixel(lat, long)
    flood_0 = 0
    flood_1 = 0
    flood_2 = 0
    flood_3 = 0
    flood_4 = 0
    flood_5 = 0
    flood_6 = 0
    for i in range(7):
        im = Image.open(f"./flood_{i}.png")
        rgb_im = im.convert('RGB')
        for h_px in range(-3,4):
            for v_px in range(-3,4):
                try:
                    colors_list.append( rgb_im.getpixel((pixel[0]+v_px, pixel[1]+h_px)) )
                except:
                    pass
        if i == 0 and (225, 87, 89) in colors_list:
            flood_0 = 1
        if i == 1 and (237, 201, 72) in colors_list:
            flood_1 = 1
        if i == 2 and (78, 121, 167) in colors_list:
            flood_2 = 1
        if i == 3 and (242, 142, 43) in colors_list:
            flood_3 = 1
        if i == 4 and (89, 161, 79) in colors_list:
            flood_4 = 1
        if i == 5 and (176, 122, 161) in colors_list:
            flood_5 = 1
        if i == 6 and (156, 117, 95) in colors_list:
            flood_6 = 1
    return [flood_0, flood_1, flood_2, flood_3, flood_4, flood_5, flood_6]


def get_facilities_near(lat, long, df_facility):
    """
    Dependencies: numpy as np, pandas as pd, geopy.distance
    Input: Latitude (float), Longitude (float), Radius in km (float)
    Output: Columns of facility features, that is:
        1. Count Mall facilities near lat and long
        2. Max stars of the Mall facilities near lat and long
        3. Sum of count reviews of the Mall facilities near lat and long
        4. Count Hospital facilities near lat and long
        5. Max stars of the Hospital facilities near lat and long
        6. Sum of count reviews of the Hospital facilities near lat and long
        7. Count Bus Station facilities near lat and long
        9. Sum of count reviews of the Bus Station facilities near lat and long
        8. Max stars of the Train Station facilities near lat and long
        10. Sum of count reviews of the Train Station facilities near lat and long
    """
    d_lat = 2/110 # let's use upper bound of 110000
    d_long = 2/111 # let's assume upper bound of cos(0) and 111000
    df = df_facility.copy(deep=True)
    df = df[ (df['latitude'].between(lat-d_lat, lat+d_lat)) & (df['longitude'].between(long-d_long, long+d_long)) ] # bbox
    try:
        df['distance'] = df.apply(lambda x: geopy.distance.distance((lat, long) , (x['latitude'], x['longitude'])).km, axis=1)
    except:
        df['distance'] = np.nan
    df = df[ df['distance'] <= 2 ]
    return [
        df[ df['category'] == 'mall']['name'].nunique()
        , df[ df['category'] == 'mall']['stars'].max()
        , df[ df['category'] == 'mall']['count_reviews'].sum()
        , df[ df['category'] == 'hospital']['name'].nunique()
        , df[ df['category'] == 'hospital']['stars'].max()
        , df[ df['category'] == 'hospital']['count_reviews'].sum()
        , df[ df['category'] == 'bus_station']['name'].nunique()
        , df[ df['category'] == 'bus_station']['count_reviews'].sum()
        , df[ df['category'] == 'train_station']['name'].nunique()
        , df[ df['category'] == 'train_station']['count_reviews'].sum()
    ]

def df_facilities_near(lat, long, df_facility):
    """
    Dependencies: numpy as np, pandas as pd, geopy.distance
    Input: Latitude (float), Longitude (float), Radius in km (float)
    Output: DataFrame of near facilites
    """
    d_lat = 2/110 # let's use upper bound of 110000
    d_long = 2/111 # let's assume upper bound of cos(0) and 111000
    df = df_facility.copy(deep=True)
    df = df[ (df['latitude'].between(lat-d_lat, lat+d_lat)) & (df['longitude'].between(long-d_long, long+d_long)) ] # bbox
    try:
        df['distance'] = df.apply(lambda x: geopy.distance.distance((lat, long) , (x['latitude'], x['longitude'])).km, axis=1)
    except:
        df['distance'] = np.nan
    return df[ df['distance'] <= 2 ]

category_icon_dict = {
    'mall':'shop'
    , 'hospital': 'hospital'
    , 'train_station': 'train'
    , 'bus_station': 'bus'
}

def transform_to_rupiah_format(value):
    str_value = str(value)
    separate_decimal = str_value.split(".")
    after_decimal = separate_decimal[0]
    before_decimal = separate_decimal[1]

    reverse = after_decimal[::-1]
    temp_reverse_value = ""

    for index, val in enumerate(reverse):
        if (index + 1) % 3 == 0 and index + 1 != len(reverse):
            temp_reverse_value = temp_reverse_value + val + "."
        else:
            temp_reverse_value = temp_reverse_value + val

    temp_result = temp_reverse_value[::-1]

    return "Rp" + temp_result + "," + before_decimal

df = pd.read_csv('kota_kecamatan_kelurahan_jalan_latlong.csv')
abs_test_error = 0.236

st.markdown('# Prediksi Harga Rumah Jakarta')

kota = st.selectbox('Pilih kota:', tuple(sorted(df['kota'].unique())))
kecamatan = st.selectbox('Pilih kecamatan:', tuple(sorted(df[df['kota'] == kota]['kecamatan'].unique())))
kelurahan = st.selectbox('Pilih kelurahan:', tuple(sorted(df[df['kecamatan'] == kecamatan]['kelurahan'].unique())))
nama_jalan = st.selectbox('Pilih nama jalan:', tuple(sorted(df[df['kelurahan'] == kelurahan]['nama_jalan'].unique())))

bedrooms = st.number_input('Masukan jumlah kamar tidur', min_value=1, max_value=50, step=1)
bathrooms = st.number_input('Masukan jumlah kamar mandi:', min_value=1, max_value=50, step=1)
floor = st.number_input('Masukan jumlah lantai:', min_value=1, max_value=50, step=1)
land_size = st.number_input('Masukan luas tanah (meter-persegi):', min_value=6, max_value=1000, step=1)
building_size = st.number_input('Masukan luas bangunan (meter-persegi):', min_value=6, max_value=1000, step=1)

calculate = st.button('Hitung harga rumah!')

if calculate:
    lat = df[df['nama_jalan'] == nama_jalan]['latitude'].iloc[0]
    long = df[df['nama_jalan'] == nama_jalan]['longitude'].iloc[0]
    df_njop = pd.read_csv('NJOP_coordinate.csv')
    X = pd.DataFrame(data={
        'data_bedrooms_s': [bedrooms]
        , 'data_bathrooms_s': [bathrooms]
        , 'floor': [floor]
        , 'data_land_size_s': [land_size]
        , 'data_building_size_s': [building_size]
    })
    X[['njop', 'neighbors', 'min_njop', 'max_njop', 'std_njop']] = get_average_njop_near(lat, long, df_njop=pd.read_csv('NJOP_coordinate.csv'))
    X[['flood_0', 'flood_1', 'flood_2', 'flood_3', 'flood_4', 'flood_5', 'flood_6']] = get_flood_scenario(lat, long)
    X[[
        'count_mall'
        , 'max_stars_mall'
        , 'sum_reviews_mall'
        , 'count_hospital'
        , 'max_stars_hospital'
        , 'sum_reviews_hospital'
        , 'count_bus_st'
        , 'sum_reviews_bus_st'
        , 'count_train_st'
        , 'sum_reviews_train_st'	
    ]] = get_facilities_near(
        lat, long, pd.read_csv('facility_latlong.csv')
    )
    X[['max_stars_mall', 'max_stars_hospital']] = X[['max_stars_mall', 'max_stars_hospital']].fillna(0)
    X[['latitude', 'longitude', 'kota', 'kecamatan']] = [lat, long, kota, kecamatan]
    # st.dataframe(X)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(X)
    bottom_price = transform_to_rupiah_format(np.round(predictions[0]*(land_size+building_size)/(1+abs_test_error)))
    price = transform_to_rupiah_format(np.round(predictions[0]*(land_size+building_size)))
    upper_price = transform_to_rupiah_format(np.round(predictions[0]*(land_size+building_size)/(1-abs_test_error)))

    st.markdown(f'#### Harga rumah adalah :green[{price}] :star:')
    st.markdown(f'##### Batas bawah harga rumah adalah :blue[{bottom_price}] :chart_with_downwards_trend:') # from test error
    st.markdown(f'##### Batas atas harga rumah adalah :red[{upper_price}] :chart_with_upwards_trend:') # from test error

    # plot map
    m = folium.Map(location=[lat, long], zoom_start=14)
    # add house marker
    folium.Marker(
        location=[lat, long]
        , popup = 'House'
        , icon=folium.Icon(color='green', icon='house-chimney', prefix='fa')
    ).add_to(m)
    # add facility marker
    for index, row in df_facilities_near(lat, long, pd.read_csv('facility_latlong.csv')).iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']]
            , popup = row['name']
            , icon=folium.Icon(icon=category_icon_dict[row['category']], prefix='fa')
        ).add_to(m)
    st_data = folium_static(m, width=725)
    st.markdown('Catatan: klik pada pin untuk melihat detail fasilitas dekat rumah')