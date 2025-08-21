import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
from PIL import Image


st.set_page_config(page_title="Halaman Modelling", layout="wide")
st.write("""
# Welcome to my portofolio Data Analyst
Hello my name is [Bramantio](https://www.linkedin.com/in/brahmantio-w/) and I am passionate about uncovering insights through data. 
\n With a strong enthusiasm for data analysis, I enjoy transforming raw information into meaningful stories that drive better decision-making.

""")
add_selectitem = st.sidebar.selectbox("Want to open about?", ("House prediction","Palm oil classification","Supermarket Profit", "Heart disease"))

def house():
        st.set_page_config(page_title="Prediksi Rumah Surabaya", layout="centered", initial_sidebar_state="auto", page_icon="🏡")
        st.write("""
        This app predicts the **House Prediction at Surabaya**
        the process of conducting final research at the end of the lecture, 
        \nwhich uses the support vector regression algorithm with parameters based on previous research to obtain optimal accuracy, namely 0.98 or 98% 
        """)
        img = Image.open("rumah1.JPG")
        st.image(img, width=500)
        st.sidebar.header('User Input Features:')
    
        # Collects user input features into dataframe
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
        if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
        else:
                def user_input_features():
                    st.header("Input your specific data")
                    Cicilan= st.number_input("Cicilan perbulan",
                        min_value=0,
                        step=1000000,
                        )
                    Kecamatan = st.selectbox('Kecamatan rumah anda:', 
                                   ['Lakasantri', 'Mulyorejo', 'Kertajaya', 'Rungkut', 'Karang Pilang',
                                    'Wiyung', 'Sukolilo', 'Kenjeran', 'Tandes', 'Tegalsari',
                                    'Tenggilis Mejoyo', 'Gayungan', 'Dukuh Pakis', 'Sambikerep',
                                    'Jambangan', 'Gunung Anyar', 'Pabean', 'Bulak', 'Wonocolo',
                                    'Wonokromo', 'Sukomanunggal', 'Benowo', 'Semampir', 'Simokerto',
                                    'Pakal', 'Krembangan', 'Sawahan', 'Tambaksari', 'Genteng',
                                    'Asemrowo', 'Bubutan'])
                    Wilayah= st.selectbox('Pilih Wilayah:', ['Surabaya Barat', 'Surabaya Timur', 'Surabaya Selatan',
                            'Surabaya Utara', 'Surabaya Pusat'])
                    jenis_perumahan= st.radio("Jenis Pemukiman rumah",
                        ["Perumahan", "Perkampungan", "Samping Jalan"])
                    kamar_tidur= st.slider("Jumlah kamar tidur",
                        min_value=1,
                        max_value=10,
                        step=1,
                        value=1)
                    kamar_mandi= st.slider("Jumlah kamar mandi",
                        min_value=1,
                        max_value=10,
                        step=1,
                        value=1)
                    luas_tanah= st.slider("Luas tanah",
                        min_value=16,
                        max_value=5000,
                        step=1,
                        value=2)
                    luas_bangunan= st.slider("Luas bangunan",
                        min_value=16,
                        max_value=2000,
                        step=1,
                        value=2)
                    Carport= st.slider("Jumlah muat mobil dihalaman",
                        min_value=1,
                        max_value=10,
                        step=1,
                        value=1)
                    sertifikat= st.radio("Jenis kepemilikan sertifikat",['SHM', 'SHGB','SHP','SHSRS','PPJB', 'Lainnya'])
                    daya_listrik= st.slider("Daya listrik yang tersedia",
                        min_value=100,
                        max_value=66000,
                        step=100,
                        value=100)
                    jumlah_lantai= st.slider("jumlah lantai bangunan",
                        min_value=1,
                        max_value=5,
                        step=1,
                        value=1)
                    garasi= st.slider("Jumlah muat kendaraaan dalam garasi",
                        min_value=0,
                        max_value=10,
                        step=1,
                        value=0)
                    kondisi_properti= st.radio("Tingkat kondisi properti",["Baru","Bagus","Perlu perbaikan","Tidak layak"])
                    Dapur = st.slider("Jumlah dapur yang tersedia",
                        min_value=1,
                        max_value=4,
                        step=1,
                        value=1)
                    ruang_makan=st.radio("Ketersediaan ruang makan",["Tersedia","Tidak tersedia"])
                    ruang_tamu=st.radio("Ketersediaan ruang tamu",["Tersedia","Tidak tersedia"])
                    kondisi_perabotan=st.radio("Kondisi fungsional rumah",["Unfurnished","Semi furnished","furnished"])
                    material_bangunan=st.radio("material bangunan",["Batako","Bata Hebel","Bata Merah","Beton"])
                    material_lantai=st.radio("material lantai",["Granit","Keramik","Marmer","Ubin"])
                    hadap=st.radio("Arah rumah",["Barat","Timur","Utara","Selatan"])
                    konsep_rumah=st.selectbox('Konsep rumah', ['Modern Glass House', 'Modern', 'Scandinavian', 'Old', 'Mordern minimalist',
                                'Minimalist', 'American Classic', 'Classic','Kontemporer', 'Pavilion','Industrial'])
                    pemandangan=st.radio("Pemandangan sekitar",["Pemukiman Warga","Perkotaan","Taman Kota"])
                    jangkauan_internet=st.radio("Jangkauan Internet",["Tersedia","Tidak tersedia","Sedang Proses"])
                    lebar_jalan=st.slider("Lebar jalan memuat berapa kendaraan",
                        min_value=1,
                        max_value=4,
                        step=1,
                        value=1)
                    tahun_bangun=st.date_input('Tahun rumah dibangun')
                    tahun_renovasi=st.date_input('Tahun renovasi rumah')
                    fasilitas_perumahan=st.multiselect('Fasilitas yang dimiliki', ['Akses parkir','Masjid','Gereja','Taman','Keamanan','One gate system','Kolam renang','Laundry','CCTV'])
                    jarak_pusat_kota=st.radio("Berapa jauh jarak dari rumah ke pusat kota",["< 5 KM","5 KM","> 5KM"])
                    data = {'Cicilan':Cicilan,
                            'Kecamatan':Kecamatan,
                            'Wilayah':Wilayah,
                            'jenis_perumahan':jenis_perumahan,
                            'kamar_tidur':kamar_tidur,
                            'kamar_mandi':kamar_mandi,
                            'luas_tanah':luas_tanah,
                            'luas_bangunan':luas_bangunan,
                            'Carport':Carport,
                            'sertifikat':sertifikat,
                            'daya_listrik':daya_listrik,
                            'jumlah_lantai':jumlah_lantai,
                            'garasi':garasi,
                            'kondisi_properti':kondisi_properti,
                            'Dapur':Dapur,
                            'ruang_makan':ruang_makan,
                            'ruang_tamu':ruang_makan,
                            'kondisi_perabotan':kondisi_perabotan,
                            'material_bangunan':material_bangunan,
                            'material_lantai':material_lantai,
                            'hadap':hadap,
                            'konsep_rumah':konsep_rumah,
                            'pemandangan':pemandangan,
                            'jangkauan_internet':jangkauan_internet,
                            'lebar_jalan':lebar_jalan,
                            'sumber_air':'PDAM',        
                            'tahun_bangun':tahun_bangun,
                            'tahun_renovasi':tahun_renovasi,
                            'fasilitas_perumahan':fasilitas_perumahan,
                            'jarak_pusat_kota':jarak_pusat_kota
                            }
                    features = pd.DataFrame([data])
                    Kecamatan={'Lakasantri':0,'Mulyorejo':1,'Kertajaya':2,'Rungkut':3,'Karang Pilang':4,'Wiyung':5,'Sukolilo':6,
                               'Kenjeran':7,'Tandes':8,'Tegalsari':9,'Tenggilis Mejoyo':10,'Gayungan':11,'Dukuh Pakis':12,'Sambikerep':13,
                               'Jambangan':14,'Gunung Anyar':15,'Pabean':16,'Bulak':17,'Wonocolo':18,'Wonokromo':19,'Sukomanunggal':20,
                               'Benowo':21,'Semampir':22,'Simokerto':23,'Pakal':24,'Krembangan':25,'Sawahan':26,'Tambaksari':27,'Genteng':28,
                               'Asemrowo':29,'Bubutan':30}
                    Wilayah = {'Surabaya Barat':0, 'Surabaya Timur':1, 'Surabaya Selatan':2, 'Surabaya Utara':3}
                    jenis_perumahan = {'Perumahan':0,'Perkampungan':1,'Samping Jalan':2}
                    sertifikat = {'SHM':0,'SHGB':1,'SHP':2,'SHSRS':3,'PPJB':4,'Lainnya':5}
                    kondisi_properti = {'Baru': 0, 'Bagus': 1, 'Perlu perbaikan': 2, 'Tidak layak':3}
                    ruang_makan = {'Tersedia': 0, 'Tidak tersedia': 1}
                    ruang_tamu = {'Tersedia': 0, 'Tidak tersedia': 1}
                    kondisi_perabotan = {'Unfurnished': 0,'Semi furnished':1, 'furnished': 2}
                    material_bangunan = {'Batako':0, 'Bata Hebel': 1, 'Bata Merah': 2,'Beton': 3}
                    material_lantai = {'Granit': 0,'Keramik': 1, 'Marmer': 2, 'Ubin':3}
                    hadap = {'Barat': 0, 'Timur': 1, 'Utara': 2, 'Selatan':3}
                    konsep_rumah = {'Minimalist': 0, 'Kontemporer': 1, 'American Classic': 2,'Modern Glass House':3,'Mordern minimalist':4,'Scandinavian':5,'Pavilion':6,'Industrial':7,'Modern':8}
                    pemandangan = {'Pemukiman Warga': 0, 'Perkotaan': 1,'Taman Kota':2}
                    jangkauan_internet = {'Tersedia': 0, 'Tidak tersedia': 1, 'Sedang proses':2}
                    sumber_air = {'PDAM': 0, 'Air sumur':1, 'PAM': 2}
                    jarak_pusat_kota = {'< 5 KM': 0, '5 KM': 1, '> 5KM': 2}

                    features['Kecamatan'] = features['Kecamatan'].map(Kecamatan)
                    features['Wilayah'] = features['Wilayah'].map(Wilayah)
                    features['sertifikat'] = features['sertifikat'].map(sertifikat)
                    features['kondisi_properti'] = features['kondisi_properti'].map(kondisi_properti)
                    features['ruang_makan'] = features['ruang_makan'].map(ruang_makan)
                    features['ruang_tamu'] = features['ruang_tamu'].map(ruang_tamu)
                    features['kondisi_perabotan'] = features['kondisi_perabotan'].map(kondisi_perabotan)
                    features['material_bangunan'] = features['material_bangunan'].map(material_bangunan)
                    features['material_lantai'] = features['material_lantai'].map(material_lantai)
                    features['hadap'] = features['hadap'].map(hadap)
                    features['konsep_rumah'] = features['konsep_rumah'].map(konsep_rumah)
                    features['pemandangan'] = features['pemandangan'].map(pemandangan)
                    features['jangkauan_internet'] = features['jangkauan_internet'].map(jangkauan_internet)
                    features['sumber_air'] = features['sumber_air'].map(sumber_air)
                    features['jenis_perumahan'] = features['jenis_perumahan'].map(jenis_perumahan)
                    features['jarak_pusat_kota'] = features['jarak_pusat_kota'].map(jarak_pusat_kota)

                    features['tahun_bangun'] = pd.to_datetime(features['tahun_bangun'])
                    features['tahunbangunan'] = features['tahun_bangun'].dt.year
                    features['tahun_renovasi'] = pd.to_datetime(features['tahun_renovasi'])
                    features['tahunrenovasi'] = features['tahun_renovasi'].dt.year

                    features['jumlah_fasilitas'] = features['fasilitas_perumahan'].apply(lambda x: len(str(x).split(',')))

                    features['efisiensi_ruangan'] = features['luas_bangunan'] / features['luas_tanah']
                    features['kualitas_bangunan'] = (features['kondisi_properti'] + features['material_bangunan'] + features['material_lantai'] + features['konsep_rumah']) / 4
                    from datetime import datetime
                    tahun_sekarang = datetime.now().year
                    features['usia_bangunan'] = tahun_sekarang - features['tahunbangunan']
                    features['kualitas_infrastruktur'] = (features['sumber_air'] + features['jangkauan_internet'] + features['lebar_jalan'] + features['jarak_pusat_kota']) / 4
                    return features
                        
        input_df = user_input_features()
        if st.button('Predict Now!'):
                with open("tesis.pkl","rb") as file:
                        st.write(input_df)
                        model = pickle.load(file)
                        prediction1 = model.predict(input_df)
                        prediction = np.expm1(prediction1)
                with st.spinner('Wait for it...'):
                        time.sleep(4)
                        st.success(f"Hasil prediksi: Rp{prediction[0]:,.2f}")
                        
def palm():
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        st.set_page_config(page_title="PT.Kelapa Sawit", layout="centered", initial_sidebar_state="auto", page_icon="🌰")
        st.header("Implementasi Convolutional Neural Network untuk Identifikasi Tingkat Kematangan Buah Kelapa Sawit")
        st.write("Dataset yang digunakan berasal dari [kaggle](https://www.kaggle.com/datasets/ramadanizikri112/ripeness-of-oil-palm-fruit)")
        st.write("Submit gambar kelapa sawit yang anda miliki, kemudian model akan mengklasifikasikan gambar antara Belum Matang, Matang, atau Terlalu Matang")

        # Load model
        model = load_model("sawit_model.keras",compile=False)

        # Upload file
        uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
                    # tampilkan
                    img_pil = Image.open(uploaded_file).convert("RGB")
                    st.image(img_pil, caption="Gambar yang diunggah", use_column_width=True)

                    # resize & preprocess
                    img_resized = img_pil.resize((224, 224))
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    # Prediksi
                    preds = model.predict(img_array)
                    predicted_class = np.argmax(preds, axis=1)
    
                    # Mapping kelas ke label
                    label_mapping = {
                        0: "Belum Matang",
                        1: "Matang",
                        2: "Terlalu Matang"
                        }
                    label_prediksi = label_mapping[predicted_class[0]]

                    st.write(f"Hasil Prediksi: {label_prediksi}")


def profit():
        # Muat daftar fitur yang telah disimpan
        with open('selected_features.pkl', 'rb') as file:
                selected_features = pickle.load(file)

        # Muat model yang telah dilatih
        with open('supermarket.pkl', 'rb') as file:
                model = pickle.load(file)

        # Judul aplikasi
        st.set_page_config(page_title="PT.Supermarket Sejahtera", layout="centered", initial_sidebar_state="auto", page_icon="🏣")
        st.title("""
        Prediksi Profit Supermarket
        \n Dataset yang digunakan berasal dari [Kaggle](https://www.kaggle.com/datasets/roopacalistus/superstore)
        """)

        # Formulir untuk input data dari pengguna
        st.subheader('Masukkan detail produk:')
        Ship_mode = st.radio('Ship Mode', ['Standard Class', 'Second Class', 'First Class', 'Same Day'])
        Segment = st.radio('Segment', ['Consumer','Corporate','Home Office'])
        Country = st.radio('Country', ['United States'])
        City = st.selectbox('City',['Henderson', 'Los Angeles', 'Fort Lauderdale', 'Concord',
                                    'Seattle','Fort Worth', 'Madison', 'West Jordan', 'San Francisco',
                                    'Fremont', 'Philadelphia', 'Orem', 'Houston', 'Richardson',
                                    'Naperville', 'Melbourne', 'Eagan', 'Westland', 'Dover',
                                    'New Albany', 'New York City', 'Troy', 'Chicago', 'Gilbert',
                                    'Springfield', 'Jackson', 'Memphis', 'Decatur', 'Durham',
                                    'Columbia', 'Rochester', 'Minneapolis', 'Portland', 'Saint Paul',
                                    'Aurora', 'Charlotte', 'Orland Park', 'Urbandale', 'Columbus',
                                    'Bristol', 'Wilmington', 'Bloomington', 'Phoenix', 'Roseville',
                                    'Independence', 'Pasadena', 'Newark', 'Franklin', 'Scottsdale',
                                    'San Jose', 'Edmond', 'Carlsbad', 'San Antonio', 'Monroe',
                                    'Fairfield', 'Grand Prairie', 'Redlands', 'Hamilton', 'Westfield',
                                    'Akron', 'Denver', 'Dallas', 'Whittier', 'Saginaw', 'Medina',
                                    'Dublin', 'Detroit', 'Tampa', 'Santa Clara', 'Lakeville',
                                    'San Diego', 'Brentwood', 'Chapel Hill', 'Morristown',
                                    'Cincinnati', 'Inglewood', 'Tamarac', 'Colorado Springs',
                                    'Belleville', 'Taylor', 'Lakewood', 'Arlington', 'Arvada',
                                    'Hackensack', 'Saint Petersburg', 'Long Beach', 'Hesperia',
                                    'Murfreesboro', 'Layton', 'Austin', 'Lowell', 'Manchester',
                                    'Harlingen', 'Tucson', 'Quincy', 'Pembroke Pines', 'Des Moines',
                                    'Peoria', 'Las Vegas', 'Warwick', 'Miami', 'Huntington Beach',
                                    'Richmond', 'Louisville', 'Lawrence', 'Canton', 'New Rochelle',
                                    'Gastonia', 'Jacksonville', 'Auburn', 'Norman', 'Park Ridge',
                                    'Amarillo', 'Lindenhurst', 'Huntsville', 'Fayetteville',
                                    'Costa Mesa', 'Parker', 'Atlanta', 'Gladstone', 'Great Falls',
                                    'Lakeland', 'Montgomery', 'Mesa', 'Green Bay', 'Anaheim',
                                    'Marysville', 'Salem', 'Laredo', 'Grove City', 'Dearborn',
                                    'Warner Robins', 'Vallejo', 'Mission Viejo', 'Rochester Hills',
                                    'Plainfield', 'Sierra Vista', 'Vancouver', 'Cleveland', 'Tyler',
                                    'Burlington', 'Waynesboro', 'Chester', 'Cary', 'Palm Coast',
                                    'Mount Vernon', 'Hialeah', 'Oceanside', 'Evanston', 'Trenton',
                                    'Cottage Grove', 'Bossier City', 'Lancaster', 'Asheville',
                                    'Lake Elsinore', 'Omaha', 'Edmonds', 'Santa Ana', 'Milwaukee',
                                    'Florence', 'Lorain', 'Linden', 'Salinas', 'New Brunswick',
                                    'Garland', 'Norwich', 'Alexandria', 'Toledo', 'Farmington',
                                    'Riverside', 'Torrance', 'Round Rock', 'Boca Raton',
                                    'Virginia Beach', 'Murrieta', 'Olympia', 'Washington',
                                    'Jefferson City', 'Saint Peters', 'Rockford', 'Brownsville',
                                    'Yonkers', 'Oakland', 'Clinton', 'Encinitas', 'Roswell',
                                    'Jonesboro', 'Antioch', 'Homestead', 'La Porte', 'Lansing',
                                    'Cuyahoga Falls', 'Reno', 'Harrisonburg', 'Escondido', 'Royal Oak',
                                    'Rockville', 'Coral Springs', 'Buffalo', 'Boynton Beach',
                                    'Gulfport', 'Fresno', 'Greenville', 'Macon', 'Cedar Rapids',
                                    'Providence', 'Pueblo', 'Deltona', 'Murray', 'Middletown',
                                    'Freeport', 'Pico Rivera', 'Provo', 'Pleasant Grove', 'Smyrna',
                                    'Parma', 'Mobile', 'New Bedford', 'Irving', 'Vineland', 'Glendale',
                                    'Niagara Falls', 'Thomasville', 'Westminster', 'Coppell', 'Pomona',
                                    'North Las Vegas', 'Allentown', 'Tempe', 'Laguna Niguel',
                                    'Bridgeton', 'Everett', 'Watertown', 'Appleton', 'Bellevue',
                                    'Allen', 'El Paso', 'Grapevine', 'Carrollton', 'Kent', 'Lafayette',
                                    'Tigard', 'Skokie', 'Plano', 'Suffolk', 'Indianapolis', 'Bayonne',
                                    'Greensboro', 'Baltimore', 'Kenosha', 'Olathe', 'Tulsa', 'Redmond',
                                    'Raleigh', 'Muskogee', 'Meriden', 'Bowling Green', 'South Bend',
                                    'Spokane', 'Keller', 'Port Orange', 'Medford', 'Charlottesville',
                                    'Missoula', 'Apopka', 'Reading', 'Broomfield', 'Paterson',
                                    'Oklahoma City', 'Chesapeake', 'Lubbock', 'Johnson City',
                                    'San Bernardino', 'Leominster', 'Bozeman', 'Perth Amboy',
                                    'Ontario', 'Rancho Cucamonga', 'Moorhead', 'Mesquite', 'Stockton',
                                    'Ormond Beach', 'Sunnyvale', 'York', 'College Station',
                                    'Saint Louis', 'Manteca', 'San Angelo', 'Salt Lake City',
                                    'Knoxville', 'Little Rock', 'Lincoln Park', 'Marion', 'Littleton',
                                    'Bangor', 'Southaven', 'New Castle', 'Midland', 'Sioux Falls',
                                    'Fort Collins', 'Clarksville', 'Sacramento', 'Thousand Oaks',
                                    'Malden', 'Holyoke', 'Albuquerque', 'Sparks', 'Coachella',
                                    'Elmhurst', 'Passaic', 'North Charleston', 'Newport News',
                                    'Jamestown', 'Mishawaka', 'La Quinta', 'Tallahassee', 'Nashville',
                                    'Bellingham', 'Woodstock', 'Haltom City', 'Wheeling',
                                    'Summerville', 'Hot Springs', 'Englewood', 'Las Cruces', 'Hoover',
                                    'Frisco', 'Vacaville', 'Waukesha', 'Bakersfield', 'Pompano Beach',
                                    'Corpus Christi', 'Redondo Beach', 'Orlando', 'Orange',
                                    'Lake Charles', 'Highland Park', 'Hempstead', 'Noblesville',
                                    'Apple Valley', 'Mount Pleasant', 'Sterling Heights', 'Eau Claire',
                                    'Pharr', 'Billings', 'Gresham', 'Chattanooga', 'Meridian',
                                    'Bolingbrook', 'Maple Grove', 'Woodland', 'Missouri City',
                                    'Pearland', 'San Mateo', 'Grand Rapids', 'Visalia',
                                    'Overland Park', 'Temecula', 'Yucaipa', 'Revere', 'Conroe',
                                    'Tinley Park', 'Dubuque', 'Dearborn Heights', 'Santa Fe',
                                    'Hickory', 'Carol Stream', 'Saint Cloud', 'North Miami',
                                    'Plantation', 'Port Saint Lucie', 'Rock Hill', 'Odessa',
                                    'West Allis', 'Chula Vista', 'Manhattan', 'Altoona', 'Thornton',
                                    'Champaign', 'Texarkana', 'Edinburg', 'Baytown', 'Greenwood',
                                    'Woonsocket', 'Superior', 'Bedford', 'Covington', 'Broken Arrow',
                                    'Miramar', 'Hollywood', 'Deer Park', 'Wichita', 'Mcallen',
                                    'Iowa City', 'Boise', 'Cranston', 'Port Arthur', 'Citrus Heights',
                                    'The Colony', 'Daytona Beach', 'Bullhead City', 'Portage', 'Fargo',
                                    'Elkhart', 'San Gabriel', 'Margate', 'Sandy Springs', 'Mentor',
                                    'Lawton', 'Hampton', 'Rome', 'La Crosse', 'Lewiston',
                                    'Hattiesburg', 'Danville', 'Logan', 'Waterbury', 'Athens',
                                    'Avondale', 'Marietta', 'Yuma', 'Wausau', 'Pasco', 'Oak Park',
                                    'Pensacola', 'League City', 'Gaithersburg', 'Lehi', 'Tuscaloosa',
                                    'Moreno Valley', 'Georgetown', 'Loveland', 'Chandler', 'Helena',
                                    'Kirkwood', 'Waco', 'Frankfort', 'Bethlehem', 'Grand Island',
                                    'Woodbury', 'Rogers', 'Clovis', 'Jupiter', 'Santa Barbara',
                                    'Cedar Hill', 'Norfolk', 'Draper', 'Ann Arbor', 'La Mesa',
                                    'Pocatello', 'Holland', 'Milford', 'Buffalo Grove', 'Lake Forest',
                                    'Redding', 'Chico', 'Utica', 'Conway', 'Cheyenne', 'Owensboro',
                                    'Caldwell', 'Kenner', 'Nashua', 'Bartlett', 'Redwood City',
                                    'Lebanon', 'Santa Maria', 'Des Plaines', 'Longview',
                                    'Hendersonville', 'Waterloo', 'Cambridge', 'Palatine', 'Beverly',
                                    'Eugene', 'Oxnard', 'Renton', 'Glenview', 'Delray Beach',
                                    'Commerce City', 'Texas City', 'Wilson', 'Rio Rancho', 'Goldsboro',
                                    'Montebello', 'El Cajon', 'Beaumont', 'West Palm Beach', 'Abilene',
                                    'Normal', 'Saint Charles', 'Camarillo', 'Hillsboro', 'Burbank',
                                    'Modesto', 'Garden City', 'Atlantic City', 'Longmont', 'Davis',
                                    'Morgan Hill', 'Clifton', 'Sheboygan', 'East Point', 'Rapid City',
                                    'Andover', 'Kissimmee', 'Shelton', 'Danbury', 'Sanford',
                                    'San Marcos', 'Greeley', 'Mansfield', 'Elyria', 'Twin Falls',
                                    'Coral Gables', 'Romeoville', 'Marlborough', 'Laurel', 'Bryan',
                                    'Pine Bluff', 'Aberdeen', 'Hagerstown', 'East Orange',
                                    'Arlington Heights', 'Oswego', 'Coon Rapids', 'San Clemente',
                                    'San Luis Obispo', 'Springdale', 'Lodi', 'Mason'])
        State = st.selectbox('State',['Kentucky', 'California', 'Florida', 'North Carolina',
                'Washington', 'Texas', 'Wisconsin', 'Utah', 'Nebraska',
                'Pennsylvania', 'Illinois', 'Minnesota', 'Michigan', 'Delaware',
                'Indiana', 'New York', 'Arizona', 'Virginia', 'Tennessee',
                'Alabama', 'South Carolina', 'Oregon', 'Colorado', 'Iowa', 'Ohio',
                'Missouri', 'Oklahoma', 'New Mexico', 'Louisiana', 'Connecticut',
                'New Jersey', 'Massachusetts', 'Georgia', 'Nevada', 'Rhode Island',
                'Mississippi', 'Arkansas', 'Montana', 'New Hampshire', 'Maryland',
                'District of Columbia', 'Kansas', 'Vermont', 'Maine',
                'South Dakota', 'Idaho', 'North Dakota', 'Wyoming',
                'West Virginia'])
        Postal_code = st.number_input('Postal_code',
                                   min_value=0,
                                   step=1,)
        Region = st.radio('Region',['South', 'West', 'Central', 'East'])
        Category = st.radio('Category',['Furniture', 'Office Supplies', 'Technology'])
        Sub_Category = st.selectbox('Sub_Category',['Bookcases', 'Chairs', 'Labels', 'Tables', 'Storage',
                                    'Furnishings', 'Art', 'Phones', 'Binders', 'Appliances', 'Paper',
                                    'Accessories', 'Envelopes', 'Fasteners', 'Supplies', 'Machines',
                                    'Copiers'])
        Sales = st.number_input('Sales', value=100.0)
        Quantity = st.number_input('Quantity', value=1, min_value=1)
        Discount = st.number_input('Discount', value=0.0, min_value=0.0)

        if st.button('Prediksi Profit'):
                # Kumpulkan data input
                input_data = {
                        'Ship_mode': Ship_mode,
                        'Segment': Segment,
                        'Country': Country,
                        'City':City,
                        'State':State,
                        'Postal_code':Postal_code,
                        'Region':Region,
                        'Category':Category,
                        'Sub_Category':Sub_Category,
                        'Sales': Sales,
                        'Quantity': Quantity,
                        'Discount': Discount,
                            }
    
                features = pd.DataFrame([input_data])
    
                # Lakukan semua langkah preprocessing yang sama seperti saat pelatihan
                #feature combination
                features['revenue_item'] = features['Sales'] / features['Quantity']

                #log transform
                features['Sales'] = np.log1p(features['Sales'])
                features['Postal_code'] = np.log1p(features['Postal_code'])
                features['revenue_item'] = np.log1p(features['revenue_item'])

                # encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                features['Ship_mode_emb'] = le.fit_transform(features['Ship_mode'])
                features['Segment_emb'] = le.fit_transform(features['Segment'])
                features['Country_emb'] = le.fit_transform(features['Country'])
                features['City_emb'] = le.fit_transform(features['City'])
                features['State_emb'] = le.fit_transform(features['State'])
                features['Region_emb'] = le.fit_transform(features['Region'])
                features['Category_emb'] = le.fit_transform(features['Category'])
                features['Sub_Category_emb'] = le.fit_transform(features['Sub_Category'])

                # Pilih fitur-fitur yang sudah diseleksi
                features = features[selected_features]
    
                # Lakukan prediksi
                prediction_log = model.predict(features)
                prediction = np.expm1(prediction_log)[0]
    
                # Tampilkan hasil
                st.subheader('Hasil Prediksi:')
                if prediction > 0:
                        st.success(f"Profit yang diprediksi: **${prediction:,.2f}** (Untung)")
                else:
                        st.error(f"Profit yang diprediksi: **${prediction:,.2f}** (Rugi)")
                
            
def heart():
    st.set_page_config(page_title="Penyakit Jantung", layout="centered", initial_sidebar_state="auto", page_icon="🫀")
    st.write("""
    This app predicts the **Heart Disease**
    
    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    """)
    st.sidebar.header('User Input Features:')
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features2():
            st.sidebar.header('Manual Input')
            age = st.sidebar.slider("Usia", 29, 77, 30)
            sex = st.sidebar.selectbox("Jenis Kelamin", ('Perempuan', 'Pria'))
            if sex == "Perempuan":
                sex = 0
            else:
                sex = 1 
            cp = st.sidebar.slider('Chest pain type', 1,4,2)
            if cp == 1.0:
                wcp = "Nyeri dada tipe angina"
            elif cp == 2.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil"
            elif cp == 3.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
            else:
                wcp = "Nyeri dada yang tidak terkait dengan masalah jantung"
            st.sidebar.write("Jenis nyeri dada yang dirasakan oleh pasien", wcp)
            trestbps = st.sidebar.slider ("Tingkat BPS Pasien", 94, 200, 94)
            chol = st.sidebar.slider("Tingkat Kolesterol Pasien", 126, 564, 126)
            fbs = st.sidebar.slider("Tingkat Kadar gula pasien", 0, 1, 0)
            restecg = st.sidebar.slider("EKG Pasien Saat Istirahat", 0, 2, 0)
            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
            oldpeak = st.sidebar.slider("Seberapa banyak ST segmen menurun atau depresi", 0.0, 6.2, 1.0)
            slope = st.sidebar.slider("Kemiringan segmen ST pada elektrokardiogram (EKG)", 0, 2, 1)
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Hasil tes thalium", 1, 3, 1)
            data = {'age':age,
                    'sex': sex,
                    'cp': cp,
                    'trestbps':trestbps,
                    'chol':chol,
                    'fbs':fbs,
                    'restecg':restecg,
                    'thalach': thalach,
                    'exang': exang,
                    'oldpeak': oldpeak,
                    'slope': slope,
                    'ca':ca,
                    'thal':thal,
                   }
            features = pd.DataFrame(data, index=[0])
            return features
    
    input_df = user_input_features2()
    img = Image.open("heart-disease.jpg")
    st.image(img, width=500)
    if st.sidebar.button('Predict!'):
        with open("modelheart.pkl", 'rb') as file:
            st.write(input_df)
            model = pickle.load(file)
            prediction = model.predict(input_df)        
            result = ['No Heart Disease' if prediction == 0 else 'Yes Heart Disease']
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")
         
if add_selectitem == "House prediction":
    house()
elif add_selectitem == "Palm oil classification":
    palm()
elif add_selectitem == "Supermarket Profit":
    profit()
elif add_selectitem == "Heart disease":
    heart()
