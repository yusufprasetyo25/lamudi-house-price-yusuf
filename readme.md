# Jakarta House Price Prediction from [Lamudi](https://www.lamudi.co.id/) Price

**TL;DR**

I demonstrated how to create an web application to predict house prices in Jakarta using only quantitative measurements (e.g. land sizes, building sizes, street location) that will prevent any bias and easily usable by everyone. I created this project to highlight how we can:
1. Answer [real problem](https://www.cnnindonesia.com/ekonomi/20220713182830-92-821146/sri-mulyani-sebut-milenial-sulit-beli-rumah-apa-peran-pemerintah). Sri Mulyani, minister of finance in Indonesia, said that millenials will have a hard time buying a house. So, knowing fair prices will definitely help millenials to manage their expectations and plan their house expenses
2. Extract data from external sources creatively (with various formats available on the internet) to create features for the machine learning model
3. Create simple and easy-to-use interactive machine-learning-powered application using Python to calculate fair house prices

Please try using the application and give feedback to me, especially if you are in Jakarta!

**Important Notes**:

- I used data from [Lamudi](https://www.lamudi.co.id/), definitely check it out if you are looking to buy any property.
- You can trace my workflow from Git associated in this folder. However, there are lots of unexpected things happened during the real project (e.g. parsed streets have wrong _kelurahan_ name, so the actual process was cyclical and quite uneasy to follow). Therefore, I created `consolidated.ipynb` that captures the workflow from start to end. *Please note that I didn't really used that notebook to make the whole project, so please inform me if you find any error while running the code. Thank you!

**Key Features**:

1. Scrapes real house price data from [Lamudi](https://www.lamudi.co.id/)
2. Uses governmental Nilai Jual Objek Pajak (NJOP) from [Pergub Nomor 17 Tahun 2021 Tentang Penetapan NJOP PBB-P2 Tahun 2021](https://bprd.jakarta.go.id/peraturan-perpajakan/unduh/pergub-nomor-17-tahun-2021-tentang-penetapan-njop-pbbp2-tahun-2021) and parse the PDF to extract useful feature
3. Uses open source [Nominatim](https://nominatim.org/) with OpenStreetMap data to get the street coordinate
4. Uses Jakarta flood scenario from [Governmental Dashboard](https://public.tableau.com/app/profile/jsc.data/viz/DataPendukungPotensiGenangan/PetaAwal) as feature
5. Scrapes facilities and ratings from [Google Maps](https://www.google.com/maps) as feature
6. Uses Python from start-to-end and [streamlit](https://streamlit.io/) to deploy the application
7. Uses Mean Absolute Percentage Error (MAPE) as error metric to simulate real situation (more expensive houses allow for more prediction error)
8. Outputs near facilities in the web application

**Technical Notes**:

- Negative MAPE Train: -0.22285646275637008
- Negative MAPE Test: -0.23603570643641283

**Possible improvements**:
1. We didn't try to use different radius for every facility. For better result, we may try to improve the model from the feature engineering step by choosing different radius for every facility
2. Prediction quality is not quite good for houses in alleys, we can improve it if we have databases of street width. Related to number 1, we can also hypothesize that number of neighboring streets in smaller radius can capture the "alleyness" of a street, i.e., small streets should have many near neighbors
3. I just realized that there were many typos, even in parsed NJOP streets. We could recreate the model after cleaning the NJOP streets
4. We can enrich the model by using more scraped features from the internet. Some examples are:
    - distance to inner ring road gate (hypothesis: the nearer, the higher price)
    - distance to outer ring road gate (hypothesis: the nearer, the higher price)
    - office buildings facility (hypothesis: the more and better from ratings, the higher price)