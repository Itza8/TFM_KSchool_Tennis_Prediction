# Tennis Predictions

# Descargar los partidos (>25MB Github no permite subir)

Fichero all_matches.csv contenido en -> [Tennis Large Dataset](https://www.kaggle.com/ehallmar/a-large-tennis-dataset-for-atp-and-itf-betting#all_matches.csv)

## Crear entorno de conda con la orden `conda env create -f environment.yml`

Si solo se desea ejecutar la interfaz web es posible ya que en el repo están guardados los csv y los pkl con lo necesario.

## Ejecutar los notebooks en orden (Hello_Dataset es solo un primer contacto):

1. Data_Cleaning
2. Exploratory_Data_Analysis
3. Clustering
4. Modelo_supervisado

## Por último ya podremos ejecutar la interfaz web:

`python app.py`
