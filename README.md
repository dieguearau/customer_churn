
# Predicción de Tasa de Abandono (Churn) de Clientes

<img src="Fig/Churn_rate_img.png" alt="Churn Rate" width="70%"/>

## Introducción

En este proyecto, se aborda el problema de la **tasa de abandono de clientes** (churn) en una empresa. El churn se refiere a la cantidad de clientes que dejan de utilizar los servicios de una compañía en un período determinado. Reducir la tasa de churn es crucial para cualquier empresa, ya que retener clientes es generalmente más rentable que adquirir nuevos.

El objetivo principal de este proyecto es identificar el modelo de machine learning que mejor se ajuste a los datos y ofrezca la mayor capacidad predictiva para detectar a los clientes con mayor probabilidad de darse de baja del servicio. Esto permitirá implementar estrategias de retención más efectivas y, en consecuencia, mejorar la retención de clientes y la sostenibilidad a largo plazo de la empresa.

## Dataset

Para este análisis, se ha utilizado el conjunto de datos de [Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset). Este dataset contiene información sobre varios aspectos del comportamiento del cliente, como su demografía, historial de compra, y otros factores relevantes que podrían influir en su decisión de abandonar el servicio.

## Metodología

El proyecto se encuentra en el archivo [Customer_Churn.ipynb](Customer_Churn.ipynb) y sigue una metodología estructurada que incluye las siguientes etapas:

1. **Exploración y preprocesamiento de datos**:
   - Análisis descriptivo
   - Análisis de dispersión y gráficos exploratorios
   - Análisis de correlación

2. **Modelado y Evaluación**:
   - Aplicación de varios modelos de machine learning como Random Forest, Redes neuronales y XGBoost, entre otros.
   - Evaluación de los modelos utilizando métricas como precisión, recall, F1-score.

3. **Optimización de Hiperparámetros**:
   - Uso de framework de Optuna para la búsqueda de hiperparámetros.

4. **Implementación de Estrategias de Retención**:
   - Basado en los resultados del modelo, se proponen estrategias específicas para mejorar la retención de clientes.



Este proyecto proporciona una guía completa para la implementación de un modelo de machine learning orientado a la predicción de churn. La combinación de técnicas de preprocesamiento, modelado, y optimización asegura que el modelo final tenga una alta capacidad predictiva, lo que es esencial para cualquier estrategia de retención de clientes.

---

