#!/usr/bin/env python
# coding: utf-8

# # REGRESION LINEAL
# ## INTRODUCCION
# 
# La regresión lineal es un método estadístico que trata de modelar la relación entre una variable continua y una o más variables independientes mediante el ajuste de una ecuación lineal. Se llama regresión lineal simple cuando solo hay una variable independiente y regresión lineal múltiple cuando hay más de una. Dependiendo del contexto, a la variable modelada se le conoce como variable dependiente o variable respuesta, y a las variables independientes como regresores, predictores.
# 

# ## BREVE HISTORIA
# 
# La primera forma de regresión lineal documentada fue el método de los mínimos cuadrados que fue publicada por Legendre en 1805, Gauss publicó un trabajo en donde desarrollaba de manera más profunda el método de los mínimos cuadrados,1​ y en dónde se incluía una versión del teorema de Gauss-Márkov.
# 
# El término regresión se utilizó por primera vez en el estudio de variables antropométricas: al comparar la estatura de padres e hijos, donde resultó que los hijos cuyos padres tenían una estatura muy superior al valor medio, tendían a igualarse a este, mientras que aquellos cuyos padres eran muy bajos tendían a reducir su diferencia respecto a la estatura media; es decir, "regresaban" al promedio.2​ La constatación empírica de esta propiedad se vio reforzada más tarde con la justificación teórica de ese fenómeno.
# 
# El término lineal se emplea para distinguirlo del resto de técnicas de regresión, que emplean modelos basados en cualquier clase de función matemática. Los modelos lineales son una explicación simplificada de la realidad, mucho más ágiles y con un soporte teórico mucho más extenso por parte de la matemática y la estadística.
# 
# Pero bien, como se ha dicho, se puede usar el término lineal para distinguir modelos basados en cualquier clase de aplicación.(WIKIPEDIA)
# I[IMAGEN](https://es.wikipedia.org/wiki/Regresi%C3%B3n_lineal#:~:text=La%20primera%20forma%20de%20regresi%C3%B3n,del%20teorema%20de%20Gauss%2DM%C3%A1rkov.)
# 
# 

# ## plantamiento del algoritmo
# 
# El modelo de regresión lineal (Legendre, Gauss, Galton y Pearson) considera que, dado un conjunto de observaciones  {yi,xi1,...,xnp}ni=1 , la media  μ  de la variable respuesta  y  se relaciona de forma lineal con la o las variables regresoras  x1  ...  xp  acorde a la ecuación:
# 
# μy=β0+β1x1+β2x2+...+βpxp
#  
# El resultado de esta ecuación se conoce como la línea de regresión poblacional, y recoge la relación entre los predictores y la media de la variable respuesta.
# 
# Otra definición que se encuentra con frecuencia en los libros de estadística es:
# 
# yi=β0+β1xi1+β2xi2+...+βpxip+ϵi
#  
# En este caso, se está haciendo referencia al valor de  y  para una observación  i  concreta. El valor de una observación puntual nunca va a ser exactamente igual al promedio, de ahí que se añada el término de error  ϵ .
# 
# En ambos casos, la interpretación de los elementos del modelo es la misma:
# 
# β0 : es la ordenada en el origen, se corresponde con el valor promedio de la variable respuesta  y  cuando todos los predictores son cero.
# 
# βj : es el efecto promedio que tiene sobre la variable respuesta el incremento en una unidad de la variable predictora  xj , manteniéndose constantes el resto de variables. Se conocen como coeficientes parciales de regresión.
# 
# e : es el residuo o error, la diferencia entre el valor observado y el estimado por el modelo. Recoge el efecto de todas aquellas variables que influyen en  y  pero que no se incluyen en el modelo como predictores.
# 
# En la gran mayoría de casos, los valores  β0  y  βj  poblacionales se desconocen, por lo que, a partir de una muestra, se obtienen sus estimaciones  β^0  y  β^j . Ajustar el modelo consiste en estimar, a partir de los datos disponibles, los valores de los coeficientes de regresión que maximizan la verosimilitud (likelihood), es decir, los que dan lugar al modelo que con mayor probabilidad puede haber generado los datos observados.
# 
# El método empleado con más frecuencia es el ajuste por mínimos cuadrados ordinarios (OLS), que identifica como mejor modelo la recta (o plano si es regresión múltiple) que minimiza la suma de las desviaciones verticales entre cada dato de entrenamiento y la recta, elevadas al cuadrado.

# El modelo de regresión lineal (Legendre, Gauss, Galton y Pearson) considera que, dado un conjunto de observaciones  {yi,xi1,...,xnp}ni=1 , la media  μ  de la variable respuesta  y  se relaciona de forma lineal con la o las variables regresoras  x1  ...  xp  acorde a la ecuación:
# 
# μy=β0+β1x1+β2x2+...+βpxp
#  
# El resultado de esta ecuación se conoce como la línea de regresión poblacional, y recoge la relación entre los predictores y la media de la variable respuesta.
# 
# Otra definición que se encuentra con frecuencia en los libros de estadística es:
# 
# yi=β0+β1xi1+β2xi2+...+βpxip+ϵi
#  
# En este caso, se está haciendo referencia al valor de  y  para una observación  i  concreta. El valor de una observación puntual nunca va a ser exactamente igual al promedio, de ahí que se añada el término de error  ϵ .
# 
# En ambos casos, la interpretación de los elementos del modelo es la misma:
# 
# β0 : es la ordenada en el origen, se corresponde con el valor promedio de la variable respuesta  y  cuando todos los predictores son cero.
# 
# βj : es el efecto promedio que tiene sobre la variable respuesta el incremento en una unidad de la variable predictora  xj , manteniéndose constantes el resto de variables. Se conocen como coeficientes parciales de regresión.
# 
# e : es el residuo o error, la diferencia entre el valor observado y el estimado por el modelo. Recoge el efecto de todas aquellas variables que influyen en  y  pero que no se incluyen en el modelo como predictores.
# 
# En la gran mayoría de casos, los valores  β0  y  βj  poblacionales se desconocen, por lo que, a partir de una muestra, se obtienen sus estimaciones  β^0  y  β^j . Ajustar el modelo consiste en estimar, a partir de los datos disponibles, los valores de los coeficientes de regresión que maximizan la verosimilitud (likelihood), es decir, los que dan lugar al modelo que con mayor probabilidad puede haber generado los datos observados.
# 
# El método empleado con más frecuencia es el ajuste por mínimos cuadrados ordinarios (OLS), que identifica como mejor modelo la recta (o plano si es regresión múltiple) que minimiza la suma de las desviaciones verticales entre cada dato de entrenamiento y la recta, elevadas al cuadrado.
# 

# ## Algoritmo

# 
# 

# In[ ]:





# In[12]:


# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[ ]:




