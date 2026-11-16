# Las cinco preguntas del informe final

Para cada respuesta, expliquen qué decidieron y respáldenlo con un número medido en el proyecto.

## 1. ¿Su modelo agrega valor?

Pongan en una tabla el MAE de persistencia, media móvil de siete días y del modelo final sobre el mismo Test. Si la mejora sobre el mejor baseline queda por debajo del 10 %, expliquen qué ganan con la complejidad adicional y si vale la pena mantenerla.

**Respuesta**

Sí, aunque la mejora sobre el mejor baseline queda por debajo del 10 %. Test: 165 días, del 27/04/2023 al 24/10/2023 (`results/evaluacion_test_v1.0.0.json`).

| Método | MAE Test (kW) | RMSE (kW) | R² |
|---|---:|---:|---:|
| Persistencia | 3.369 | 4.410 | 0,40 |
| Media móvil 7 días | 3.033 | 3.889 | 0,54 |
| **Ridge v1.0.0 (10 variables)** | **2.773** | **3.651** | **0,59** |

La mejora sobre la media móvil es de 260 kW por día, un 8,6 %. Lo que ganamos con la complejidad:

- **Cumplir el objetivo.** La media móvil queda en 3.033 kW, por encima del límite de 3.000; el modelo queda 227 kW por debajo.
- **Una ventaja que se repite en tres períodos distintos.** El modelo le gana a la media móvil en la validación cruzada temporal dentro de Train (3.252 contra 3.564 kW), en Validation (2.265 contra 2.484) y en Test (2.773 contra 3.033). Que la diferencia aparezca en los tres tramos indica que no es ruido de un período en particular.

El costo es mantener el pipeline de preprocesamiento y que `predict.py` necesite 6 columnas del Excel, contra una sola de la media móvil. Lo consideramos justificado porque sin el modelo no se alcanza el límite de 3.000 kW.

Limitaciones medidas: el MAE equivale al 11,6 % del consumo medio del período (23.894 kW); el error mediano es menor (2.091 kW), pero el 38 % de los días supera los 3.000 kW de error; agosto es el mes con más error (4.191 kW), y el modelo no incluye variables de clima.

## 2. ¿Qué variables se parecen demasiado al objetivo?

Busquen variables con correlación superior a 0,98 con Frio (kW) del mismo día. Digan si decidieron usarlas y expliquen cómo comprobaron cuándo está disponible cada medición. Justifiquen qué datos del día D usan para predecir D+1.

**Respuesta**

Con nuestros datos no encontramos ninguna. Calculamos la correlación de todas las columnas de todas las hojas, en el cierre diario de las 23:59 y sin los desbordes de contador, contra `Frio (Kw) (Consolidado EE)` del mismo día. La más alta es `Servicios (Kw) (Consolidado EE)`, con 0,942, seguida por `Agua Planta (Hl)` (0,793) y `KW Trafo 10` (0,776). En el EDA (sección «Variables casi idénticas al Frío del mismo día») también da cero columnas por encima de 0,98, tanto contra el Frío del día como contra el target.

`Servicios` es la más parecida porque agrupa equipos del sistema de frío. No entra al modelo: en Train correlaciona 0,942 con el Frío, y el filtro de redundancia (|r| ≥ 0,90 en Train) se queda solo con la variable de cada grupo más correlacionada con el target. En este caso gana el Frío del día (0,847 contra 0,807 de `Servicios`).

Usar datos del día D no es fuga de información: el cierre de las 23:59 del día D ya se conoce cuando empieza D+1. Lo comprobamos de cuatro maneras:

1. El target se construye por fecha de calendario: la etiqueta del día D es el cierre de Frío de D+1, buscado con `reindex(D + 1 día)` sobre una grilla diaria continua. Si falta el día siguiente, la etiqueta queda vacía en lugar de correrse a otra fecha.
2. Los rezagos se calculan sobre esa grilla de calendario (`shift(1)` y `shift(7)`) y la media móvil usa `closed='left'`, que solo incluye días anteriores a D.
3. `preprocesamiento.ipynb` verifica, antes de exportar, que cada etiqueta coincida con el cierre de Frío de D+1, y corta con un error si alguna no coincide.
4. Ninguna columna del día D tiene correlación mayor a 0,98 con el target. La variable más parecida al target es el propio Frío del día (0,843), lo esperable en una serie con mucha inercia.

## 3. ¿Qué resultado contradice la intuición?

Comparen la hipótesis de un patrón semanal con lo que midieron en el EDA. Si las dos cosas difieren, den el número, expliquen qué pudo haber borrado el patrón y cuenten qué decidieron hacer.

**Respuesta**

La intuición era que la planta consume menos frío los fines de semana. El EDA no lo confirma:

| Medición | Resultado |
|---|---|
| Kruskal-Wallis entre los siete días | p = 0,44 |
| η² (varianza explicada por el día de la semana) | 0,006; 0,039 descontando la semana de cada día |
| Fin de semana contra días hábiles (Mann-Whitney) | p = 0,40 |
| Mayor diferencia entre días, respecto de su propia semana | 2.018 kW (8 % del consumo medio) |
| Autocorrelación con rezago 7 contra rezago 6 y 8 | 0,736 contra 0,734 y 0,728: sin pico semanal |
| Boxplot por día ajustado por mes | las siete medianas quedan cerca de cero |

Lo que pudo haber borrado el patrón es que el frío de una cervecería depende sobre todo de la fermentación y la maduración. Los tanques de bodega y los circuitos de glicol funcionan las 24 horas, todos los días, con procesos que duran varios días, así que no siguen el calendario de turnos. Además, la estacionalidad anual y el nivel de cada año pesan mucho más que el día de la semana.

Decisión: no priorizamos el día de la semana. El pipeline crea `Dia_Semana` y `Es_Fin_De_Semana`, pero la selección no las elige: en Train su estadístico F contra el target es de 2,4 (p = 0,12) y 0,2 (p = 0,63), cuando la décima variable seleccionada tiene F = 672.

## 4. ¿Qué mejora mínima justifica reemplazar el modelo?

Definan cuántos kW de diferencia en MAE ameritan cambiar un modelo desplegado. Deriven el umbral de una medición, como el desvío entre folds, la variabilidad del target o el costo de los errores.

**Respuesta**

Nuestro umbral es **300 kW**: un modelo nuevo tiene que bajar el MAE al menos 300 kW para reemplazar al desplegado. Lo derivamos de dos mediciones independientes del ruido del MAE:

- **Desvío entre folds.** Con validación cruzada temporal en Train (5 folds de 100 días), el MAE del modelo final varía entre 2.796 y 3.644 kW, con un desvío de 352 kW (317 sin el primer fold, que entrena con menos datos). Una diferencia menor a eso puede deberse solo al período evaluado.
- **Incertidumbre del MAE en Validation.** Con un bootstrap por bloques de una semana, el MAE de Validation (2.265 kW) tiene un error estándar de 152 kW y un intervalo del 95 % entre 1.974 y 2.567 kW. Dos errores estándar son unos 300 kW.

Los dos caminos coinciden en unos 300 kW, el 10 % del límite de 3.000 kW de la consigna.

Aplicación: comparamos usar 5 o 10 variables. k = 5 depende de menos columnas del Excel, pero la diferencia de MAE con k = 10 es de 34 kW en Validation, muy por debajo del umbral. No hay evidencia para cambiar el modelo, así que mantuvimos k = 10. Por el mismo criterio, la ventaja del modelo sobre la media móvil en un solo período (260 kW en Test) está cerca del umbral. Por eso, en la pregunta 1 nos apoyamos en que esa ventaja se repite en los tres períodos medidos.

## 5. ¿Qué dos técnicas decidieron descartar?

Elijan dos técnicas de los prácticos 1 a 5 que no usaron, como PCA, MICE, codificación cíclica, LOF o DVC. Expliquen por qué descartaron cada una en este proyecto.

**Respuesta**

**PCA.** Lo descartamos por dos razones medidas:

- **Robustez de `predict.py`.** Cada componente principal combina todas las columnas de entrada, así que para predecir haría falta que el Excel trajera las casi 500 candidatas. Entre nuestros propios archivos ya desaparecieron 3 sensores (`Vapor _Vapor_L5` y dos totalizadores de glicol): están en 2020_2022 y no en 2021_2023 ni en 2022_2023. Con PCA, el Excel de 2024–2025 tendría muchas más chances de no servir. Con la selección actual, `predict.py` solo exige 6 columnas.
- **No hace falta para reducir dimensiones.** SelectKBest pasa de 499 a 10 variables. En la comparación de métodos de `preprocesamiento.ipynb`, Ridge baja el MAE de Validation de 6.474 kW (con todas) a 2.273 (con 10), conservando variables con significado físico que se pueden explicar en la defensa.

**LOF (Local Outlier Factor).** Lo descartamos porque en estos datos los valores anómalos no se detectan bien por densidad:

- Los errores reales son desbordes de contador (de 10¹² a 10¹⁶) y negativos en totalizadores acumulados. Se identifican con reglas deterministas y el EDA los reemplaza por faltantes (758 valores en 84 columnas).
- Los días con consumo muy alto o muy bajo son reales: en el cierre de Frío hay solo 3 fuera de 1,5·IQR, y son días que el modelo tiene que poder predecir. Si LOF los eliminara, el modelo aprendería un rango de consumo más estrecho que el real.
- LOF calcula distancias en todas las dimensiones, y con casi 500 columnas, muchas con faltantes, esas distancias pierden sentido.
