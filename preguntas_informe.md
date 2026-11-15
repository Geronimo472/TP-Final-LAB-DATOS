# Las cinco preguntas del informe final

Para cada respuesta, expliquen qué decidieron y respáldenlo con un número medido en el proyecto.

## 1. ¿Su modelo agrega valor?

Pongan en una tabla el MAE de persistencia, media móvil de siete días y del modelo final sobre el mismo Test. Si la mejora sobre el mejor baseline queda por debajo del 10 %, expliquen qué ganan con la complejidad adicional y si vale la pena mantenerla.

**Respuesta**

## 2. ¿Qué variables se parecen demasiado al objetivo?

Busquen variables con correlación superior a 0,98 con Frio (kW) del mismo día. Digan si decidieron usarlas y expliquen cómo comprobaron cuándo está disponible cada medición. Justifiquen qué datos del día D usan para predecir D+1.

**Respuesta**

## 3. ¿Qué resultado contradice la intuición?

Comparen la hipótesis de un patrón semanal con lo que midieron en el EDA. Si las dos cosas difieren, den el número, expliquen qué pudo haber borrado el patrón y cuenten qué decidieron hacer.

**Respuesta**

## 4. ¿Qué mejora mínima justifica reemplazar el modelo?

Definan cuántos kW de diferencia en MAE ameritan cambiar un modelo desplegado. Deriven el umbral de una medición, como el desvío entre folds, la variabilidad del target o el costo de los errores.

**Respuesta**

## 5. ¿Qué dos técnicas decidieron descartar?

Elijan dos técnicas de los prácticos 1 a 5 que no usaron, como PCA, MICE, codificación cíclica, LOF o DVC. Expliquen por qué descartaron cada una en este proyecto.

**Respuesta**
