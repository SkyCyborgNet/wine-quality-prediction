#####################################################################
# PROYECTO FINAL - HARVARDX CAPSTONE: CHOOSE YOUR OWN
# VERSIÓN DEFINITIVA - TODOS LOS ERRORES CORREGIDOS
# AUTOR: David Rodriguez
# FECHA: 2026
#####################################################################

cat("==================================================\n")
cat("=== INICIANDO PIPELINE DE DATA SCIENCE ===\n")
cat("==================================================\n\n")

# =================================================================
# FASE 1: CONFIGURACIÓN Y GESTIÓN DE PAQUETES
# =================================================================

# Configurar el directorio de trabajo
setwd("C:/Users/Eric Flores/Downloads/- Universidad/Coursera/Data Science Harvard/Capstone Project")

options(digits = 4)

# Definición de paquetes requeridos
paquetes_proyecto <- c(
    "tidyverse",
    "caret",
    "corrplot",
    "rpart",
    "randomForest",
    "xgboost",
    "matrixStats",
    "ggplot2",
    "gridExtra"
)

# Función para instalar y cargar paquetes
gestionar_librerias <- function(paquete) {
    if (!require(paquete, character.only = TRUE)) {
        cat(paste(" -> Instalando paquete:", paquete, "...\n"))
        install.packages(paquete, repos = "http://cran.us.r-project.org", dependencies = TRUE)
        library(paquete, character.only = TRUE)
    } else {
        cat(paste(" -> Paquete cargado:", paquete, "\n"))
    }
}

# Cargar todos los paquetes
invisible(lapply(paquetes_proyecto, gestionar_librerias))

cat("\n[OK] Entorno configurado correctamente.\n\n")

# =================================================================
# FASE 2: CARGA DEL DATASET DE VINO TINTO
# =================================================================

cat("=== CARGANDO DATASET DE VINO TINTO ===\n")

# El archivo ya está descargado, solo lo cargamos
archivo_local <- "winequality-red.csv"

if(!file.exists(archivo_local)) {
    cat("Descargando dataset...\n")
    url_datos <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    download.file(url_datos, destfile = archivo_local)
}

# Cargar datos
datos_vino <- read_delim(archivo_local, delim = ";", show_col_types = FALSE)
colnames(datos_vino) <- make.names(colnames(datos_vino))

cat("Dimensiones del dataset:", dim(datos_vino), "\n")
cat("Variables disponibles:", colnames(datos_vino), "\n\n")

# =================================================================
# FASE 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =================================================================

cat("=== ANÁLISIS EXPLORATORIO DE DATOS ===\n")

# Resumen estadístico
print(summary(datos_vino))

# Verificar valores nulos
null_counts <- colSums(is.na(datos_vino))
cat("\nValores nulos por variable:\n")
print(null_counts)

# Distribución de la calidad
cat("\nDistribución de calidad:\n")
print(table(datos_vino$quality))

# =================================================================
# FASE 4: INGENIERÍA DE VARIABLES
# =================================================================

cat("\n=== INGENIERÍA DE VARIABLES ===\n")

# Crear variable objetivo binaria
datos_vino <- datos_vino %>%
    mutate(
        Calidad_Binaria_Factor = factor(ifelse(quality >= 6, "Bueno", "Malo"),
                                        levels = c("Malo", "Bueno")),
        Calidad_Binaria_Numerica = as.numeric(quality >= 6)
    )

# Verificar balance de clases
cat("Balance de clases:\n")
print(table(datos_vino$Calidad_Binaria_Factor))
cat("Proporción de vinos buenos:", mean(datos_vino$Calidad_Binaria_Numerica), "\n\n")

# =================================================================
# FASE 5: DIVISIÓN DE DATOS (TRAIN/TEST)
# =================================================================

cat("=== DIVISIÓN DE DATOS ===\n")

set.seed(1, sample.kind = "Rounding")

indice_entrenamiento <- createDataPartition(datos_vino$Calidad_Binaria_Factor, 
                                            p = 0.8, 
                                            list = FALSE)

train_set <- datos_vino[indice_entrenamiento, ]
test_set <- datos_vino[-indice_entrenamiento, ]

cat("Tamaño entrenamiento:", nrow(train_set), "\n")
cat("Tamaño prueba:", nrow(test_set), "\n")
cat("Proporción entrenamiento:", nrow(train_set)/nrow(datos_vino), "\n\n")

# =================================================================
# FASE 6: CONFIGURACIÓN DE VALIDACIÓN CRUZADA
# =================================================================

control_cv <- trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = defaultSummary,
    verboseIter = FALSE
)

# =================================================================
# FASE 7: ENTRENAMIENTO DE MODELOS
# =================================================================

cat("=== ENTRENAMIENTO DE MODELOS ===\n")

# --------------------------------------------
# MODELO 1: Árbol de Decisión (CART)
# --------------------------------------------
cat("\n[1/3] Entrenando CART...\n")

set.seed(1, sample.kind = "Rounding")
modelo_cart <- train(
    Calidad_Binaria_Factor ~ .,
    data = train_set %>% select(-Calidad_Binaria_Numerica),
    method = "rpart",
    trControl = control_cv,
    tuneLength = 10
)

pred_cart <- predict(modelo_cart, test_set)
cm_cart <- confusionMatrix(pred_cart, test_set$Calidad_Binaria_Factor)

cat("Accuracy CART:", cm_cart$overall["Accuracy"], "\n")
cat("Mejor CP:", modelo_cart$bestTune$cp, "\n")

# --------------------------------------------
# MODELO 2: Random Forest
# --------------------------------------------
cat("\n[2/3] Entrenando Random Forest...\n")

set.seed(1, sample.kind = "Rounding")
rejilla_rf <- expand.grid(mtry = c(2, 4, 6, 8))

modelo_rf <- train(
    Calidad_Binaria_Factor ~ .,
    data = train_set %>% select(-Calidad_Binaria_Numerica),
    method = "rf",
    trControl = control_cv,
    tuneGrid = rejilla_rf,
    ntree = 150,
    importance = TRUE
)

pred_rf <- predict(modelo_rf, test_set)
cm_rf <- confusionMatrix(pred_rf, test_set$Calidad_Binaria_Factor)

cat("Accuracy RF:", cm_rf$overall["Accuracy"], "\n")
cat("Mejor mtry:", modelo_rf$bestTune$mtry, "\n")

importancia_rf <- varImp(modelo_rf)

# --------------------------------------------
# MODELO 3: XGBoost
# --------------------------------------------
cat("\n[3/3] Entrenando XGBoost...\n")

train_xgb <- train_set %>% 
    select(-Calidad_Binaria_Factor, -Calidad_Binaria_Numerica) %>%
    as.matrix()

train_y_xgb <- train_set$Calidad_Binaria_Numerica

test_xgb <- test_set %>%
    select(-Calidad_Binaria_Factor, -Calidad_Binaria_Numerica) %>%
    as.matrix()

test_y_xgb <- test_set$Calidad_Binaria_Numerica

dtrain <- xgb.DMatrix(data = train_xgb, label = train_y_xgb)
dtest <- xgb.DMatrix(data = test_xgb, label = test_y_xgb)

watchlist <- list(train = dtrain, test = dtest)

params_xgb <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    gamma = 0,
    nthread = 2,
    seed = 1
)

set.seed(1, sample.kind = "Rounding")
modelo_xgb <- xgb.train(
    params = params_xgb,
    data = dtrain,
    nrounds = 200,
    verbose = 0,
    early_stopping_rounds = 10,
    evals = watchlist,
    maximize = FALSE
)

pred_xgb_prob <- predict(modelo_xgb, dtest)
pred_xgb_class <- factor(ifelse(pred_xgb_prob >= 0.5, "Bueno", "Malo"),
                         levels = c("Malo", "Bueno"))

cm_xgb <- confusionMatrix(pred_xgb_class, test_set$Calidad_Binaria_Factor)
cat("\nAccuracy XGBoost:", cm_xgb$overall["Accuracy"], "\n")

# =================================================================
# FASE 8: COMPARACIÓN DE MODELOS
# =================================================================

cat("\n=== COMPARACIÓN DE MODELOS ===\n")

resultados <- data.frame(
    Modelo = c("CART (Árbol)", "Random Forest", "XGBoost"),
    Accuracy = c(cm_cart$overall["Accuracy"],
                 cm_rf$overall["Accuracy"],
                 cm_xgb$overall["Accuracy"]),
    Kappa = c(cm_cart$overall["Kappa"],
              cm_rf$overall["Kappa"],
              cm_xgb$overall["Kappa"])
)

print(resultados)

mejor_modelo <- resultados[which.max(resultados$Accuracy), ]
cat("\n🏆 MEJOR MODELO:", mejor_modelo$Modelo, 
    "con Accuracy:", round(mejor_modelo$Accuracy, 4), "\n")

# =================================================================
# FASE 9: VERIFICACIÓN DE VISUALIZACIONES
# =================================================================

cat("\n=== VERIFICACIÓN DE VISUALIZACIONES ===\n")

# Verificar que las imágenes existen
imagenes <- c("matriz_correlacion.png", "distribucion_calidad.png", 
              "comparacion_modelos.png", "importancia_variables.png")

for (img in imagenes) {
    if (file.exists(img)) {
        cat("✅", img, "- OK\n")
    } else {
        cat("❌", img, "- NO ENCONTRADO\n")
    }
}

# =================================================================
# FASE 10: REPORTE FINAL
# =================================================================

cat("\n==================================================\n")
cat("=== REPORTE FINAL DEL PROYECTO ===\n")
cat("==================================================\n\n")

cat("1. DATASET DE VINO TINTO\n")
cat("   - Muestras:", nrow(datos_vino), "\n")
cat("   - Predictores:", ncol(datos_vino) - 2, "\n")
cat("   - Mejor modelo:", mejor_modelo$Modelo, "\n")
cat("   - Accuracy:", round(mejor_modelo$Accuracy, 4), "\n\n")

cat("2. DESEMPEÑO DE MODELOS\n")
for (i in 1:nrow(resultados)) {
    cat("   -", resultados$Modelo[i], ":", round(resultados$Accuracy[i], 4), "\n")
}

cat("\n==================================================\n")
cat("=== PIPELINE COMPLETADO EXITOSAMENTE ===\n")
cat("==================================================\n")