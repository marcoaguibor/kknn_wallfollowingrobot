require(mlbench)
require(kknn)
set.seed(331)

data <- read.csv('https://raw.githubusercontent.com/andersonara/datasets/master/wall-robot-navigation.csv', sep = ';')

# Plota os dados
with(data, {
  plot(X1, X2, col = Y, pch = 19, main = 'Wall-Robot Navigation')
  legend("topright", legend = sort(unique(data$Y)), col = sort(unique(data$Y)), pch = 19)
})


# data splitting
N = nrow(data)
l <- sample(1:nrow(data), 0.7*nrow(data))
amostra.treino <- data[l,]
amostra.teste <- data[-l,]

# Modelo KNN simples
model <- function(k){
  mod <- kknn(as.factor(Y)~., kernel = 'rectangular', k = k, amostra.treino, amostra.teste)
  yfit <- fitted(mod)
  return (yfit)
}
# Modelo KNN com validação leave one out
model_l1o <- function(k){
  return(train.kknn(as.factor(Y)~., data, kmax = 25, ks = k, kernel ='rectangular'))
}
# Confusion matrix macro para o holdout simples
CM_holdout_macro <- function(k){
  modelo <- model(k)
  return(table(modelo, amostra.teste$Y))
}
# Confusion matrix macro para o leave one out
CM_leave_one_out_macro <- function(modelo, k){
  MC <- table(modelo$fitted.values[[1]], data[,3])
  return(MC)
}
model_l1o(1)
# Confusion matrix micro para qualquer um dos modelos
CM_micro <- function(validacao = c('holdout', 'l1o'), k){
  cm_macro <- switch (validacao,
    'holdout' = CM_holdout_macro(k),
    'l1o' = CM_leave_one_out_macro(modelo = model_l1o(k), k)
  )
  CM1=cm_macro[c(1,2),c(1,2)]
  CM2=cm_macro[c(1,3),c(1,3)]
  CM3=cm_macro[c(1,4),c(1,4)]
  CM4=cm_macro[c(2,3),c(2,3)]
  CM5=cm_macro[c(2,4),c(2,4)]
  CM6=cm_macro[c(3,4),c(3,4)]
return (list(CM1, CM2, CM3, CM4, CM5, CM6))
}

# Calcula MCC
MCC <- function(TP, TN, FP, FN){
  numerador <- (TP * TN) - (FP * FN)
  denonimador <- sqrt(TP+FP) * sqrt(TP+FN) * sqrt(TN+FP) * sqrt(TN+FN)
  return(numerador/denonimador)
}

# Calcula F1-Score
f1 <- function(TP, TN, FP, FN){
  numerador <- 2*(TP/(TP+FP)) * (TP/(TP+FN))
  denominador <-  (TP/(TP+FP)) + (TP/(TP+FN))
  return(numerador/denominador)
}

# Calcula o MCC para qualquer um dos métodos de validação
calcular_metricas <- function(validacao = c('holdout', 'l1o'), k){
  CM_macro <- switch (validacao,
          'holdout' = CM_holdout_macro(k),
          'l1o' = CM_leave_one_out_macro(k)
  )
  CM_micro_teste <- CM_micro(validacao = validacao, k)
  TP1=CM_micro_teste[[1]][1,1]
  FP1=CM_micro_teste[[1]][2,1]
  TN1=CM_micro_teste[[1]][2,2]
  FN1=CM_micro_teste[[1]][1,2]
  
  TP2=CM_micro_teste[[2]][1,1]
  FP2=CM_micro_teste[[2]][2,1]
  TN2=CM_micro_teste[[2]][2,2]
  FN2=CM_micro_teste[[2]][1,2]
  
  TP3=CM_micro_teste[[3]][1,1]
  FP3=CM_micro_teste[[3]][2,1]
  TN3=CM_micro_teste[[3]][2,2]
  FN3=CM_micro_teste[[3]][1,2]
  
  TP4=CM_micro_teste[[4]][1,1]
  FP4=CM_micro_teste[[4]][2,1]
  TN4=CM_micro_teste[[4]][2,2]
  FN4=CM_micro_teste[[4]][1,2]
  
  TP5=CM_micro_teste[[5]][1,1]
  FP5=CM_micro_teste[[5]][2,1]
  TN5=CM_micro_teste[[5]][2,2]
  FN5=CM_micro_teste[[5]][1,2]
  
  TP6=CM_micro_teste[[6]][1,1]
  FP6=CM_micro_teste[[6]][2,1]
  TN6=CM_micro_teste[[6]][2,2]
  FN6=CM_micro_teste[[6]][1,2]

  MCC1 <- MCC(TP1, TN1, FP1, FN1)
  MCC2 <- MCC(TP2, TN2, FP2, FN2)
  MCC3 <- MCC(TP3, TN3, FP3, FN3)
  MCC4 <- MCC(TP4, TN4, FP4, FN4)
  MCC5 <- MCC(TP5, TN5, FP5, FN5)
  MCC6 <- MCC(TP6, TN6, FP6, FN6)
  MCC_macro=mean(MCC1,MCC2,MCC3,MCC4, MCC5, MCC6)
  
  F11 <- f1(TP1, TN1, FP1, FN1)
  F12 <- f1(TP2, TN2, FP2, FN2)
  F13 <- f1(TP3, TN3, FP3, FN3)
  F14 <- f1(TP4, TN4, FP4, FN4)
  F15 <- f1(TP5, TN5, FP5, FN5)
  F16 <- f1(TP6, TN6, FP6, FN6)
  F1_macro = mean(F11, F12, F13, F14, F15, F16)
  
  sum_tp = TP1+TP2+TP3+TP4+TP5+TP6
  sum_tn = TN1+TN2+TN3+TN4+TN5+TN6
  sum_fp = FP1+FP2+FP3+FP4+FP5+FP6
  sum_fn = FN1+FN2+FN3+FN4+FN5+FN6
  
  MCC_micro=((sum_tp*sum_tn-sum_fp*sum_fn) / (sqrt(exp(log(sum_tp+sum_fp)+log(sum_tp+sum_fn)+log(sum_tn+sum_fp)+log(sum_tn+sum_fn)))))
  
  F1_micro_numerador = exp(log(2) + log(sum_tp/(sum_tp+sum_fp)) + log(sum_tp/(sum_tp + sum_fn)))
  F1_micro_denominador = (sum_tp/(sum_tp+sum_fp)) + (sum_tp/(sum_tp+sum_fn))
  F1_micro = F1_micro_numerador/F1_micro_denominador
  return(c(MCC_macro, MCC_micro, F1_macro, F1_micro))
}

ks <- 1:25
metricas_holdout_simples <- sapply(ks,function(x)calcular_metricas(validacao = 'holdout',x))
metricas_l1o <- sapply(ks,function(x)calcular_metricas(validacao = 'l1o', x))

# Calcula as métricas macro e micro para holdout simples
mccs_macro_holdout <- metricas_holdout_simples[1,]
mccs_micro_holdout <- metricas_holdout_simples[2,]
f1_macro_holdout <- metricas_holdout_simples[3, ]
f1_micro_holdout <- metricas_holdout_simples[4,]

# Calcula as métricas macro e micro para leave one out
mccs_macro_l1o <- metricas_l1o[1,]
mccs_micro_l1o <- metricas_l1o[2,]
f1_macro_l1o <- metricas_l1o[3,]
f1_micro_l1o <- metricas_l1o[4,]

# Holdout Simples
par(mfrow = c(1,4))
plot(ks, mccs_macro_holdout, ylab = 'MCC', xlab = 'k', main = paste0('Holdout Simples\nMCC macro', '\nMelhor k: ', ks[which.max(mccs_macro_holdout)], ' (', round(max(mccs_macro_holdout), 7), ')'), ylim = c(0.97, 1))
plot(ks, mccs_micro_holdout, ylab = 'MCC', xlab = 'k', main = paste0('Holdout Simples\nMCC micro', '\nMelhor k: ', ks[which.max(mccs_micro_holdout)], ' (', round(max(mccs_micro_holdout), 7), ')'), ylim = c(0.97, 1))
plot(ks, f1_macro_holdout, ylab = 'F1', xlab = 'k', main = paste0('Holdout Simples\nF1 macro', '\nMelhor k: ', ks[which.max(f1_macro_holdout)], ' (', round(max(f1_macro_holdout), 7), ')'), ylim = c(0.97, 1))
plot(ks, f1_micro_holdout, ylab = '', xlab = 'k', main = paste0('Holdout Simples\nF1 micro', '\nMelhor k: ', ks[which.max(f1_micro_holdout)], ' (', round(max(f1_micro_holdout), 7), ')'), ylim = c(0.97, 1))

# Leave one out      
plot(ks, mccs_macro_l1o, ylab = 'MCC', xlab = 'k', main = paste0('MCC macro - L1Out', '\nMelhor k: ', ks[which.max(mccs_macro_l1o)], ' (', round(max(mccs_macro_l1o), 7), ')'), ylim = c(0.97, 1))
plot(ks, mccs_micro_l1o, ylab = '', xlab = 'k', main = paste0('MCC micro - L1Out', '\nMelhor k: ', ks[which.max(mccs_micro_l1o)], ' (', round(max(mccs_micro_l1o), 7), ')'), ylim = c(0.97, 1))
plot(ks, f1_macro_l1o, ylab = 'F1', xlab = 'k', main = paste0('F1 macro - L1Out', '\nMelhor k: ', ks[which.max(f1_macro_l1o)], ' (', round(max(f1_macro_l1o), 7), ')'), ylim = c(0.97, 1))
plot(ks, f1_micro_l1o, ylab = '', xlab = 'k', main = paste0('F1 micro - L1Out', '\nMelhor k: ', ks[which.max(f1_micro_l1o)], ' (', round(max(f1_micro_l1o), 7), ')'), ylim = c(0.97, 1))


####################
# Fronteira de decisão pro modelo final
modelo_final <- model_l1o(1)

x1_range <- seq(0, 5, length.out = 300)
x2_range <- seq(0, 5, length.out = 300)
grid <- expand.grid(X1 = x1_range, X2 = x2_range)
 

with(data, {
  plot(X1, X2, col = Y, pch = 19, main = 'Wall-Robot Navigation')
  points(grid, col = predict(modelo_final, grid), pch = ".", cex = 0.5)
  legend("topright", legend = sort(unique(data$Y)), col = sort(unique(data$Y)), pch = 19)
})

# Fronteira de decisão para outros modelos
par(mfrow = c(1,3))
model_l1o_2 <- model_l1o(2)
model_l1o_3 <- model_l1o(3)
model_l1o_4 <- model_l1o(4)
with(data, {
  plot(X1, X2, col = Y, pch = 19, main = '2NN LOOCV')
  points(grid, col = predict(model_l1o_2, grid), pch = ".", cex = 0.5)
  plot(X1, X2, col = Y, pch = 19, main = '3NN LOOCV')
  points(grid, col = predict(model_l1o_3, grid), pch = ".", cex = 0.5)
  plot(X1, X2, col = Y, pch = 19, main = '4NN LOOCV')
  points(grid, col = predict(model_l1o_4, grid), pch = ".", cex = 0.5)
})

