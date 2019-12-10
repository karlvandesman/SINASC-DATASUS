#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 10
@author: karlvandesman
"""

import pandas as pd
from matplotlib import pyplot as plt

from scipy.stats import kurtosis
from scipy.stats import skew
from statsmodels.graphics.gofplots import qqplot

# Statistical tests
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest

# Non-parametric tests
from scipy.stats import kruskal
from scikit_posthocs import posthoc_nemenyi

path = 'dataset/recife_etlsinasc_2016.csv'

df = pd.read_csv(path, index_col=0)

print('df dimensões (original): ', df.head())

# Criar classe binária para peso saudável
#df['PESO_BAIXO'] = 0 + ((df_drop['PESO']>2500).values)

#%%
# =============================================================================
# Estatística Descritiva
# =============================================================================

peso = df.PESO.values

qqplot(peso, fit=True, line='45')
plt.xlabel('Quantis da Normal')
plt.ylabel('Quantis observados da amostra')
plt.title('Gráfico Q-Q dos pesos de nascidos em Recife - 2016')
plt.figure(figsize=(20,20))
plt.show()

curtose_peso = kurtosis(peso)
assimetria_peso = skew(peso)

print('Características da distribuição de peso:')
print('Curtose = ', curtose_peso)
print('Assimetria = ', assimetria_peso)
print()

#%%
# =============================================================================
# Testes de aderência: Normalidade
# =============================================================================

# Kolmogorov-Smirnov 
ks_stat, ks_p = kstest(df.PESO.values, 'norm')

# Shapiro-Wilk
shapiro_stat, shapiro_p = shapiro(df.PESO.values)   #UserWarning: p-value may 
                                                    #not be accurate for N>5000


# D’Agostino e Pearson
DP_stat, DP_p = normaltest(df.PESO.values)

print('*** Testes de aderência: Normalidade ***')
print('Kolmogorov-Smirnov: estatística=%.4f, p-valor=%.8f'%(ks_stat, ks_p))
print('Shapiro-Wilk: estatística=%.4f, p-valor=%.8f'%(shapiro_stat, shapiro_p))
print('D’Agostino e Pearson: estatística=%.4f, p-valor=%.8f'%(DP_stat, DP_p))
print()

#%%
############################################
# Teste de Hipótese para peso segundo raça #
############################################

# Separar por raça os valores dos pesos
peso_branca = df.PESO[df.RACACOR==1].values
peso_preta = df.PESO[df.RACACOR==2].values
peso_amarela = df.PESO[df.RACACOR==3].values
peso_parda = df.PESO[df.RACACOR==4].values
peso_indigena = df.PESO[df.RACACOR==5].values

pesos_racas = [peso_branca, peso_preta, peso_amarela, peso_parda, 
               peso_indigena]

# Teste de Kruskal-Wallis, que compara o conjunto
stat_peso_raca, p_peso_raca = kruskal(peso_branca, peso_preta, peso_amarela,
                                      peso_parda, peso_indigena)

print('Teste de Kruskal-Wallis para variação de raça/cor:')
print('Estatística do teste = %.2f, p-valor = %.7f'%(stat_peso_raca, 
                                                     p_peso_raca))

# Se o teste de Kruskal rejeitar a hipótese nula (mesmas distribuições), então
# é feito o pós-teste de Nemenyi, comparando par a par os testes

alpha = 0.05
if p_peso_raca > alpha:
    print('Mesmas distribuições de peso para as raças/cores (falha em rejeitar H0)')
    print()
else:
    print('Diferentes distribuições segundo a raça/cor (rejeita H0)')
    print()
    print('Pós-teste de Nemenyi:')
    print(posthoc_nemenyi(pesos_racas))
