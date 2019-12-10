#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 10
@author: karlvandesman
"""

import pandas as pd
import numpy as np

# Statistical Test
from scipy.stats import mannwhitneyu

path = 'dataset/recife_etlsinasc_2016.csv'

df = pd.read_csv(path, index_col=0)

#%%
#############################################
# Teste de Hipótese para doenças congênitas #
#############################################
    
alpha=0.05
idade_limite = 45

# Novo atributo, que indica se o nascido tem ou não a anomalia
df['ANOMAL'] = (df.CODANOMAL.notnull()).astype('int')

for idade in range(20, idade_limite+1):
    print('### IDADE : %d ### '%idade)
    
    # Separa a distribuição com base no atributo criado, 'ANOMAL'
    # e na idade colocada como limite
    dist_anomal_risco = (df.ANOMAL[df.IDADEMAE>=idade]).values
    dist_anomal_segura = (df.ANOMAL[df.IDADEMAE<idade]).values

    #### Cálculo das proporções ###
    # Proporção da idade de risco
    num_idade_risco = len(df[df.IDADEMAE>=idade])
    num_casos_risco = (df.ANOMAL[df.IDADEMAE>=idade]).sum()
    
    p_idade_risco = num_casos_risco/num_idade_risco
    
    # Proporção da idade 'segura'
    num_idade_segura = len(df[df.IDADEMAE<idade])
    num_casos_seguro = (df.ANOMAL[df.IDADEMAE<idade]).sum()
    
    p_idade_segura = num_casos_seguro/num_idade_segura

    # Proporção combinada
    p_combinada = (num_casos_risco+num_casos_seguro)/len(df)
    q_combinada = 1 - p_combinada

    print('Proporção risco = %.4f | Proporção idade segura: %.4f'%(p_idade_risco, p_idade_segura))

    ### Teste Não-paramétrico para as proporções ###
    statistic_anomal, pvalue_anomal = mannwhitneyu(dist_anomal_risco, dist_anomal_segura, alternative='greater')
    
    print('Teste não-paramétrico Mann-Whitney:\n\t', end='')
    if(pvalue_anomal<0.05):
        print('Distribuições diferentes, pvalue= ', pvalue_anomal)
    else:
        print('Distribuições iguais, pvalue=', pvalue_anomal)
    print()
    
    # Se os requisitos para o teste Z não forem satisfeitos, pula para a próxima
    # iteração
    if(num_casos_seguro*p_idade_segura<5 or num_casos_seguro*(1 - p_idade_segura)<5
       or num_casos_risco*p_idade_risco<5 or num_casos_risco*(1 - p_idade_risco)<5):
        print('X X X - Requisitos não satisfeitos para o teste Z - X X X')
        print()
        #print(num_casos_risco*p_idade_risco)
        #print(num_casos_risco*(1 - p_idade_risco))
        continue
    
    ### Teste estatístico Z para as proporções ###
    z = (p_idade_segura - p_idade_risco)/np.sqrt(p_combinada*q_combinada * (1/num_casos_seguro + 1/num_casos_risco))
    
    print('Teste paramétrico Z:\n\t', end='')
    if(abs(z)>1.96):
        print('Proporções significativamente diferentes (z=%.4f) para idade limite = %d'%(z, idade))
    else:
        print('Proporções iguais (z=%.4f) para idade = %d'%(z, idade))
        
    print()