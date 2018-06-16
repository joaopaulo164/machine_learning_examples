
# coding: utf-8

# # Projeto de K Means Clustering  
# 
# 
# Para este projeto, tentaremos usar o KMeans Clustering para agrupar Universidades em dois grupos: Privadas e Públicas.
# 
# 
# ___
# É muito importante observar, nós realmente temos os rótulos para este conjunto de dados, mas NÃO os usaremos para o algoritmo de agrupamento KMeans, pois esse é um algoritmo de aprendizado não supervisionado. **
# 
# Ao usar o algoritmo Kmeans em situações reais, você não possuirá rótulos. Nesse caso, usaremos os rótulos para tentar ter uma idéia do quão bem o algoritmo foi executado, apenas.
# ___
# 
# ## Os dados
# 
# Usaremos um quadro de dados com 777 observações sobre as 18 variáveis a seguir.
# * Private: Um fator com níveis Não e Sim, indicando universidade privada ou pública.
# * Apps: Número de inscrições recebidas.
# * Accept: Quantidade de inscrições aceitas.
# * Enroll: Número de estudantes matriculados.
# * Top10perc: Percentual de novos estudantes vindo do grupo de 10% melhores do segundo grau.
# * Top25perc: Percentual de novos estudantes vindo do grupo de 25% melhores do segundo grau.
# * F.Undergrad: Número de alunos de graduação em tempo integral.
# * P.Undergrad Número de alunos de graduação em tempo parcial.
# * Outstate: Aulas fora do estado.
# * Room.Board: Custos da sala.
# * Books: Custos de livros estimados.
# * Personal: Estimativa de gastos por pessoa.
# * PhD: Percentual de PHD's na universidade.
# * Terminal: Percentual da faculdade com graduação.
# * S.F.Ratio: Taxa estudantes/faculdade.
# * perc.alumni: Percentual dos ex-alunos que doam.
# * Expend: Despesas da instituição por aluno.
# * Grad.Rate: Taxa de graduação

# ## Importar bibliotecas
# 
# ** Importe as bibliotecas que você costuma usar para análise de dados. **

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().magic('matplotlib inline')


# ## Obtenha os dados

# ** Leia no arquivo College_Data usando read_csv. Descubra como setar a primeira coluna como índice. **

df = pd.read_csv('College_Data',index_col=0)

# ** Verifique o cabeçalho dos dados **

df.head()


# ** Verifique os métodos info() e describe() do DataFrame. **

df.info()

df.describe()


# ## Análise exploratória de dados
#
# É hora de criar algumas visualizações de dados.
#
# ** Crie um scatterplot de Grad.Rate versus Room.Board onde os pontos são coloridos pela coluna "Private". **
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)


# ** Crie um scatterplot de F.Undergrad versus Outstate onde os pontos são coloridos pela coluna Private. **
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)


# ** Crie um histograma empilhado que mostra o "Out of State Tuition" com base na coluna Private. Tente fazer isso usando [sns.FacetGrid](https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.FacetGrid.html). Se isso for muito complicado, veja se você pode fazê-lo apenas usando duas instâncias de pandas.plot(kind='hist'). **
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)


# ** Crie um histograma semelhante para a coluna Grad.Rate. **
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# ** Observe que parece haver uma escola particular com uma taxa de graduação superior a 100%. Qual é o nome dessa escola? **
print("Escola particular com uma taxa de graduação superior a 100%")
df[df['Grad.Rate'] > 100]
print(df[df['Grad.Rate'] > 100])


# ** Defina a taxa de graduação dessa escola para 100 para que isso faça sentido. Você pode obter um aviso (e não um erro) ao fazer esta operação basta usar operações de dataframe ou simplesmente re-fazer a visualização do histograma para garantir que ela realmente foi alterado. **

# In[26]:


df['Grad.Rate']['Cazenovia College'] = 100


# In[27]:


df[df['Grad.Rate'] > 100]


# In[28]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# ## Criação de clusters "K Means"
#
# Agora é hora de criar os rótulos de Cluster!
#
# ** Importe KMeans da SciKit Learn. **

# In[29]:


from sklearn.cluster import KMeans


# ** Crie uma instância do modelo K Means com 2 clusters. **

# In[30]:


kmeans = KMeans(n_clusters=2)


# ** Fite o modelo para todos os dados, exceto para o rótulo privado. **

# In[31]:


kmeans.fit(df.drop('Private',axis=1))


# ** Quais são os vetores centrais do cluster?**

# In[32]:

print("Verotes centrais")
kmeans.cluster_centers_
print(kmeans.cluster_centers_)


# ## Avaliação
# 
# Não há uma maneira perfeita de avaliar o agrupamento se você não tiver os rótulos, no entanto, como isso é apenas um exercício, temos os rótulos então aproveitamos isso para avaliar nossos clusters. Tenha em mente que não terá esse luxo no mundo real.
# 
# ** Crie uma nova coluna para df chamado 'Cluster', que é 1 para escola particular e 0 para uma escola pública. **

# In[33]:


def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


# In[34]:


df['Cluster'] = df['Private'].apply(converter)


# In[35]:


df.head()


# ** Crie uma matriz de confusão e um relatório de classificação para ver o quão bem o clustering K Means funcionou sem ter nenhum rótulo. **

# In[36]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))


# Não tão ruim, considerando que o algoritmo está usando apenas os recursos para agrupar as universidades em 2 grupos distintos. Espero que você possa começar a ver como K Means é útil para agrupar dados não rotulados!

plt.show(block=True)