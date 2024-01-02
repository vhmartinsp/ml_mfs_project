# Projeto de Machine Learning: Avaliação de fadiga mental de pessoas funcionárias

O projeto demonstra como usar o modelo para fazer predições sobre fadiga mental de pessoas colaboradoras, no intuito de enviar possíveis desgastes e deteriorização do bem-estar destas pessoas.

## Funcionalidades:

Carregamento dos Dados:

- Importação de  as bibliotecas necessárias:

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

- Download dos arquivos (train e test) .csv contendo o dados de treinamento e teste.

Pré-processamento dos Dados:

- Seleção dos atributos relevantes, como designação e  alocação de recursos.
- Realiza o tratamento de dados ausentes ou inconsistentes.
-  Faz a aivisão em Conjuntos de Treinamento e Teste:
- Utiliza a biblioteca scikit-learn para dividir os dados em conjuntos de treinamento e teste.

Treinamento do Modelo de Regressão Linear:

Inicializar e treinar um modelo de regressão linear utilizando o conjunto de treinamento.

Avaliação do Modelo:

- Calcula a métrica de Raiz Quadrada do Erro Quadrático Médio (RMSE) para avaliar a precisão do modelo.
- Visualiza gráficos de dispersão para comparar as predições com os valores reais.

Predição de Fadiga Mental:

- Utiliza o modelo treinado para fazer predições sobre a fadiga mental com base em novos dados.

## Como Usar

- Faça o download do arquivo.ipynb: se você ainda não fez isso, baixe o arquivo do projeto, geralmente com a extensão.ipynb, para o seu computador.
- Abra o Google Colab.
- Carregue o arquivo.ipynb: no menu "Arquivo" do Google Colab, selecione a opção "Fazer upload de notebook". Em seguida, escolha o arquivo.ipynb que você baixou no passo anterior.

- Importe as bibliotecas necessárias: No início do notebook, as bibliotecas necessárias já devem estar importadas. Verifique se as importações são as seguintes:

import numpy as np;
import pandas as pd;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import mean_squared_error, accuracy_score;
from sklearn import tree;
import graphviz

- Execute o notebook: Clique em "Runtime" no menu superior e selecione "Run All". Isso executará todas as células do notebook, incluindo o download dos dados, a divisão dos dados, o treinamento do modelo e a avaliação do modelo.



## Observações

- Certifique-se de que todas as bibliotecas necessárias estão instaladas no ambiente do Google Colab.
- Ao executar o notebook, observe as mensagens de saída para garantir que cada etapa seja concluída com êxito.
- Os resultados, métricas de avaliação e visualizações do modelo serão apresentados ao final da execução do notebook.
- Explore os comentários documentação interna e textos no notebook para uma compreensão mais detalhada de cada etapa do projeto.
