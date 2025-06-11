# Um estudo sobre amostragem em grandes volumes de dados em Redes Sociais Digitais

Este projeto tem por objetivo avaliar o uso de técnicas de amostragem em modelagem de tópicos (bertopic_use_case_1st_turn_day/berttopic_random.ipynb e bertopic_use_case_1st_turn_day/berttopic_stratified.ipynb), em específico, utilizando o BERTopic.

Para avaliar essas técnicas, foram feitas 2 abordagens:
1. Classificação do dataset completo para garantir que os principais tópicos são mantidos.
2. Comparação com um modelo treinado utilizando o dataset completo (bertopic_use_case_1st_turn_day/berttopic_full.ipynb), por meio da combinação de clusters, o que permitiu classificar as amostras em relação a um modelo treinado com o dataset inteiro.

A partir dos resultados obtidos, foi alimentado o Gemini com as principais palavras dos tópicos para gerar os nomes finais para os tópicos, mas somente por razões ilustrativas do que cada tópico seria, dado que as comparações são realizadas em cima dos valores dos tópicos e modelos treinados.

Para acesso ao dataset completo, favor entrar em contato (devido a limitações do GitHub, não é possível disponibilizá-lo).

## Pré-requisitos
Neste trabalho foram utilizados o UMAP e o HDBSCAN da biblioteca cuML. Essa biblioteca utiliza GPU para processamento dos dados. Para instalá-la, seguir as instruções disponíveis em: https://docs.rapids.ai/ - no meu caso, fiz a instalação da versão Stable, por um Docker no WSL, utilizando CUDA 12.0-12.8 e o Python 3.11. Como pacotes adicionais, instalei o PyTorch.

Por meio do Python desse Docker, instalei as bibliotecas sentence-transformers e bertopic, para complementar com as ferramentas necessárias para execução do projeto.