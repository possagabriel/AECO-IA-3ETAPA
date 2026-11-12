# Classificação do dataset Iris com rede neural em NumPy

Atividade da disciplina de Inteligência Artificial (UNIPAC Barbacena).

Implementação, do zero, de uma rede neural artificial com uma camada escondida
para classificar as três espécies de flores do dataset Iris. As operações
matriciais usam apenas **NumPy**; o **Scikit-Learn** é utilizado exclusivamente
para carregar e preparar os dados.

## Dataset e pré-processamento

- `load_iris` fornece 150 amostras, 4 características e 3 classes.
- Divisão estratificada de 80% para treino e 20% para teste.
- Padronização com média 0 e desvio padrão 1. O `StandardScaler` é ajustado
  somente no conjunto de treino, evitando vazamento de dados.
- Rótulos convertidos para one-hot encoding com NumPy.
- Semente padrão `42` para reprodutibilidade.

## Arquitetura

| Camada  | Neurônios | Ativação |
| ------- | --------- | -------- |
| Entrada | 4         | —        |
| Oculta  | 8         | ReLU     |
| Saída   | 3         | Softmax  |

O forward pass é dado por:

```text
z1 = X · W1 + b1        a1 = ReLU(z1)
z2 = a1 · W2 + b2       p  = Softmax(z2)
```

A perda é a entropia cruzada média:

```text
L = -(1/N) · Σ Σ y · log(p)
```

O backward pass aplica a regra da cadeia:

```text
dz2 = (p - y) / N
dW2 = a1ᵀ · dz2        db2 = Σ dz2
da1 = dz2 · W2ᵀ
dz1 = da1 · ReLU'(z1)
dW1 = Xᵀ · dz1         db1 = Σ dz1
```

Os pesos são inicializados com He/Xavier e os vieses com pequenos valores
aleatórios. A atualização usa gradiente descendente em lote completo:
`parâmetro -= taxa_de_aprendizado · gradiente`.

## Estrutura do projeto

```text
.
├── main.py              # CLI de treino e avaliação
├── iris_nn/
│   ├── __init__.py
│   ├── data.py          # carregamento, split, padronização e one-hot
│   └── network.py       # forward, loss, backprop, gradiente descendente
├── tests/
│   ├── test_data.py
│   └── test_network.py
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Parâmetros opcionais:

```bash
python main.py --epochs 2000 --learning-rate 0.05 --hidden-size 8 \
  --test-size 0.2 --seed 42 --log-every 100
```

## Testes e qualidade

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
ruff format --check .
```

A suíte cobre o pré-processamento, as ativações, a estabilidade numérica da
Softmax, a forma dos tensores e a verificação dos gradientes por diferenças
finitas, além de confirmar que a rede aprende o Iris.

## Resultados

Execução padrão (`python main.py`: 1000 épocas, taxa 0.1, 8 neurônios ocultos,
seed 42), com Python 3.14.3, NumPy 2.5.4 e Scikit-Learn 1.9.1:

| Métrica                    | Valor   |
| -------------------------- | ------- |
| Perda inicial              | 1.899856 |
| Perda final (treino)       | 0.048610 |
| Acurácia de treino         | 97.50%  |
| Acurácia de teste          | 96.67%  |

O conjunto de teste possui 30 amostras; 96,67% corresponde a 29 acertos.

## Escopo das bibliotecas

- **NumPy**: inicialização, operações matriciais, ativações, perda, gradientes
  e atualização dos parâmetros.
- **Scikit-Learn**: somente `load_iris`, `train_test_split` e `StandardScaler`.
- Nenhum framework de deep learning foi utilizado.
