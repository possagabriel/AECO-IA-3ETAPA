# Classificação do dataset Iris com rede neural em NumPy

Rede neural artificial 4-8-3 implementada **do zero** para classificar as três
espécies de flores do dataset Iris. Toda a matemática — forward pass,
entropia cruzada, backpropagation e gradiente descendente — usa apenas
**NumPy**. O **Scikit-Learn** é utilizado exclusivamente para carregar e
preparar os dados.

Projeto desenvolvido para a disciplina de Inteligência Artificial da UNIPAC
Barbacena, com foco em clareza, reprodutibilidade e boas práticas de engenharia
de software.

## Destaques

- Rede 4 → 8 (ReLU) → 3 (Softmax) implementada sem frameworks de deep learning.
- Pré-processamento sem vazamento de dados: padronização ajustada só no treino.
- Divisão estratificada de 80% para treino e 20% para teste.
- Treinamento por gradiente descendente em lote completo com seed configurável.
- Relatórios em texto, JSON e CSV, além de persistência do modelo em `.npz`.
- CLI com validação de argumentos, logging configurável e artefatos opcionais.
- Testes com verificação numérica dos gradientes e **100% de cobertura**.
- Lint com Ruff, tipagem estrita com mypy e integração contínua no GitHub Actions.

## Requisitos

- Python 3.11 ou superior
- NumPy 2.x
- Scikit-Learn 1.5 ou superior

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

Para instalar apenas as dependências de execução:

```bash
pip install .
```

Os arquivos `requirements.txt` e `requirements-dev.txt` continuam disponíveis
como atalhos para instalação via `pip`.

## Uso rápido

```bash
python main.py
```

Formas equivalentes:

```bash
python -m iris_nn
iris-nn
```

Exemplo com artefatos e parâmetros personalizados:

```bash
iris-nn \
  --epochs 2000 \
  --learning-rate 0.05 \
  --hidden-size 8 \
  --seed 7 \
  --report reports/run.json \
  --history reports/loss.csv \
  --save-model models/iris.npz
```

### Opções da CLI

| Opção             | Padrão | Descrição                                      |
| ----------------- | ------ | ---------------------------------------------- |
| `--epochs`        | `1000` | Número de épocas de treinamento                |
| `--learning-rate` | `0.1`  | Taxa de aprendizado do gradiente descendente   |
| `--hidden-size`   | `8`    | Neurônios da camada escondida                  |
| `--test-size`     | `0.2`  | Fração reservada para teste                    |
| `--seed`          | `42`   | Semente para inicialização e divisão dos dados |
| `--log-every`     | `100`  | Intervalo de épocas entre logs de perda        |
| `--log-level`     | `INFO` | Nível dos logs (`DEBUG` a `CRITICAL`)          |
| `--report`        | —      | Grava um relatório JSON                        |
| `--history`       | —      | Grava o histórico de perda em CSV              |
| `--save-model`    | —      | Salva o modelo treinado em `.npz`              |
| `--version`       | —      | Exibe a versão do pacote                       |

## Saída esperada

```text
========================================================================
         Classificação de flores Iris com rede neural em NumPy
========================================================================
Treino: 120 amostras | Teste: 30 amostras
Arquitetura: 4 -> 8 (ReLU) -> 3 (Softmax)
Épocas: 1000 | Taxa de aprendizado: 0.1 | Seed: 42
Parâmetros treináveis: 67
------------------------------------------------------------------------
Perda inicial:   1.899856
Perda final:     0.048610
Tempo de treino: 0.1516 s
------------------------------------------------------------------------
Métricas
                        Treino       Teste
  Acurácia             97.50%      96.67%
  Precisão macro       97.52%      96.97%
  Recall macro         97.50%      96.67%
  F1 macro             97.50%      96.66%

Matriz de confusão de teste (linhas = real, colunas = previsto)
                      setosa  versicolor   virginica
setosa                    10           0           0
versicolor                 0           9           1
virginica                  0           0          10
========================================================================
```

## Arquitetura

```mermaid
flowchart LR
    A["4 características<br/>padronizadas"] --> B["8 neurônios<br/>ReLU"]
    B --> C["3 neurônios<br/>Softmax"]
    C --> D["Probabilidades<br/>por espécie"]
```

| Camada  | Neurônios | Ativação |
| ------- | --------- | -------- |
| Entrada | 4         | —        |
| Oculta  | 8         | ReLU     |
| Saída   | 3         | Softmax  |

Forward pass:

```text
z1 = X · W1 + b1        a1 = ReLU(z1)
z2 = a1 · W2 + b2       p  = Softmax(z2)
```

Perda de entropia cruzada média:

```text
L = -(1/N) · Σ Σ y · log(p)
```

Backward pass pela regra da cadeia:

```text
dz2 = (p - y) / N
dW2 = a1ᵀ · dz2        db2 = Σ dz2
da1 = dz2 · W2ᵀ
dz1 = da1 · ReLU'(z1)
dW1 = Xᵀ · dz1         db1 = Σ dz1
```

Atualização dos parâmetros:

```text
W1 -= η · dW1           b1 -= η · db1
W2 -= η · dW2           b2 -= η · db2
```

Os pesos usam inicialização He/Xavier e os vieses recebem pequenos valores
aleatórios. A Softmax é calculada com deslocamento pelo máximo para evitar
estouro numérico.

## Pré-processamento

1. `load_iris` fornece 150 amostras, 4 características e 3 classes.
2. Divisão estratificada: 120 amostras para treino e 30 para teste.
3. `StandardScaler` ajustado somente no treino e aplicado ao teste.
4. Rótulos convertidos para one-hot encoding com NumPy.
5. Semente padrão `42` para divisão, inicialização e reprodutibilidade.

## Métricas

Além da acurácia exigida na atividade, o projeto calcula:

- precisão, recall e F1 por classe;
- médias macro de precisão, recall e F1;
- matriz de confusão;
- suporte e número total de amostras avaliadas.

As métricas são implementadas em NumPy no módulo `iris_nn.metrics` e exportadas
nos relatórios JSON.

## Resultados

Execução padrão (`python main.py`, 1000 épocas, taxa 0.1, 8 neurônios ocultos,
seed 42), com Python 3.14.3, NumPy 2.5.4 e Scikit-Learn 1.9.1:

| Métrica               | Treino | Teste  |
| --------------------- | ------ | ------ |
| Acurácia              | 97.50% | 96.67% |
| Precisão macro        | 97.52% | 96.97% |
| Recall macro          | 97.50% | 96.67% |
| F1 macro              | 97.50% | 96.66% |

- Perda inicial: `1.899856`
- Perda final: `0.048610`
- Parâmetros treináveis: `67`
- Acurácia de teste: **29 acertos em 30 amostras**

Matriz de confusão do conjunto de teste:

| Real \ Previsto | setosa | versicolor | virginica |
| --------------- | ------ | ---------- | --------- |
| setosa          | 10     | 0          | 0         |
| versicolor      | 0      | 9          | 1         |
| virginica       | 0      | 0          | 10        |

## Uso programático

```python
from iris_nn import NeuralNetwork, load_prepared_iris

dataset = load_prepared_iris(test_size=0.2, random_state=42)
model = NeuralNetwork(n_inputs=4, n_hidden=8, n_outputs=3, learning_rate=0.1, seed=42)

history = model.fit(
    dataset.X_train,
    dataset.y_train,
    epochs=1000,
    log_every=100,
    verbose=False,
)

accuracy = model.accuracy(dataset.X_test, dataset.y_test)
print(f"Acurácia de teste: {accuracy:.2%}")
```

Para carregar um modelo salvo:

```python
from iris_nn.persistence import load_model

model = load_model("models/iris.npz")
```

## Estrutura do projeto

```text
.
├── .github/workflows/ci.yml
├── iris_nn/
│   ├── __init__.py          # API pública e versão
│   ├── __main__.py          # python -m iris_nn
│   ├── _typing.py           # aliases de tipos NumPy
│   ├── _version.py          # versão canônica
│   ├── cli.py               # interface de linha de comando
│   ├── data.py              # carregamento e pré-processamento
│   ├── experiment.py        # orquestração de treino e avaliação
│   ├── metrics.py           # métricas e matriz de confusão
│   ├── network.py           # forward, loss, backprop e gradiente descendente
│   ├── persistence.py       # save/load de modelos .npz
│   ├── py.typed             # marcador de tipagem PEP 561
│   └── reporting.py         # relatórios em texto, JSON e CSV
├── tests/
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_data.py
│   ├── test_entrypoints.py
│   ├── test_experiment.py
│   ├── test_metrics.py
│   ├── test_network.py
│   ├── test_persistence.py
│   └── test_reporting.py
├── Makefile
├── main.py
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Qualidade e testes

```bash
make test       # pytest
make coverage   # pytest + cobertura
make lint       # ruff + ruff format --check + mypy
make format     # aplica correções e formatação
```

A suíte possui **103 testes** e **100% de cobertura** de linhas e branches do
pacote. Os testes incluem:

- verificação dos gradientes por diferenças finitas;
- estabilidade numérica da Softmax;
- validação de formatos e valores de entrada;
- reprodutibilidade com a mesma seed;
- round-trip de persistência do modelo;
- conteúdo dos relatórios JSON e CSV;
- testes de fumaça dos pontos de entrada (`main.py` e `python -m iris_nn`).

O workflow de CI executa lint, verificação de formatação, tipagem estrita e
testes com cobertura em Python 3.11, 3.12, 3.13 e 3.14.

## Decisões de projeto

- **Gradiente descendente em lote completo:** adequado ao tamanho do Iris e
  fiel ao enunciado da atividade.
- **Sem early stopping:** não há conjunto de validação separado; o conjunto de
  teste é usado apenas na avaliação final.
- **Inicialização controlada:** He para a camada ReLU, Xavier para a saída e
  vieses pequenos, conforme solicitado.
- **Métricas macro:** a média macro trata todas as classes com o mesmo peso e é
  informativa em problemas com classes desbalanceadas.
- **Artefatos opcionais:** relatórios, histórico e modelo só são gravados quando
  os caminhos são informados.

### Limitações

- O modelo não utiliza regularização, dropout ou early stopping.
- O Iris é um dataset pequeno, balanceado e de baixa dimensionalidade; os
  resultados não devem ser generalizados para problemas mais complexos.
- A avaliação usa uma única divisão treino/teste, embora a seed garanta
  reprodutibilidade.

## Escopo das bibliotecas

- **NumPy:** inicialização, operações matriciais, ativações, perda, gradientes,
  métricas e atualização dos parâmetros.
- **Scikit-Learn:** somente `load_iris`, `train_test_split` e `StandardScaler`.
- **Biblioteca padrão:** CLI, logging, relatórios, persistência e tipagem.
- **Ferramentas de desenvolvimento:** pytest, pytest-cov, Ruff e mypy.

Nenhum framework de deep learning foi utilizado.

## Referências

- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic
  problems*. Annals of Eugenics, 7(2), 179–188.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT
  Press.
- Harris, C. R. et al. (2020). *Array programming with NumPy*. Nature, 585,
  357–362.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*.
  Journal of Machine Learning Research, 12, 2825–2830.

---

Projeto desenvolvido para fins acadêmicos.
