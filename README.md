# Projeto CNN Local com Streamlit

Projeto didático baseado em uma CNN simples para classificação binária de imagens, adaptado para rodar localmente no VS Code.

## Estrutura esperada do dataset

Coloque as imagens neste formato:

```text
data/
  training_set/
    cat/
      imagem1.jpg
    dog/
      imagem2.jpg
  test_set/
    cat/
      imagem3.jpg
    dog/
      imagem4.jpg
  single_prediction/
    exemplo.jpg
```

Os nomes das pastas `cat` e `dog` podem ser alterados, mas o problema deve continuar binário.

## 1. Criar ambiente virtual

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## 3. Treinar o modelo

```bash
python src/train.py
```

O treinamento irá salvar:

```text
models/cnn_model.keras
models/class_indices.json
reports/metrics.json
reports/confusion_matrix.png
```

## 4. Fazer uma predição por linha de comando

```bash
python src/predict.py --image data/single_prediction/exemplo.jpg
```

## 5. Executar a aplicação Streamlit

```bash
streamlit run app/streamlit_app.py
```

## Objetivo didático

Este projeto separa claramente:

- preparação dos dados;
- construção da CNN;
- treinamento;
- avaliação com métricas;
- salvamento do modelo;
- deploy local com Streamlit.
