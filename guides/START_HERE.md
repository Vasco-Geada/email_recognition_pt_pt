# 🚀 COMEÇAR AQUI

Bem-vindo ao **Sistema de Gold Annotations** para Email Recognition PT-PT!

## ⚡ 30 Segundos para Começar

```bash
# Navegar para diretório
cd gold_annotations

# Verificar instalação
python setup.py verify

# Gerar gold annotations
python gold_annotations_generator.py ../dataset/realistic_emails_v2.json output/gold.json

# Ver resultado
# (Abrir output/gold.json em editor JSON ou visualizador)
```

## 📚 Documentação por Nível

### 🟢 **Iniciante (5 minutos)**
→ Abrir: **QUICKSTART.md**
- Formato de input
- Gerar anotações em 3 passos
- Exemplo mínimo funcional

### 🟡 **Intermédio (15 minutos)**
→ Abrir: **README.md**
- Visão geral completa
- API principal
- Exemplos de uso
- Troubleshooting

### 🔴 **Avançado (30 minutos)**
→ Abrir: **TECHNICAL_DOCUMENTATION.md**
- Arquitetura detalhada
- Algoritmos de extração
- Design patterns
- Integração com modelos

### 📖 **Referência Completa**
→ Abrir: **INDEX.md**
- Mapa de ficheiros
- Todas as classes
- Todas as APIs
- Checklist de funcionalidades

## 🎯 Casos de Uso

### Caso 1: Gerar Anotações Rápidas
```bash
python gold_annotations_generator.py emails.json output.json
```
Tempo: ~1 segundo por 100 emails

### Caso 2: Validar Anotações
```python
from validators import AnnotationValidator
validator = AnnotationValidator()
result = validator.validate_batch(annotations)
print(f"Válido: {result.is_valid}")
```

### Caso 3: Avaliar Modelo
```bash
python evaluate_annotations.py gold.json predictions.json -o report.json
```
Gera relatório com Precision/Recall/F1

### Caso 4: Revisar Manualmente
1. Gerar anotações automáticas
2. Abrir JSON e editar
3. Validar novamente
4. Usar como gold standard

## 📁 Estrutura de Ficheiros

```
gold_annotations/
├── 📖 COMEÇAR_AQUI.md              ← Você está aqui!
├── 📖 QUICKSTART.md                 ← 5 minutos
├── 📖 README.md                     ← Guia completo
├── 📖 TECHNICAL_DOCUMENTATION.md    ← Detalhes
├── 📖 INDEX.md                      ← Referência
├── 📖 SUMMARY.md                    ← Resumo executivo
│
├── 🐍 heuristic_extractors.py       ← Extractores
├── 🐍 validators.py                 ← Validação
├── 🐍 gold_annotations_generator.py ← Orquestrador
├── 🐍 evaluate_annotations.py       ← Avaliação
├── 🐍 config.py                     ← Configuração
│
├── 🧪 test_gold_annotations.py      ← Testes
├── 📋 example_usage.py              ← Exemplos
├── ⚙️  setup.py                     ← Setup
├── 📦 requirements.txt              ← Dependências
└── 📂 output/                       ← Resultados
```

## ✨ Destaques

| Feature | Status | Docs |
|---------|--------|------|
| Trigger Detection | ✅ | README.md |
| Participant Extraction | ✅ | README.md |
| Temporal Expression Detection | ✅ | README.md |
| Location Detection | ✅ | README.md |
| Topic Detection | ✅ | README.md |
| Validation | ✅ | README.md |
| Evaluation (P/R/F1) | ✅ | README.md |
| CLI Interface | ✅ | QUICKSTART.md |
| Python API | ✅ | README.md |
| Unit Tests | ✅ | test_gold_annotations.py |
| Examples | ✅ | example_usage.py |

## 🔧 Requisitos

- **Python**: 3.11+ (verificado)
- **Dependências**: Nenhuma (stdlib only)
- **Espaço**: ~50MB para 1000 emails
- **Tempo**: ~10 segundos por 1000 emails

## ✅ Verificação Rápida

```bash
# Todos OK?
python setup.py verify

# Saída esperada:
# [OK] Python 3.11+ OK
# [OK] Módulos disponíveis
# [OK] Ficheiros encontrados
# [OK] Imports funcionais
# → Sistema pronto para uso!
```

## 🎬 Execução Recomendada

1. **Primeiro**: Ler **QUICKSTART.md** (5 min)
2. **Depois**: Rodar `python example_usage.py` (30 seg)
3. **Depois**: Gerar suas anotações (1 min)
4. **Depois**: Consultar **README.md** para casos avançados

## 🆘 Problemas?

| Problema | Solução |
|----------|---------|
| "ModuleNotFoundError" | Certifique-se de estar no diretório correto |
| "FileNotFoundError" | Verifique caminhos para dataset.json |
| Poucas anotações encontradas | Consultar TECHNICAL_DOCUMENTATION.md |
| Encoding errors | Use UTF-8 em editores de texto |

## 📊 Exemplo de Saída

### Input (emails.json)
```json
[{
  "subject": "Reunião amanhã",
  "body": "Boas Ana, podemos reunir às 15h?",
  "label": "agendamento_reuniao"
}]
```

### Output (gold_annotations.json)
```json
[{
  "id": 1,
  "text": "Boas Ana, podemos reunir às 15h?",
  "intent": "agendamento_reuniao",
  "trigger": ["reunir"],
  "arguments": {
    "participants": ["Ana"],
    "time": ["15h"],
    "location": [],
    "topic": []
  },
  "confidence": {...}
}]
```

## 🚀 Próximos Passos

### Para Iniciantes
- [ ] Ler QUICKSTART.md
- [ ] Rodar example_usage.py
- [ ] Gerar anotações com seu próprio dataset

### Para Desenvolvimento
- [ ] Ler TECHNICAL_DOCUMENTATION.md
- [ ] Explorar config.py
- [ ] Adicionar novos triggers/patterns

### Para Pesquisa
- [ ] Ler INDEX.md (referência)
- [ ] Usar evaluate_annotations.py
- [ ] Integrar com seu modelo

## 📞 Suporte Rápido

| Pergunta | Resposta |
|----------|----------|
| Como começar? | QUICKSTART.md |
| Como usar? | README.md |
| Como funciona? | TECHNICAL_DOCUMENTATION.md |
| O que existe? | INDEX.md |
| Resumo? | SUMMARY.md |

## 🎯 TL;DR (Too Long; Didn't Read)

```bash
# Copy-paste para começar:
cd gold_annotations
python setup.py verify
python example_usage.py
python gold_annotations_generator.py ../dataset/realistic_emails_v2.json output.json
```

Pronto! Veja seu output em `output/gold_annotations_v1.json`

---

**Versão:** 1.0  
**Última atualização:** 11 de Maio de 2024  
**Status:** ✅ Pronto para Usar

### 👉 Próxima ação: Abrir **QUICKSTART.md**
