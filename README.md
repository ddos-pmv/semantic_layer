# Antispam Semantic Campaign Layer

Проект исследует semantic layer для антиспам-систем. Его задача не решать
`spam` или `not spam`, а давать объяснимый сигнал:

```text
How similar is this message to known spam campaigns?
```

Этот сигнал должен использоваться вместе с классическими антиспам-признаками.

## Структура проекта

```text
core/
  lib/                  C++ библиотека SemCore
  tests/                CMake target для C++ тестов
  benchmarks/           CMake target для C++ benchmark
  scripts/              вспомогательные скрипты
  third_party/          vendored C++ зависимости

eval_data/              небольшие JSONL датасеты для экспериментов
models/                 локальные model artifacts
tests/                  Python pre-screen tests
requirements.txt        зависимости для Python-проверок
pytest.ini              pytest markers
```

`core/lib` сейчас содержит базовые building blocks: wrapper для ONNX Runtime
модели и Hugging Face tokenizer через `tokenizers-cpp`. Это должно развиваться
как reusable core library, без зависимости от HTTP/gRPC/service layer.

## Модель

Текущая модель:

```text
models/paraphrase-multilingual-MiniLM-L12-v2
```

Скрипт загрузки скачивает минимальный набор файлов для двух runtime-сценариев:

- `SentenceTransformers` для Python pre-screen тестов;
- `ONNX` для C++ core experiments.

Загрузить модель:

```sh
./core/scripts/download_model.sh
```

Production startup не должен скачивать модели из Hugging Face. Для production
модель должна быть versioned artifact: скачана заранее, проверена checksum,
положена во внутреннее хранилище и доставлена вместе с сервисом или volume.

## Python: pre-screen tests

Создать окружение и установить зависимости:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Запустить основные проверки модели:

```sh
pytest -q tests/test_embedding_contract.py
```

Эти тесты проверяют:

- что локальная модель полностью скачана;
- что embedding имеет ожидаемую размерность;
- что embedding содержит конечные значения;
- как top-k nearest neighbors группируются на маленьком датасете.

Запустить latency/throughput проверки:

```sh
pytest -q -m performance tests/test_performance.py
```

Performance-тесты медленнее и не являются обязательным быстрым smoke-test.

## C++ core

Установить базовые macOS-зависимости:

```sh
./core/scripts/install_deps_macos.sh
```

Для C++ части нужен ONNX Runtime. Передайте путь к установленному ONNX Runtime
через `ONNXRUNTIME_ROOT`.

Пример сборки:

```sh
cmake -S core -B core/build -G Ninja \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime

cmake --build core/build
```

Запустить C++ test target:

```sh
./core/build/tests/onnx_test
```

Запустить C++ benchmark target:

```sh
./core/build/benchmarks/onnx_bench
```

Текущие C++ targets пока являются инфраструктурными проверками: загрузка ONNX
Runtime, чтение model metadata и tokenizer. Полная semantic scoring логика еще
не реализована.

## MVP direction

Ближайшая цель проекта — reusable semantic core, который возвращает
`signal_only`, а не `block`.

Для MVP нужны:

- локальная загрузка embedding-модели;
- генерация message embedding;
- normalized vector comparison;
- маленькая campaign database;
- top-k nearest campaign search;
- explainable matched examples;
- простой evidence score;
- batch benchmark;
- thin API/service wrapper поверх core.

Semantic similarity сама по себе не должна принимать финальное антиспам-решение.
