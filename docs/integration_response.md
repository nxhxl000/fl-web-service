# Integration response: ответы flwr-репо на требования fl-web-service

Ответ на `integration_contract.md` со стороны flwr-репо (`ddl`). Документ
отвечает на открытые вопросы, описывает фактическое состояние кода, фиксирует
расхождения с требованиями и перечисляет пункты, которые flwr-репо готов
реализовать под контракт.

Дата: 2026-04-11. Актуально для коммита `f093ad7` + локальные изменения.

---

## 1. Фактическое окружение (проверено по SSH на VM)

Проверено на сервере (`83.149.250.70`) и на клиенте №0 — окружения идентичны.

| Компонент | Версия | Комментарий |
|---|---|---|
| Python | **3.12.3** | Строго. Используется и на сервере, и на всех клиентах |
| `flwr` | **1.28.0** | **Не 1.22**, как указано в `pyproject.toml`. Фактически стоит 1.28 |
| `torch` | **2.11.0** | CPU-сборка. `CUDA: False`, MKL-DNN включён |
| `torchvision` | **0.26.0** | |
| `timm` | **1.0.26** | Нужен для EfficientNet-B0 (PlantVillage) |
| `datasets` | **4.8.4** | HuggingFace Datasets |
| `numpy` | **2.4.4** | numpy 2.x — важно, 1.x сломает |
| `pandas` | **3.0.2** | Только на сервере (артефакты) |
| `matplotlib` | **3.10.8** | Только на сервере (графики) |
| `pillow` | **12.2.0** | |

**Важные детали для Docker-образа `fl-client`:**

- `torch` ставить из CPU-index:
  `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu`
  Иначе потянется CUDA-вариант и образ раздует на ~2 GB впустую.
- `matplotlib` и `pandas` **клиенту не нужны** — они только для артефактов сервера.
  Не ставьте в клиентский образ.
- `numpy` строго `>=2.0`. Если ваш backend пиннит `numpy<2`, будет конфликт.
- Python 3.12 обязателен. `flwr 1.28` работает на 3.11/3.12/3.13, но у нас
  фиксировано 3.12 — используйте ту же версию, чтобы избежать расхождений
  в pickle/cloudpickle.

**Минимальный `pip install` для клиентского Docker**:
```bash
pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cpu
pip install flwr==1.28.0 datasets==4.8.4 timm==1.0.26 \
  numpy==2.4.4 pillow==12.2.0
pip install /path/to/flwr-repo     # ставит ddl-flower-app (пакет fl_app)
```

**Расхождение с `pyproject.toml`**: пины в `pyproject.toml` устарели
(`flwr>=1.22.0`). Обновить под фактические версии — в списке работ ниже.

---

## 2. Имя пакета и способ установки (R5)

- **Имя пакета:** `ddl-flower-app` (`pyproject.toml` → `[project].name`).
  **Не `ddl`**, как предполагалось в гипотезе веб-сервиса.
- **Модули импортируются как `fl_app.*`** — пакет собирается из корня репо
  через `hatchling`:
  ```toml
  [tool.hatch.build.targets.wheel]
  packages = ["."]
  ```
- **Python import path после установки:** `from fl_app.client_app import app`,
  `from fl_app.models import build_model, get_hparams`, etc.

**Рекомендуемый способ установки (R5.1) — вариант 3: копирование исходников
в Docker-образ**:
```dockerfile
COPY ./flwr-repo /app/flwr-repo
RUN pip install /app/flwr-repo
```
Причины:
- Пакет не опубликован в PyPI.
- `pip install git+https://...` требует привязки к публичному remote — для
  приватного проекта хуже.
- Editable install (`-e`) не подходит для контейнера.
- Пакет самодостаточен: `pip install .` из корня ставит `fl_app` через hatchling.

Альтернативы работают, но требуют инфраструктурных решений (PyPI, git access).

---

## 3. Оркестрация SuperLink (R1)

### 3.1. Программного Python-API нет

SuperLink запускается **только** как CLI-процесс. Соответствующая команда:

```bash
flower-superlink --insecure
```

Порты по умолчанию:
- `9092` — Fleet API (SuperNodes → SuperLink)
- `9093` — Exec API (`flwr run` → SuperLink)

Оба порта дефолтные, переопределяются флагами `--fleet-api-address` /
`--exec-api-address`. Сейчас `--insecure` достаточно — TLS/mTLS не используются
(см. R4 контракта: explicit not needed).

### 3.2. Реакция на сигналы (R1.4)

`flower-superlink` штатно реагирует на `SIGTERM` → graceful shutdown. В нашем
деплое запускается в tmux-сессии `superlink` на сервере (см.
`deploy/start_superlink.sh`), останавливается через `stop_all.sh` / `tmux kill`.
Из веб-сервиса достаточно `subprocess.Popen` + `proc.terminate()`.

### 3.3. Запуск эксперимента (R1.3)

Параметры эксперимента задаются **не флагами `flower-superlink`**, а через
`pyproject.toml` секцию `[tool.flwr.app.config]` + команду `flwr run`:

```bash
flwr run . <federation-name>                  # запуск
flwr log <run-id> . <federation-name>         # стриминг логов (НЕ flwr run)
flwr ls . <federation-name> --runs            # список запусков
```

Где `<federation-name>` — ключ из `[tool.flwr.federations.<name>]` в
`pyproject.toml`, содержащий адрес Exec API:
```toml
[tool.flwr.federations.remote]
address = "83.149.250.70:9093"
insecure = true
```

**Важно: у веб-сервиса два пути настройки run-config**:
1. **Статический шаблон `pyproject.toml`**: подставляет значения в файл перед
   вызовом `flwr run`. Работает для всех полей.
2. **Overrides через `--run-config`**: `flwr run . remote --run-config "key=value ..."`.
   Пары через пробел в одном кавыченном аргументе, поддерживает числа/строки/bool.
   **Не поддерживает** вложенные dict, поэтому параметры стратегии
   (`proximal-mu=0.05`) нужно передавать как плоские ключи верхнего уровня — см.
   раздел 9 ниже (сейчас это не реализовано).

Пример:
```bash
flwr run . remote --run-config \
  "aggregation='fedprox' num-server-rounds=20 model='wrn_16_4' \
   partition-name='cifar100__iid__n10__s42'"
```

### 3.4. Как веб-сервис получает run-id

`flwr run` при старте печатает `run` + число. Паттерн для парсинга:
`r"run\s+(\d+)"`. Логи эксперимента затем берутся через `flwr log <run-id> . remote`.

Этот `run-id` сейчас **не связан** с `exp_name` (локальным именем директории
артефактов на сервере). Связку можно добавить — см. раздел «Что flwr-репо готов
сделать» ниже.

---

## 4. Клиентская часть (R2)

### 4.1. Точка входа

Клиент — это `ClientApp` (новый Flower 1.x API):
```python
# fl_app/client_app.py
from flwr.clientapp import ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context) -> Message: ...
```

Регистрируется в `pyproject.toml`:
```toml
[tool.flwr.app.components]
clientapp = "fl_app.client_app:app"
serverapp = "fl_app.server_app:app"
```

**Запускается через CLI Flower (внутри Docker-контейнера)**:
```bash
flower-supernode \
  --superlink <host>:9092 \
  --insecure \
  --node-config "partition-id=<N> num-partitions=<M>"
```

Никакого `python -m fl_app.client` не существует. SuperNode сам находит
`ClientApp` по `pyproject.toml` в рабочей директории.

### 4.2. Конфигурация клиента (R2.2)

**Сейчас клиент не читает переменные окружения.** Весь конфиг идёт через два канала:

1. **`context.run_config`** — значения из `[tool.flwr.app.config]`, приходят
   от сервера через Flower. Клиент читает:
   - `rc["data-mode"]`: `research` или `production`
   - `rc["model"]`: имя модели из `MODEL_REGISTRY`
   - `rc["partition-name"]`, `rc["local-epochs"]`, `rc["data-dir"]`,
     `rc["lr-decay"]`, `rc["num-server-rounds"]`

2. **`context.node_config`** — пары `key=value` из `--node-config` флага
   `flower-supernode`. **Все пары в одном кавыченном аргументе через пробел**:
   `--node-config "partition-id=0 data-dir=/data"`. Несколько отдельных
   `--node-config` argparse схлопывает до последнего — известный подводный камень.

   - **Research mode**: `context.node_config["partition-id"]` → путь
     `data_dir/partitions/<name>/client_<pid>`
   - **Production mode**: `context.node_config["data-dir"]` → абсолютный путь
     к данным клиента (HF Dataset на диске или imagefolder с папками-классами)

### 4.3. Режим `production` для веб-сервиса — уже работает

Критично: **режим `data-mode = "production"` в клиенте уже реализован**
(`fl_app/client_app.py:50-54`). Веб-сервису нужно только:
- Установить в `run_config`: `data-mode = "production"`
- Передать `--node-config "data-dir=/абсолютный/путь/в/контейнере"` при старте
  `flower-supernode`
- Смонтировать volume с данными клиента в этот путь

Формат данных поддерживается двух видов (детектируется автоматически):
- **HF Dataset на диске**: есть файл `dataset_info.json` → `load_from_disk()`
- **imagefolder**: обычные папки-классы с `.jpg/.png` → `load_dataset("imagefolder", ...)`

### 4.4. Чего не хватает под R2 и что нужно добавить

Под R2.2 веб-сервису нужны env-переменные: `SUPERLINK_ADDR`, `CLIENT_ID`,
`DATA_DIR`, `DATASET_NAME`. Это решается **entrypoint-скриптом Docker**, а не
кодом клиента:

```bash
#!/bin/bash
# docker-entrypoint.sh
exec flower-supernode \
  --superlink "${SUPERLINK_ADDR}" \
  --insecure \
  --node-config "data-dir=${DATA_DIR} client-token=${CLIENT_ID}"
```

`client-token` клиент может прочитать из `context.node_config` и проставить в
`MetricRecord` — тогда веб-сервис получит его в событиях `client_update`
(см. раздел 8).

**Docker-образ сейчас не собран** — это задача веб-сервиса (из контракта) + наша
задача предоставить entrypoint-скрипт и документацию.

### 4.5. CPU-only (R2.4)

Подтверждено. `fl_app/training.py:get_device()` возвращает `cpu`. На всех VM
стоит CPU-сборка torch (`CUDA available: False`). GPU-пути в коде не используются.

### 4.6. Восстановление после разрывов (R2.5)

`flower-supernode` сам переподключается к SuperLink при разрыве связи. Наш
код клиента stateless между раундами за исключением SCAFFOLD-control variate
(хранится в `context.state["c_client"]`). Явного тюнинга таймаутов в коде нет —
используются дефолты Flower 1.28.

---

## 5. Модели (R3.1)

`fl_app/models/__init__.py` → `MODEL_REGISTRY`. Публичный API:
- `build_model(name: str) -> nn.Module`
- `get_hparams(name: str) -> TrainHParams` (dataclass с `lr`, `batch_size`,
  `momentum`, `weight_decay`, `num_workers`)

### 5.1. Доступные модели (проверено `sum(p.numel() for p in m.parameters())`)

| Имя | Параметры | Датасет | Вход | Назначение |
|---|---|---|---|---|
| `simple_cnn` | **1.63M** | CIFAR-100 | 32×32 | Baseline |
| `wrn_16_4` | **2.77M** | CIFAR-100 | 32×32 | **Основная модель для CIFAR-100** |
| `wrn_16_4_gn` | ~2.77M | CIFAR-100 | 32×32 | GroupNorm вариант (для non-IID) |
| `wrn_28_4` | **5.87M** | CIFAR-100 | 32×32 | Слишком тяжёлая для CPU-FL |
| `efficientnet_b0` | **4.06M** | PlantVillage | 224×224 | Через `timm`, scratch training |

### 5.2. Расхождение с R3.1 (≤1M для PlantVillage)

**Модели ≤1M для PlantVillage сейчас нет.** Минимум — `efficientnet_b0` 4.06M.

Варианты:
- `MobileNetV3-small` через `timm` (~1.5M) — **заготовка есть** в
  `fl_app/models/plantvillage/mobilenet.py`, но не зарегистрирована в реестре.
  Могу добавить и проверить.
- Урезанный EfficientNet-B0 (меньше каналов) — кастомная архитектура.
- **Либо пересогласовать лимит**: 4M params = ~16 MB на раунд на клиента
  (float32). Для 10 клиентов и 20 раундов это ~3.2 GB общего трафика. CPU-обучение
  EfficientNet-B0 на PlantVillage работает, но медленно (несколько минут на
  клиент-эпоху).

**Вопрос к веб-сервису**: лимит ≤1M жёсткий или ориентир? От этого зависит,
добавляем ли MobileNet или оставляем EfficientNet.

### 5.3. Привязка модель → датасет

В коде жёсткой привязки нет — `MODEL_REGISTRY` знает только `num_classes` в
kwargs (`100` для CIFAR-100 WRN-ов, `38` для EfficientNet под PlantVillage).
Веб-сервису для фильтрации «какие модели доступны для датасета X» нужно либо:

- Жёстко прошитый маппинг (`"cifar100": ["wrn_16_4", "simple_cnn"], "plantvillage": ["efficientnet_b0"]`) — могу добавить в `MODEL_REGISTRY` или `DATASET_META`.
- Вытаскивать `num_classes` из kwargs и фильтровать по датасету.

Рекомендую первое — явнее.

---

## 6. Партиционирование данных (R3.2)

Два режима, переключаемых через `data-mode` в `run_config`:

### 6.1. Research mode (текущий)

Офлайн-скрипт `scripts/partition_utils.py` режет датасет на N клиентских
партиций + test-split + опциональный серверный датасет, сохраняет в HuggingFace
Arrow формате:

```
data/partitions/<partition-name>/
  client_0/        # HF Dataset
  client_1/
  ...
  client_N-1/
  test/            # тест-сплит (общий для сервера)
  server/          # серверный датасет (опционально, для экспериментов)
  manifest.json    # метаданные: dataset, scheme, alpha, num_classes, class_names, ...
```

Потом каждый клиент получает всю директорию `partitions/<name>/` через rsync и
по `partition-id` из `--node-config` выбирает свой `client_<pid>`.

**Публичный API** `scripts/partition_utils.py`:
- `DATASET_CONFIG` — реестр `{name: DatasetConfig(label_col, img_col, has_test, test_fraction)}`
- `get_dataset_config(name) -> DatasetConfig`
- `prepare_splits(ds, dataset_name, seed) -> (train_ds, test_ds)` — унифицированный train/test (PlantVillage не имеет test split → делает stratified 80/20)
- `partition_dataset(train_ds, num_clients, scheme, *, alpha, min_per_class, seed, label_col) -> list[Dataset]`
- `save_partitions(train_ds, test_ds, partitions, out_dir, *, dataset, scheme, alpha, min_per_class, seed, label_col, server_dataset, force) -> Path`
- `partition_dir_name(dataset, scheme, num_clients, seed, *, alpha, min_per_class, server_size) -> str`
- `load_manifest(out_dir) -> dict`

Поддерживаемые схемы: `iid`, `dirichlet` (с параметрами `alpha`, `min_per_class`
для гарантированного минимума на класс).

### 6.2. Production mode (для веб-сервиса)

Веб-сервису research-механика не нужна — вы сами решаете, откуда клиент
получает данные (volume mount, MinIO download, rsync). Клиент в этом режиме
читает `context.node_config["data-dir"]` как абсолютный путь и загружает его
как HF Dataset или imagefolder.

**Схема данных для PlantVillage** (если используете imagefolder):
```
/data/plantvillage/
  Apple___Apple_scab/
    img_001.jpg
    img_002.jpg
  Apple___healthy/
    ...
  ...
```
Папка = имя класса, класс-ID присваивается по алфавитному порядку
автоматически (`datasets.load_dataset("imagefolder", ...)`). **Важно**: все
клиенты должны иметь **одинаковый набор папок**, иначе class IDs разъедутся.
Если у клиента нет какого-то класса — создайте пустую папку или используйте
HF Dataset с явным ClassLabel.

### 6.3. Число классов и их имена (R3.3)

Три источника в зависимости от контекста:

1. **Research (есть партиция)**: `data/partitions/<name>/manifest.json` →
   поля `num_classes`, `class_names`. Используется текущим server_app.
2. **После эксперимента**: `experiments/<exp-name>/config.json` → те же поля,
   плюс весь конфиг эксперимента.
3. **Production (константа)**: **сейчас такой функции нет**. Нужно добавить
   `fl_app/datasets/metadata.py` с:
   ```python
   DATASET_META = {
       "cifar100":     {"num_classes": 100, "class_names": [...]},
       "plantvillage": {"num_classes": 38,  "class_names": [...]},
   }
   def get_dataset_meta(name: str) -> dict: ...
   ```
   Сделаю по запросу — это часть задач ниже.

---

## 7. Сохранение весов и инференс (R3.4)

### 7.1. Где и как сохраняются веса

`fl_app/server_app.py:288` — после завершения FL-цикла:
```python
torch.save(result.arrays.to_torch_state_dict(), str(model_path))
```
где `model_path = experiments/<exp-name>/model.pt`.

Формат — стандартный PyTorch `state_dict`, совместимый с `load_state_dict()`:
```python
model = build_model("wrn_16_4")
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
```

### 7.2. Структура директории эксперимента

```
experiments/<partition>__<model>__<agg>__<hash>/
  config.json                 # Полный конфиг: model, aggregation, hparams, num_classes, class_names, ...
  summary.json                # best_acc, best_f1, best_round, total_wall_time, total_comm_mb
  model.pt                    # Финальные веса (state_dict)
  train.log                   # Текстовый лог per-round
  metrics/
    rounds.csv                # Метрики по раундам: acc, f1, loss, timing, comm, drift
    clients.csv               # Метрики по клиентам: train_loss, round_time, drift
    classes.csv               # Per-class accuracy по раундам
  plots/
    accuracy.png
    f1.png
    train_loss_boxplot.png
    class_accuracy.png
  cluster_profile.json        # Только если enable-profiling=true
```

`exp_name` — строка вида `cifar100__iid__n10__s42__wrn_16_4__fedavg__abc12345`
(хеш — от timestamp, гарантирует уникальность).

### 7.3. Что нужно веб-сервису для инференса

1. **Файл весов**: `experiments/<exp>/model.pt`
2. **Имя модели** (для `build_model`): `experiments/<exp>/config.json` → `model`
3. **Имена классов**: `experiments/<exp>/config.json` → `class_names`, `num_classes`
4. **Transforms** для препроцессинга входного изображения: **сейчас публичной
   функции нет** (зашиты в `fl_app/training.py:make_dataloader`). Нужна
   `get_eval_transform(model_name_or_dataset) -> callable`. В списке задач.

Минимальный inference-код при наличии всего этого:
```python
import json, torch
from PIL import Image
from fl_app.models import build_model
from fl_app.training import get_eval_transform   # нужно добавить

exp = "experiments/cifar100__iid__n10__s42__wrn_16_4__fedavg__abc12345"
cfg = json.loads(open(f"{exp}/config.json").read())

model = build_model(cfg["model"])
model.load_state_dict(torch.load(f"{exp}/model.pt", map_location="cpu"))
model.eval()

transform = get_eval_transform(cfg["model"])
img = Image.open("input.jpg")
x = transform(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
pred_class = cfg["class_names"][logits.argmax(dim=1).item()]
```

Веб-сервис кладёт `model.pt` + `config.json` в MinIO как один «артефакт модели»
и подтягивает обратно в inference-worker. Наше дело — дать публичный
`get_eval_transform`.

---

## 8. События раундов (R4)

### 8.1. Текущее состояние: публикации нет

Сейчас никаких колбеков для внешних подписчиков **нет**. Есть:
- `LoggingStrategy` wrapper (`fl_app/strategies.py:208-247`) — перехватывает
  `aggregate_train` для сбора метрик клиентов и записи в CSV.
- `global_evaluate` callback в `server_app.py:220` — вызывается после каждого
  раунда, пишет в `rounds.csv` / `train.log` / stdout.

Всё идёт в файлы и stdout в human-readable формате (не JSON).

### 8.2. Что нужно добавить (план реализации)

Модуль `fl_app/events.py`:

```python
import os, json, time, sys
from typing import Any

class EventPublisher:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.redis = None
        url = os.environ.get("REDIS_URL")
        if url:
            try:
                import redis
                self.redis = redis.Redis.from_url(url)
                self.channel = f"fl:events:{run_id}"
            except Exception:
                self.redis = None

    def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "type": event_type,
            "run_id": self.run_id,
            "ts": time.time(),
            **payload,
        }
        line = json.dumps(event, ensure_ascii=False)
        if self.redis:
            try:
                self.redis.publish(self.channel, line)
                return
            except Exception:
                pass
        # Fallback: stdout JSONL
        print(f"FL_EVENT {line}", file=sys.stderr, flush=True)
```

Хуки (все на стороне сервера, в одном процессе с `flower-superlink` / `flwr run`):

| Событие | Место | Payload |
|---|---|---|
| `round_started` | `server_app.py`, в начале каждого раунда (через подмену `aggregate_train` или отдельный колбек) | `round`, `num_clients` |
| `client_update` | `LoggingStrategy.aggregate_train` — итерация по `replies` | `round`, `client_id` (partition-id или client-token), `num_examples`, `train_loss`, `round_time_sec`, `drift` |
| `round_finished` | `global_evaluate`, перед return | `round`, `test_acc`, `test_f1`, `test_loss`, `duration_sec`, `cum_comm_mb` |
| `run_started` | `server_app.py`, до FL-цикла | `exp_name`, `model`, `aggregation`, `num_rounds`, `num_classes` |
| `run_finished` | `server_app.py`, после `strategy.start` | `exp_name`, `best_acc`, `best_f1`, `best_round`, `total_wall_time_sec` |

### 8.3. Канал публикации (R4.2, R4.3)

- **Первичный — Redis pub/sub**. `REDIS_URL` из env, канал
  `fl:events:<run_id>` (run_id = `exp_name` либо `flwr run-id`, см. ниже).
- **Fallback — stdout JSONL**. Префикс строки `FL_EVENT ` (чтобы веб-сервис мог
  грепать stdout subprocess'а `flwr log`), за ним JSON одной строкой.
- **Формат**: JSON с обязательными полями `type`, `run_id`, `ts`
  (UNIX-время float, секунды), далее type-specific поля. Схема финализируется
  к Дню 8 дорожной карты веб-сервиса — готов согласовать точные имена полей.

### 8.4. Про run_id

**Здесь есть подводный камень**: у нас два «run_id»:
- `exp_name` — локальное имя директории артефактов, формируется в
  `make_exp_dir()` из `<partition>__<model>__<agg>__<hash>`. Гарантирует
  уникальность на диске.
- `flwr run-id` — целое число, которое печатает `flwr run` и использует
  `flwr log` / `flwr ls`. Живёт в Exec API SuperLink.

Связи между ними в артефактах **нет**. Если веб-сервис хочет матчить события от
SuperLink с артефактами на диске, нужно сохранять `flwr_run_id` в
`config.json`/`summary.json`. Делается в 2 строки (`context` в
`ServerApp.@app.main` должен иметь `run_id` — проверю на практике).

**Рекомендую**: использовать `flwr run-id` как `run_id` в событиях (веб-сервис
его знает после `flwr run`) и добавить связку `flwr_run_id → exp_name` в
`summary.json`.

### 8.5. Важно про Redis

`REDIS_URL` нужно передать **тому процессу, который крутит `ServerApp`**. В
Flower 1.28 это процесс `flower-superlink` (он спавнит SerrverApp внутри себя).
Значит:
- Либо `REDIS_URL` ставится в env при старте `flower-superlink`
  (`deploy/start_superlink.sh`).
- Либо SuperLink запускается веб-сервисом как subprocess с env уже
  проброшенным — тогда ничего править в наших скриптах не надо.

Второй вариант чище для архитектуры (веб-сервис оркестрирует SuperLink).

---

## 9. Стратегии (R1.1)

`fl_app/strategies.py` → `STRATEGY_REGISTRY`. Сейчас доступны:

| Имя | Параметры (дефолты) | Примечание |
|---|---|---|
| `fedavg` | *(нет)* | Baseline |
| `fedprox` | `proximal_mu=0.01` | |
| `fedavgm` | `server_learning_rate=1.0`, `server_momentum=0.9` | |
| `fedadam` | `eta=0.01`, `eta_l=0.001`, `beta_1=0.9`, `beta_2=0.99`, `tau=1e-9` | |
| `fedyogi` | те же что `fedadam` | |
| `fedadagrad` | `eta=0.01`, `eta_l=0.001`, `tau=1e-9` | |
| `fedmedian` | *(нет)* | Byzantine-robust, min 2 клиента |
| `fedtrimmedavg` | `beta=0.2` | Byzantine-robust, min 2 клиента |
| `krum` | `num_malicious_nodes=0` | Byzantine-robust, min 2 клиента |
| `multikrum` | `num_malicious_nodes=0`, `num_nodes_to_select=1` | Byzantine-robust |
| `bulyan` | `num_malicious_nodes=0` | Byzantine-robust, min 4 клиента |
| `scaffold` | *(нет)* | Наша реализация (`ScaffoldStrategy(FedAvg)`) |

Публичный API:
```python
def build_strategy(
    name: str, *,
    fraction_train: float,
    min_train_nodes: int,
    min_available_nodes: int,
) -> tuple[Any, dict[str, Any]]:
    """Returns (strategy_instance, params_dict)."""
```

**Расхождение с R1.1**: сейчас параметры стратегий **прошиты в `STRATEGY_REGISTRY`
как дефолты из dataclass-ов и не принимаются извне**. Чтобы веб-сервис мог
настраивать `proximal_mu`, `eta`, `beta_1` и т.д., нужно:

- Расширить `build_strategy` до `**overrides: Any` и мержить в `cfg`.
- Договориться о ключах: плоские `proximal-mu=0.05` в `run_config`, или
  вложенный `strategy-params={"proximal_mu": 0.05}`. `run-config` не
  поддерживает вложенные dict → плоский вариант с префиксом, например
  `strategy-proximal-mu=0.05`.
- `build_strategy` читает из `context.run_config` ключи с префиксом
  `strategy-*` и мержит.

Готов реализовать. В списке задач ниже.

---

## 10. Что flwr-репо готов сделать под контракт

Пункты, которые нужны веб-сервису и которых сейчас нет в коде. Каждый — оценка
трудозатрат (часы).

| # | Задача | Часы | Приоритет |
|---|---|---|---|
| 1 | Обновить пины в `pyproject.toml` под фактические версии VM (`flwr==1.28.0` и т.д.) | 0.5 | P0 |
| 2 | `fl_app/datasets/metadata.py` + `DATASET_META` с `num_classes`/`class_names` для CIFAR-100 и PlantVillage | 1 | P0 |
| 3 | `fl_app/training.get_eval_transform(name)` — публичный preprocessing для инференса | 1 | P0 |
| 4 | `fl_app/events.py` + интеграция хуков в `LoggingStrategy` и `server_app` (Redis pub/sub + stdout JSONL fallback) | 4 | P0 |
| 5 | Docker entrypoint-скрипт для `fl-client` (env → `flower-supernode --node-config`) | 1 | P0 |
| 6 | `build_strategy` принимает overrides из `run_config` (префикс `strategy-*`) | 2 | P1 |
| 7 | Связать `flwr_run_id` с `exp_name` в `summary.json` | 0.5 | P1 |
| 8 | Модель ≤1M для PlantVillage (MobileNetV3-small через `timm`) — если лимит жёсткий | 2 | P1 |
| 9 | Маппинг `dataset → [model, ...]` для веб-UI (в `DATASET_META` или отдельный реестр) | 0.5 | P1 |
| 10 | Поддержка production test-сета на сервере (альтернатива `part_dir/test`) | 2 | P2 |
| 11 | `fl_app/artifacts.save_final_model(arrays, path)` — вынести `torch.save` из `server_app` | 0.5 | P2 |
| 12 | `client-token` в `MetricRecord` (веб-id клиента в событиях, не partition-id) | 0.5 | P2 |

**Итого P0**: ~7.5 часов. **P0+P1**: ~12 часов. Укладывается в один рабочий день.

Очерёдность зависит от того, когда веб-сервис подойдёт к Дню 8 (события
раундов) и к Дню N (Docker-образ клиента).

---

## 11. Известные подводные камни Flower 1.28 (не в контракте, но важно знать)

1. **`ConfigRecord` не всегда пробрасывается через distributed pipeline** →
   все динамические значения клиент шлёт обратно через `MetricRecord`. Если
   что-то не доходит до сервера — смотрите в эту сторону.
2. **`InconsistentMessageReplies`** — Flower проверяет, что ответы клиентов
   имеют одинаковый набор ключей в `ArrayRecord`/`MetricRecord`. Если колбек
   добавит разные ключи разным клиентам — весь `aggregate_train` упадёт. У нас
   это решено в SCAFFOLD (`del rep.content["c_delta"]` до `super()`) и в
   `ProfilingStrategy` (не вызывает `super().aggregate_train()` вообще).
3. **`--node-config` с несколькими флагами** — argparse схлопывает до последнего.
   Все пары **в одном кавыченном аргументе через пробел**:
   `--node-config "partition-id=0 num-partitions=10"`.
4. **`pgrep -x flower-superlink` не работает** — процесс числится как `python3`.
   Использовать `pgrep -f flower-superlink`.
5. **SuperNodes подключаются к внутреннему IP SuperLink** (`10.10.0.30:9092` в
   нашей инфре), не внешнему — 9092 закрыт снаружи. Для веб-сервиса это
   внутренняя деталь деплоя, не влияет на контракт, но знать стоит.
6. **`num_workers > 0` в DataLoader → Segfault** через fork в FL subprocess.
   У нас зашито `num_workers=0` во всех `TrainHParams` — не меняйте.
7. **SCAFFOLD несовместим с SGD+momentum** — дивергенция 5-6×/раунд. В нашем
   коде при `aggregation=scaffold` автоматически используется plain SGD
   (`momentum=0.0`, `scheduler=None`). Если веб-сервис будет выставлять
   `strategy-aggregation=scaffold` — работает из коробки.
8. **`flwr run` в stdout показывает только connect/disconnect**. Реальные логи
   обучения — через `flwr log <run-id> . <federation>`. Это отдельная команда,
   её нужно запускать параллельно `flwr run`.

---

## 12. Вопросы обратно к веб-сервису

1. **Лимит параметров модели для PlantVillage ≤1M** — жёсткий или ориентир?
   От ответа зависит, добавляем ли MobileNetV3-small.
2. **Формат `run_id` в событиях** — использовать `flwr run-id` (целое), наш
   `exp_name` (строка с хешем), или свой UUID от веб-сервиса?
3. **Канал событий — Redis или stdout?** Redis требует, чтобы `REDIS_URL`
   был в env процесса `flower-superlink` — это ваша зона (subprocess env).
   Оба канала реализуем, первичный настраивается env-переменной.
4. **Параметры стратегии через `--run-config`** — согласны на префикс
   `strategy-*` (например `strategy-proximal-mu=0.05`)?
5. **Data volumes**: HF Dataset на диске или imagefolder для production?
   Оба поддержаны, но imagefolder проще для «накидали jpg в папки и поехали».
6. **`class_names` для PlantVillage** — у нас сейчас в коде захардкожено
   `num_classes=38` в `MODEL_REGISTRY["efficientnet_b0"]`. Имена классов берём
   из `manifest.json` партиции. Какие имена будут у PlantVillage на стороне
   веб-сервиса — из HF Hub (`nelorth/oxford-flowers`?) или свои? Это влияет на
   то, что положить в `DATASET_META["plantvillage"]["class_names"]`.

---

## Приложение A: структура `run_config`, которую видит ServerApp/ClientApp

Полный список ключей, которые читает наш код. Веб-сервису нужно обеспечить
их в `pyproject.toml` или через `flwr run --run-config`.

```toml
[tool.flwr.app.config]
# Режим
data-mode = "research"           # research | production
experiments-dir = "experiments"  # где хранить артефакты

# Research-only
partition-name = "cifar100__iid__n10__s42"

# Модель и стратегия
model       = "wrn_16_4"
aggregation = "fedavg"

# FL параметры
num-server-rounds   = 20
fraction-train      = 1.0
min-train-nodes     = 10
min-available-nodes = 10
local-epochs        = 3
data-dir            = "data/"

# Опциональные
enable-profiling = "false"
adaptive-mode    = "maximize-epochs"
lr-decay         = "none"        # none | cosine | step
server-mode      = "disabled"    # disabled | shared | exclusive
```

Для production веб-сервиса минимум:
```toml
data-mode = "production"
model = "efficientnet_b0"       # или другая
aggregation = "fedavg"
num-server-rounds = 20
min-train-nodes = <число клиентов>
min-available-nodes = <число клиентов>
```

И `--node-config "data-dir=/абсолютный/путь"` при старте каждого `flower-supernode`.
