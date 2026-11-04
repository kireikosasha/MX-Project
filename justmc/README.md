# Millennium (RNN) для JustMC

Порт RNN-части библиотеки Millennium (версия 5) из MX на код JustMC ([JMCC](https://github.com/Deolure/JMCC)).
Millennium 4.5 не переносился.

Перенесено: `RNNModelML` (checkData, learnByData, trainBatch, trainEpochs с подбором порога,
ROC/PR-AUC, лучшим чекпоинтом и final-fit), препроцессоры RAW / STATISTICAL / HYBRID,
StackedBiLSTM + LayerNorm, пулинги LAST / MEAN / MAX / ATTENTION, BinaryHead, AdamW,
формат весов `m1-rnn.dat` (через JSON).

## Файлы

```
build/millennium_trainer.json   модуль: библиотека + команды ~ml
build/millennium_mx_demo.json   модуль: то же + демо проверки аима (AimMLCheck)
weights/quark-e.json            предобученная модель MX (m1-rnn.dat): манифест
weights/quark-e-000..071.json   её части
src/                            исходники JMCC
tools/build.py                  сборка модулей
tools/ml_convert.py             конвертер весов .dat <-> .json
```

## Установка

1. Залить нужный `.json` из `build/` на любой хостинг и загрузить в мир:
   ```
   /module loadUrl force <ссылка>
   ```
   затем `/play`.
2. Загрузить веса или создать новую модель:
   - `~ml model import <ссылка на quark-e.json>` — манифест, части скачаются сами
     (JustMC обрезает ответ сайта на 25000 символов, поэтому веса разбиты на файлы до 24000);
   - `~ml model reset` — случайные веса.
3. Проверка: `~ml selftest`. С весами quark-e должно быть `semantics ok: 1` и
   вероятность `0.942535907508501`.

## Команды

Доступны владельцу мира и админам (`~ml admin add <ник>`). Префикс `~`, потому что `!` в JustMC
пишет в глобальный чат. Меняется константами `ML_CMD_PREFIX` и `MX_CMD_PREFIX`.

| Команда | |
|---|---|
| `~ml info` | состояние модели и датасета |
| `~ml log` | лог обучения в чат |
| `~ml param info` | гиперпараметры |
| `~ml param <lr\|dr\|rdr\|wd\|gc\|ls\|th\|bs\|im\|pm> <значение>` | изменить параметр |
| `~ml train <эпохи>` / `~ml stop` | обучение в фоне / остановка с откатом весов |
| `~ml dataset info\|clear\|undo` | датасет |
| `~ml model reset` | новая модель |
| `~ml model import <url>` | загрузка модели: один JSON или манифест со списком частей |
| `~ml model export <url>` | отправка модели частями, каждая отдельным POST-запросом |
| `~ml model save\|load\|delete <имя>`, `~ml model list` | несколько моделей в save-переменных мира |
| `~ml selftest` | самопроверка: 4 строки, без ошибок, если всё в порядке |
| `~ml selftest full` | то же плюс 20 проб и трассировка каждой стадии модели |

Демо (`millennium_mx_demo.json`):

| Команда | |
|---|---|
| `~mx dataset <legit\|cheat\|off> <игрок\|all>` | запись сэмплов по 600 поворотов в бою |
| `~mx debug` | показывать вероятность каждой проверки |
| `~mx vl` | VL игроков |

Демо проверяет каждые 150 поворотов в бою: `> 0.6` UNUSUAL, `> 0.7` STRANGE, `> 0.8` SUSPECTED.
Наказание при достижении лимита VL — функция `mx_on_punish`.

## Использование из своего кода

```ts
import "./ml_admin.jc";                  // или millennium.jc без команд

event<world_start> { ml_admin_boot(); }  // или ml_boot()

ml_dataset_add(a, b, 1);                 // сэмпл: два списка чисел и метка 1 / 0
ml_train_async(16);                      // trainEpochs в фоновом процессе
p = ml_predict(a, b);                    // вероятность класса 1
if (p.greater(ml_threshold())) { ... }
ml_learn(a, b, 1);                       // один шаг обучения (learnByData)
```

Для одного сигнала передавайте один список дважды. Значения ожидаются порядка ±100
(RAW делит на 100 и берёт tanh). При длине от 40 HYBRID считает статистические признаки
по окнам.

Модель хранится в save-переменных `ml.*` и переживает перезапуск мира.

## Перенос весов между JustMC и MX

```bash
python tools/ml_convert.py dat2json m1-rnn.dat weights/model.json         # MX -> манифест + части
python tools/ml_convert.py json2dat m1-rnn.dat "export/*.json"            # части из export -> plugins/MX/models/
python tools/ml_convert.py info "weights/model-*.json"
```

Манифест и части должны лежать в одной папке на хостинге: имена частей в манифесте
берутся относительно его адреса.

## Нагрузка

| Операция | Действий JustMC |
|---|---|
| проверка 150 поворотов | ~80 тыс. |
| forward + backward чанка из 150 | ~170 тыс. |
| шаг AdamW по 81 266 весам (раз на батч) | ~2 млн |

Тяжёлые части работают в процессах и уступают тик каждые `ml.budget` действий
(по умолчанию 4000), а также пока загрузка CPU мира выше `ml.cpu_max` (75%).
Для быстрых экспериментов можно уменьшить сеть: `ml.hid` = 16, `ml.layers` = 1, затем
`~ml model reset`.

## Особенности JustMC, которые учтены в коде

- Сложение, вычитание, умножение и деление округляют результат до 9 знаков после запятой,
  а `pow` и «Математическое ожидание» считают точно. Поэтому активации, LayerNorm, обратный
  проход и AdamW написаны через `mathematical_expectation` и `pow` (`mli_mul`, `mli_add2`,
  `mli_sub2`, `mli_inv` в `ml_base.jc`). Вывод совпадает с MX до ~1e-7, веса после шага
  обучения — до ~1e-5 (в симуляторе с таким же округлением).
- Список, записанный как `[переменная]`, хранит не число, а обёртку, которую
  «Математическое ожидание» не читает. Все списки собираются добавлением элементов
  (`mli_cons`).
- Ответ веб-запроса обрезается на 25000 символах: веса разбиты на файлы по 24000 с манифестом.
- `~ml selftest full` печатает 20 проб и трассировку прохода по стадиям; по ним видно, что именно
  сломалось, если сервер изменит поведение.

## Сборка

```bash
python tools/build.py --jmcc <папка JMCC>
```

`build.py` компилирует оба модуля, проставляет функциям иконки и описания
(`tools/function_meta.py`) и проверяет результат (`tools/check_build.py`): каждая функция
должна помещаться в одну строку кода JustMC (43 блока), без ранних выходов и необработанных
вложенных выражений.
