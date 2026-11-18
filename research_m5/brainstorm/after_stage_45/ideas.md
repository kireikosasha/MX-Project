# Мозговой штурм после этапа 45: сырая запись

Шесть взглядов (развитие ребёнка; «новое — забытое старое»; одна машина вместо трёх; читатель; рекуррентная глубина; рецензент-противник), по четыре идеи от каждого, три судьи (общность, выполнимость, доказательность), сведение в план. Текст агентов как есть, на том языке, на каком он написан; пути к скриптам заменены на пути в этой папке. Сводка и план — в README, этап 46.

## Диагнозы

### cognition

Seen through developmental psychology, the project has already rebuilt several of Spelke's core systems as exact machines: object identity (named exact memory, which is close to object files and FINSTs), number (the integer register and amounts), space (the Z^n register) and change of state (latest-value records). It also has Piaget's reversibility (going against a link takes the step back, and the loop conserves mass) and a cross-situational word learner of Siskind's kind (closure: two paths must agree). Three things that make the human ground productive are missing. First, there is no small closed basis from which these systems are combined, so a family that is a combination (heading x position, generation x gender) needs a new hand-written machine. Second, meaning is learnt per whole sentence kind (`text.view`), not per word composed by grammar, so nothing carries across families through language (no "time is space"). Third, nothing persists between datasets: `one.py` starts from zero every time, so the only learner across families is the designer, and that is the circle. A measurement I made (scratchpad/cog/kin_cascade3.py on the CLUTRR 1.0 graph data) suggests the machines are points of one tiny form space rather than separate inventions. The kinship table that `Tabled` learns as an opaque table is reproduced by a cascade of two up/down generation counters, where one direction of travel cancels and the other does not (the bicyclic monoid), plus two spouse bits and a gender flip-flop: 22 states. With answer names counted from 2-3-link chains, it answers 4-10-link chains at 95.0-98.3% and states nothing new. On the 3-link tests of the other splits it gets only 39-52%, and all misses fall in four in-law compositions: some are truly ambiguous (which side of the family), and in some CLUTRR 1.0 calls a man's wife's mother "mother". The register (Z^2) and this kinship cascade are the two primitives of Krohn-Rhodes theory, groups/counters and flip-flops. So the way out is a closed generative basis, plus words as operators, plus logic as lifts every form inherits, plus a cumulative prior measured on families written by others. Order: idea 1, then 2, 3 and 4, with the protocol of idea 4 frozen from day one.

### forgotten-old

Seen through "the new is the forgotten old", the three machines are three hand-picked points in two spaces that classical theory has already mapped completely. First: what a story does to a state over time is a transformation semigroup. By Krohn–Rhodes (1965), every finite machine is a cascade of two kinds of primitive cells: groups (the register's counters, cyclic counters, permutations) and resets or flip-flops (the recall machine's "latest value holds"; a pointer step in loop.py is a reset driven by memory). Second: "what is X to Y" is a relation algebra (Tarski 1941, Allen 1983). The table is its composition table, and reading without answers is path consistency (Montanari 1974). The circle exists because the outer loop that Soar/GPS put inside the system runs in my head: I notice an unexplained kind of question, diagnose which structure is missing, write it, and keep it. The way out is not another machine. It is (i) making both spaces grammars that the existing method (soft choice, harden, polish, pick by carry-over) searches, (ii) making the diagnosis mechanical through algebraic property tests and minimal-pair asking, and (iii) keeping what was found as reusable pieces (chunks). I checked half of this on CPU (scripts in brainstorm/after_stage_45/scripts/kr/). Over a 14-cell grammar, the correct cell for six families that none of the three machines covers (Z7 weekdays, S3 shell game, S3 rotate+flip seats, Z2 lies, a 4-value reset, a sign counter) was picked from 30 stories of 1–3 steps and is 100% exact at length 200, in about 15 s each; the register alone gets 14–52% at length 50 on the five non-counter families. But a dependent two-cell cascade (a turtle whose step depends on its heading, i.e. BBH navigate) was not found. The best result was 87% carry-over and at most 15% at length 100, and even with the true heading cell given, walk.py's register gradient fits only 98.5% of 400 stories. So the frontier is composition from answers alone, not missing cell types. Frames and scripts (Schank) and case-based reasoning matter for the reader and for bAbI 20, not for writing machines.

### unify

Четыре машины на деле один механизм, просто в четырёх точках. Каждая сворачивает последовательность событий в маленькой алгебре, а потом выдаёт ответ через выученную таблицу. Счётчик (`walk.Walk`, `one.Register`) сворачивает путь между двумя вещами в Zⁿ, и ответ берётся по знакам. Таблица (`walk.Table`, `one.Tabled`) сворачивает тот же путь в выученной конечной таблице композиции. Петля (`loop.Machine`) сворачивает орбиту указателя («идти по полю, пока не встретится вид») в (N, +1 / +значение / max). Вспоминание (`recall.py`) сворачивает рассказ по времени в «последнее значение», множество и Z-количество, с ключом по вещи (это каскад) и срезами по времени, а вопрос там — композиция GET/INV/замыкание/счёт. Круг возникает потому, что каждая машина зашивает руками тройку «какая последовательность × какая алгебра × какой выход» как монолитный класс. one.py выбирает между классами, но не собирает тройку сам, так что новая тройка или одна лишняя композиция требует от меня нового класса. Я это замерил. Три задачи bAbI, которых one.py не берёт (10, 16 и 20: 42,6, 38,0 и 80,9 %), оказались программами глубины ≤3 в таком языке: 98,2, 100 и 100 % со 150 вопросов, перебрано от 3 до 63 программ, меньше секунды. Не хватало композиции, а не машины. Счётчик внутри таблицы CLUTRR находится без градиента: это одномерное ядро уравнений композиции, то есть поколение (отец +1, дед +2, сын −1, брат и муж 0). Оно выполняется на 9074 из 9074 учебных рассказов и находится уже по 30. Выйти из круга значит сделать пространство машин порождаемым, то есть типизированным языком примерно из 15 команд, полнота конечной части которого гарантирована теоремой Крона–Роудса. Структуру (градуировки, конгруэнции) при этом выводить из выученной таблицы алгеброй, а программу искать перебором, выборы внутри неё подгонять градиентом и проверять точно на переносе с коротких рассказов на длинные. Тогда новую машину строит поиск, а не я.

### reader

From the reader's side the circle looks like this. The only part with weights gets rebuilt for every machine and every dataset: templates for recall, a bag-of-words GRU that outputs register offsets, and another that outputs answer words. Each is trained from scratch on a whole-sentence view that almost never recurs (4.7% of CLUTRR test sentences occur in teaching), and its output space is the machine's own choices. So nothing a reader learns about English carries over, and every new machine needs a new reader.

The project's own numbers say capacity is not the limit (16x width gave +1.5 points in stage 41). Exposure and structure are: 5x more authors gave +15, RoBERTa's pretraining +10 link points, and consistency over unlabeled text +6 on CLUTRR and 59->100% on StepGame. The null result "own readings returned as lessons" is the expected one: lessons recycled from the same answers on the same stories carry no information the answers did not already have.

Three measurements I ran today (CPU, scripts in brainstorm/after_stage_45/scripts/reader_lens/):
- **Path counting with no trained weights.** With spaCy's 12 MB parser and nothing trained, the kin word lies on or hangs off the dependency path between the two names in 80.6% of test links, and in 98% of teaching links the meaning is that word or its inverse. Path signatures of the test recur in teaching for 58% of links, against 4.7% for whole sentences. A counting lexicon over them gets the kind of kin right on 80.6% of one-sentence links, the small reader's level.
- **Sex from rules.** Crude pronoun and apposition rules read the sex of 89.6% of people, and are right 96.9% of the time where they read it.
- **A small pretrained reader.** BERT-mini (11M weights) put in the existing text.Encoder slot reads links at 84.4% (85.9% when the same sentence is read the same everywhere) and answers 73.9/77.8%. For comparison, five scratch readers of 0.5M weights give 80.5% and 69.5/73.6%, and RoBERTa-base gives 90.1% and 78.4/81.8%. So 1/11 of the weights buys about half of RoBERTa's gain. This was one seed on the working set, already opened many times.

So the reader should be factored like the rest of the project:
- **Grammar** in one shared, small module with weights: who relates to whom, by which words, on which side, and coreference.
- **Word meanings** as rows of an exact lexicon that keeps growing; this is the owner's external base.
- **Exact invariants** that teach from unlabeled text: one sex per person, inverses, uniqueness, closed cycles.
- **The size of the grammar module** should be measured, not set at 0.2B in advance.

### recurrent

From the recurrent-depth angle, the circle has one cause. Every exact machine so far is a hand-picked point in a small, known space. Most are a monoid acting on a value along a pointer path through exact memory: the register is integer counters, the table is a finite relation monoid, recall is resets keyed by thing, and loop.py is a pointer plus counter/max. The rest is a monotone fixed point over a finite lattice (closure reading). So each benchmark is won by me choosing the point, not by the system searching the space.

Huginn, Ouro, HRM and TRM take the opposite extreme. A free latent vector has to learn both the machine and its own stability; the 2026 looped-LM papers, e.g. STARS (arXiv 2605.26733, read at abstract level only), are about restoring fixed points after the fact. That costs data and breaks outside the training distribution.

The join I propose: the recurrent core carries only exact state, and the network only proposes. The state is either a distribution over finite-automaton states moved by row-stochastic matrices (mass conserved, as in loop.py), or option sets that can only shrink (monotone, so a fixed point is guaranteed by Knaster–Tarski). The network proposes transitions, branches and program edits, which are hardened and checked exactly; this is the AlphaProof pattern at every scale.

Two things measured here on CPU (scripts in brainstorm/after_stage_45/scripts/kr/):
- One soft-automaton learner with an optional counter, taught on lengths 1–10, chose its own size. It recovered parity, mod 5, cycle, latest value, S3, first=last, majority and count-since-reset, each exact at length 500. It saw on its own teaching data when a counter was needed.
- Sudoku-Extreme and Maze-Hard both fell to mined invariants plus monotone propagation and search, with no network at all. Sudoku: 810/810 constraints from 100 solutions, 3000/3000 test puzzles. Maze: the dataset's tie-break convention was found among 192 candidates, 1000/1000 test mazes. TRM gets 87.4% and 85.3%.

So those two benchmarks can test stability but not generality. ARC-AGI-1 is the only one of the three where no per-task machine can be hand-written, and it is the right Colab yardstick against TRM (44.6%, 7M weights) and HRM (40.3% claimed, 32% verified by ARC Prize on the semi-private set). ARC-AGI-2 is a stretch (TRM 7.8%, the 2025 Kaggle winner NVARC 24.03%). ARC-AGI-3 is interactive and not a Colab tiny-model test.

### red-team

Для рецензента круг — это не нехватка частей. Пространство гипотез пишет проектировщик, и пишет после того, как увидел формат набора. Счётчик в one.py — это два целых числа (`--numbers 2`) с шагами из {-1,0,1} (`walk.steps_of`) и ответом по знакам, то есть ровно порождающая модель StepGame (и целочисленный TransE 2013 года). Таблица — это таблица композиции со словами ответов и четырьмя запасными. Вспоминание читает шаблоны bAbI, переходник отмечает вещи по формату каждого файла, а «выбор машины по переносу» выбирает из трёх машин, каждая из которых написана под один из тех самых наборов, на которых её меряют. По числам рецензент скажет следующее. Базы старые и удобные (MemN2N 2015, TP-MANN 2022, GAT/CTP из статьи 2020), а сильные соперники не названы: ILP Mitra и Baral 2016 — 99,68 % на bAbI; R5 (ICLR 2022) — по чужому пересказу 96–99 % на 7–10 звеньях CLUTRR, не проверено; связка GPT-3+ASP (Yang, Ishay, Lee, 2023) — одна программа для bAbI, StepGame, CLUTRR и gSCAN. Свои трансформеры-соперники сломаны (18 % при k=1 против 45 % у UT в статье). На StepGame одна правка сделана после теста (65 -> 95 % при k=10). Преимущество над DSR-LM по одним ответам держится на списке слов родства: внутри one.py 62,7 против 61,0±2,6, то есть ничья. bAbI 17/19 учились на 1000 вопросов, остальные задачи — на 30–40 рассказах. Чтение без ответов транзитивно по тесту и требует около 39 тыс. избыточных рассказов (с 2 тыс. прочитано 7 видов из 635). «Слепые» наборы написала одна модель (Haiku) по брифу проектировщика (не больше 40 слов, 1–3 шаблона на факт), и восемь раз подряд они дают 86–92 % при примерно двух новых видах провала на набор. Dyna-bAbI, сделанный именно чтобы ломать такие системы, советовали на этапах 22 и 32, но так и не запустили. Там, где чтение не шаблонное (текст CLUTRR на A100), верных ответов 76 % при k=2 и 45 % при k=10: «перенос на любую длину устройством» верен только при почти безошибочном чтении. Общую архитектуру доказала бы только такая проверка: семейства, скрытые от проектировщика (их генерирует третья сторона по спецификации, запечатанной до прогона), замороженный код с хешем коммита, пороги опровержения записаны до прогона, один выстрел, все зёрна. Главное число — сколько строк рукописного кода понадобилось на новое семейство, а не проценты. Состав: bAbI/StepGame/CLUTRR как рабочие, затем Dyna-bAbI, ProofWriter (глубина 5 при учении на 3 и меньше), SCAN/COGS (сами по себе ничего не доказывают: нейросимвольные системы берут SCAN на 100 % с 2020 года, по памяти), CLRS-30 вне обученного размера, SVAMP/GSM8K, и ARC-AGI-1 как честный контроль «ноль без новой машины» (TRM на 7 млн весов — 45 %). Защитимо: «при небольшой библиотеке точных типов состояния учение выбором сквозь точное исполнение даёт перенос на глубину x5–x10 по построению, на один-два порядка меньше данных и вычислений, калиброванное "я выучил" (99,5 % там, где сказано) и чтение новых формулировок без ответов в избыточном тексте». Незащитимо: «бьёт фронтирные LLM», «общая архитектура как у человека», «логика выучена с нуля» (она написана руками) и «0,2 млрд хватит» (вне CLUTRR не мерилось). Выход из круга с этой точки зрения — сделать единицей измерения число правок человека на новое семейство, а единицей механизма — систему, которая сама пишет себе машину (тип состояния и операторы) и сама говорит, когда не может.

## Идеи по сумме оценок судей

### 1. One grammar of state machines: Krohn–Rhodes cells and cascades, learnt by the project's own soft-choice method (forgotten-old-1, сумма 16.5, недель 3)

**Что.** A module cells.py replaces the per-family state machines with a grammar of cells. (a) Counter over Z, read by sign: exactly one number of today's Register. (b) Cyclic Z_m, m up to 12: shifts and resets. (c) Permutation S_k on up to 5 slots, plus resets. (d) Reset/flip-flop over v values, or over an entity pointer: recall.py's 'latest value', and loop.py's step along a field. A machine is a cascade of up to 3 cells. A downstream cell's action is chosen per (symbol, upstream state), which is the wreath product that makes the turtle and the calendar odometer possible. A symbol is what one.py already produces: a sentence kind from the reader, or a link kind with its sign. Each cell's action per symbol is a softmax over transformation matrices, as Register in one.py does over its 9 offsets. Then harden, then walk.Walk.polish, then recount the readout. Entity-indexed banks of cascades (one per thing) give recall-style memory. The choice of machine plugs into one.py's existing selection: fit on short ways, then carry-over to longer ways, then fewest states. The register becomes cascade(counter, counter), and StepGame and bAbI 17/18 must come out unchanged. Test whether CLUTRR kin comes out as counter(generation) plus reset(gender of last link) plus a small reset, with no hand-written table.

**Первый опыт.** Already run here (CPU, minutes): cascade.py finds the single cells, 6/6 families, 100% at length 200, from 30 stories. turtle.py, upstream2.py and downstream.py show that the dependent cascade fails. Next, 2–3 days on CPU: make cascades learnable with three evidence sources. (a) Enumerate the upstream cell exactly (8^|symbols| options, pruned by state relabelling) and fit the downstream with walk.py's register gradient from several starts. (b) Use intermediate questions where the data have them (bAbI asks between statements), as dense supervision. (c) L*-style asking, reusing the stage-13 mechanism: build the shortest story on which the top-2 cascades disagree and ask only its answer. Families: turtle/navigate; odometer (weekday Z7 with week carry); CLUTRR links via walk.py --clutrr (teach on 2–3 links, test on 10); the six single-cell families. Then an external battery: BIG-Bench Hard web_of_lies (Z2), navigate (heading cascade), tracking_shuffled_objects_3/5/7 (S_n), date_understanding (calendar cascade). Negative controls: dyck_languages and boolean_expressions. About 40 examples to teach, the rest to test, with longer variants generated from the same templates. Task contents are from memory; check the files.

**Что опровергает.** Named beforehand. With (a)+(c), turtle and odometer must reach at least 99% at length 100 from at most 400 short stories plus at most 40 asked. On BBH web_of_lies, navigate and tracking_shuffled_objects_5, at least 95% with at most 50 teaching examples. On dyck/boolean, 'not sure' on at least 90% of questions. Kill it if turtle stays below 90% at length 100 with 40 asked stories. Also kill it if one.py with cells in place of Register and Table loses more than 2 points on StepGame k=10 or on bAbI 17/19, or if CLUTRR links at k=10 fall below 95% without the hand table.

**Судьи.** generality 5.5: It is the only cascade idea that faces the measured bottleneck: dependent cascades are not identifiable from final answers (turtle fails). It proposes exact enumeration of the upstream cell, intermediate questions and L*-style asking, and it expects 'not mine' on dyck/boolean. But the BBH battery is chosen because those tasks are known cascades, which selects the families to fit the grammar. The grammar itself is still designer-made. / feasibility 6: This is the most concrete plan aimed at the measured blocker: a hidden upstream state cannot be identified from final answers alone. Exact enumeration of the upstream cell is cheap when there are few symbols (the turtle has 4), and L*-style asking reuses stage 13. The BBH tasks are templated and need small readers. The turtle and odometer at 99% look reachable in 3 weeks. Everything at once (BBH at 95% plus not-sure on Dyck plus no regression and CLUTRR without the hand table) is a lot. / evidence 5: The BBH tasks (web_of_lies, navigate, tracking_shuffled_objects, date_understanding) are public, and dyck and boolean_expressions are good negative controls for 'not sure'. The Krohn-Rhodes, Liu et al. 2023 and Merrill et al. 2024 references are real. Leakage: the designer chose exactly the BBH tasks that map to Z2, heading and S_n cells, and the 'longer variants from the same templates' are self-generated. Reading BBH text with 40 examples will need template-aware reading. The idea itself reports that learning dependent cascades from answers failed, which weakens the chance of a positive result but not its value.

### 2. One learnt finite-state register (a Krohn–Rhodes cascade) in place of Register, Tabled and Recalled (recurrent-1, сумма 16, недель 3)

**Что.** A new module, cascade.py, with a class Cascade made of components.
- **Finite component:** a soft automaton with one row-stochastic S×S matrix per input symbol (the symbol is the link kind the reader gives). Mass is carried as in loop.py's Machine.forward. A later component may read the hardened states of earlier ones (cascade/wreath product).
- **Counter component:** an integer whose move per (symbol, state of the earlier components) is chosen from -1, 0, +1 or reset. These are walk.Walk's offsets plus loop.py's register ops.
- **Readout:** a learnt choice between a table from (states, signs of counters) and the counter value itself.

The pointer side stays as it is. walk.way and text.together pick the path through exact memory; recall.py's keyed records give one cascade instance per thing (latest value = reset, how many = counter, chain end = pointer).

Teaching uses the project's own recipe: soft, gradient through the fold, entropy push late, harden, then polish every hard choice exactly (as Walk.polish), with several beginnings.

Choosing the machine (one.py step 5) becomes growing the cascade: 1, 2, 3, 4, 6, 8, 12 states, then add a counter, then a second component. Growth stops at the first one whose hardened form explains every teaching story. Ties are broken by short-to-long transfer, the criterion one.py already uses.

text.chart (the soft table averaged over all bracketings) is no longer needed: a left fold of a learnt action needs no brackets.

Register.close generalises. 'Two ways between the same things give the same transformation' is solved for an unknown sentence kind by trying the transformations of the learnt monoid, just as close() now votes over offs. So reading without answers would work for the kin table too, not only for the counter.

**Первый опыт.** The proof of mechanism is already done on CPU: automaton.py and cascade.py in brainstorm/after_stage_45/scripts/kr/; results in out_*.txt and cas_*.txt there; about 10 minutes on 4 cores.

Next step, CPU, 2–3 days:
1. Put Cascade into one.py behind a flag, replacing Register and Tabled. Use Recalled's value tracking via one keyed instance per thing.
2. Rerun the one.py sets with the same commands as results/one: bAbI 17/18/19, StepGame k=1..10, CLUTRR 1.0 links, CLUTRR 1.3, and all 20 bAbI tasks. Compare set by set with the hand machines.
3. Run Delétang's public generator (regular + counter tasks, longer test lengths) and Liu et al.'s S5 / permutation tasks.

Everything runs in minutes per task on CPU; no GPU needed.

**Что опровергает.** Any of these kills the idea:
- On any of the one.py sets the single cascade is more than 2 points below the hand-written machine.
- Its short-to-long choice disagrees with one.py's current machine choice on 2 or more sets.
- Any Delétang regular task stays below 99% at the longest test length with 8 beginnings.
- S5 (120 states) is not learnt in 30 CPU-minutes. Then the relaxation does not scale, and the cascade must be built from small factors; that would need to be shown before claiming generality.

**Судьи.** generality 5: The mechanism is measured. One learner finds minimal automata exact at length 500, and it shows on the teaching data that majority needs a counter, so the system notices its own missing component. Delétang's generator gives external, frozen-code tests. But the falsifier is mostly regression against one.py, cascade identifiability on stories is not addressed, and the reader is assumed. / feasibility 6: The mechanism is already shown on a CPU (exact at length 500 with minimal states, and the counter appears when needed). Delétang's regular and counter tasks need no reader and give a quick external number. The strict falsifier (within 2 points of every hand machine, CLUTRR from 2-3 links, S5 in 30 CPU-min) is likely to break somewhere: finite control learnt by gradient from short stories has been measured to under-identify. A partial convincing result fits in 3 weeks. / evidence 5: It uses public generators (Delétang et al., ICLR 2023; Liu et al. S5) with published neural baselines, and the falsifiers are precise. Experts will find exact length generalisation on regular tasks by a finite-automaton learner unsurprising, since it is the classic Giles/Omlin and RPNI/L* line. Delétang's suite has regular, deterministic context-free and context-sensitive tasks, not a separate 'counter' class, so that part of the plan needs fixing. The one.py consolidation results stay on sets the designer has seen.

### 3. Ретро-слепая проверка: переизобретёт ли система заплаты самого проектировщика (обучение библиотеки вместо дописывания частей) (red-team-1, сумма 16, недель 3)

**Что.** Механизм — автоматический внешний цикл (новый invent.py поверх records.py/rows.py/recall.py), который делает то, что сейчас делаю я. (1) Срабатывание: калиброванный сигнал «не объяснил вид вопроса / не прочёл вид предложения», он уже печатается. (2) Выразимость: перечисление программ из операторов algebra.py до глубины 4–5 по хранилищам. Прецедент уже есть: --deep собрал недостающую часть для H3 за 9 с, а на этапе 23 объяснено 21 вид из 22. (3) Обучение библиотеки: программы (прочтения и запросы) всех задач, решённых на данном коммите, сжимаются антиунификацией в не больше чем 20 новых абстракций. Например, «крайний из множества по ключу», написанный руками четырнадцать раз, должен выпасть одной абстракцией. (4) Принятие по одной оценке в битах: длина библиотеки, плюс длина программ всех задач, плюс биты необъяснённых ответов. Абстракция принимается, только если переносится на эпизоды в 3 и 10 раз длиннее и не роняет 104 рабочие задачи. Оценка в два шага. Сначала откат по git: эталоном служат около 12 коммитов, где часть добавлена после слепого прогона: 7d790a7 (крайний порядка, B3/C3/E3), 45722bc (bAbI-14), 62098af (bAbI 3/17/19), 15656cf (событие как вещь, D6/A6), bb0bf77 (после G), cefc224 (противоположные слова, I5), b084029 (все порядки событий, H6), 9b44fe9 (J4/J10), 8fddfe6 (K1), 7147602 (обмен и «значение значения» после L), 5f04428. Для каждого коммита берётся рабочая копия на родителе, цикл запускается на задачах, ради которых делалась правка, и сравнивается с исправленным кодом. Затем вперёд вслепую: цикл замораживается, и подряд прогоняются Dyna-bAbI, ProofWriter (глубина 5 при учении на 3 и меньше) и два новых слепых набора, без единой правки.

**Первый опыт.** Только процессор. Шаг 0 (полдня): вручную разметить 12 заплат на «выразима композицией операторов, существовавших у родителя» и «новый тип состояния» (например, миры событий во всех порядках для H6). Шаг 1 (около суток на 4 ядрах): 12 коммитов по 1–3 мотивирующие задачи, бюджет не больше 30 CPU-минут на задачу. Заплата считается восстановленной, если тест на её задачах не дальше 2 пунктов от исправленного кода, а 104 рабочие задачи не падают больше чем на 0,5 пункта. Шаг 2 (2–3 дня): замороженный цикл на Dyna-bAbI (20–40 рассказов на задачу) с метрикой «покрытие при точности не ниже 99 %» против замороженного кода без цикла.

**Что опровергает.** Записать до прогона: восстановлено 3 заплаты из 12 или меньше (проходной порог — 6 из 12). Или на Dyna-bAbI покрытие при точности 99 % с циклом не выше, чем без него. Или восстанавливаются только заплаты, помеченные на шаге 0 как «композиция»: тогда круг наполовину закрываем, а наполовину это новые типы состояния, и этот вывод надо записать.

**Судьи.** generality 7: This measures the criterion directly. A git rollback over 12 real patches made after blind failures asks whether the system re-invents what the designer wrote. Then a frozen loop runs on Dyna-bAbI and ProofWriter and counts human edits per family, adapter edits included. The mechanism is the one known route out (enumeration, library compression, MDL acceptance). The leak (algebra.py operators were written after sets A-H) is stated, and step 0 splits 'composition' patches from 'new state type' patches. Likely result: only the composition half is recovered. That is still the most honest number any idea offers. / feasibility 5: Either outcome is informative, and the git ground truth already exists, so a clean negative still counts. But it needs an anti-unification library learner, runs on parent commits, and a Dyna-bAbI adapter that itself counts as an edit. Many patches are new state types that composition cannot recover. Reaching 6 of 12 is uncertain within 3 weeks. / evidence 4: Clever internal validity: the 12 commits are real (checked with git) and serve as an answer key that cannot be fitted after the fact, and Dyna-bAbI is a real public generator. The literature is real (DreamCoder, Stitch, Metagol, Plotkin anti-unification). But the algebra.py operators were written after sets A-H (admitted leakage), the metric is internal ('patches recovered'), and outside experts will not weigh results on the project's own git history.

### 4. Mother structures: one executor built from counters and flip-flops (the Krohn-Rhodes basis); the three machines become settings of a few law bits (cognition-1, сумма 15, недель 3)

**Что.** A new class `Cascade` in one.py (or cascade.py) replaces both `Register` and `Tabled`, and later the value and amount memories of recall.py.
- State: a bounded grid over components. Counters c_i in [-R,R] (R from the longest teaching path, as `far` is now). Rings Z_m with m from 2 to 12. Flags with 2-4 values.
- Meaning of a link view: a soft choice of one primitive action per component (counter -1/0/+1, ring step, flag keep/set v/toggle).
- Execution: p_{t+1} = sum_a b_t(a) T_a p_t, where T_a is a sparse deterministic transition on the grid. This is the same mass-conserving update as `Register` (its torch.roll is exactly T_a for an abelian counter) and the same fixed-point loop as loop.py.
- Going against a link: the inverse action where the component is a group, otherwise a learnt converse.
- Discrete law bits, chosen before soft training: (a) per counter pair, abelian (Z^2, as StepGame, bAbI 17) or bicyclic (up/down where down-then-up cancels and up-then-down does not, as in a tree; kinship); (b) wreath bit: a counter's increment depends on a flag or ring (heading, then forward).
- Readout: a soft table from a projection of the state (signs, small exact values, flags) to answer words, as `says` is now.
- Search: presentations are enumerated from fewest states up. Each is taught (2 beginnings x 300 turns) on short-path stories and kept by carry-over to longer paths, the check `solve()` already uses; ties go to fewer states (Occam).
- Dual route, words and rules: compositions that often contradict the cascade are memorised, as `text.Known` memorises frequent views. This is how the CLUTRR 1.0 in-law convention would be absorbed.
- Growth by placeholder (Carey): if no presentation carries over, add one uninterpreted counter or flag and keep it only if it shortens the description of the answers.
- Then port recall.py's 'latest value' and 'how many' as cascades keyed by thing (object files), so all three machines of one.py are one.

**Первый опыт.** Days 1-3, CPU:
- Write `Cascade`.
- Regression on StepGame (/home/user/zhengxiangshi/stepgame, the one.py protocol), the CLUTRR graph version (/home/user/azreasoners/llm-asp/clutrr/data/clutrr/data_089907f8, edge_types: teach on 2-3 links, test 4-10) and bAbI 17/18/19 (1000 questions each).
- Expected: Z^2 picked on StepGame (95% or more at k=10), Z on bAbI 18, bicyclic x flags on CLUTRR (97% or more at k=10 with the dual route; 95-98% without, by my script), Z^2 with an ordered readout on bAbI 19.

Then freeze the code at a commit hash. Have the owner or outside LLM authors, given only theme names, write 5 never-coded families:
- turtle with turns
- weekdays/clock, 'n days later'
- floors with numerals
- seats around a round table
- ancestor yes/no
Each family: 500 teaching stories with at most 3 links, and 1000 test stories at 5/10/20 links. All of it takes about 1-2 h on CPU.

Colab A100, 30 min per family: a decoder-only transformer trained with a known recipe as the rival. Report it only if it reaches 95% or more at teaching length (rival.py is known broken).

**Что опровергает.** Kill if any of these holds:
- with frozen code, fewer than 3 of the 5 new families reach 95% at 10 links when taught on at most 3;
- the single executor falls below 92% at k=10 on StepGame (the register gets 95%);
- it falls below 95% at k=10 on the CLUTRR graph version (the table gets 100%).

**Судьи.** generality 5: It turns 'which machine' into choosing a point in a Krohn-Rhodes space by carry-over, and it tests on 5 families written by outside authors with frozen code, which is a real circle test. But it treats expressibility as solved and ignores the measured fact that the wreath product (turtle) and the finite control (CLUTRR K=6: 67.5%) are not learnt from answers. It assumes the reader hands it symbols, and anything outside the basis goes back to the designer. / feasibility 6: The single cells are already exact (kr/automaton.py and cascade.py: 6/6 families at length 500), and the regression sets are on disk. But the turtle needs a wreath product, and learning that from answers alone has been measured to fail (turtle.py, downstream.py). The family list still makes 3 of 5 reachable: weekdays, round table, floors and ancestor are each one cell. Most of the time goes on finding outside authors and keeping the state grid small on a CPU. A partial win in 3 weeks is likely; the clean 'no new code' claim is not. / evidence 4: The Krohn-Rhodes completeness argument and the citations are real (Kemp & Tenenbaum 2008, Carey 2009, Piaget, Liu et al. 2023). But the five test themes (turtle, weekdays, floors, round table, ancestor) were picked by the designer as cascade-shaped, so outside authorship of the generators only partly removes the leakage. The regression sets are already-solved public toys, and the transformer rival is reported only if it works. Completeness of the basis also says nothing about whether a cascade can be learnt.

### 5. Logic as mental models: lifts that every form inherits (possibilities, rules run to a fixed point, kinds) (cognition-3, сумма 15, недель 2.5)

**Что.** Three general constructions over any cascade or keyed memory in recall.py and one.py, each written once:

(1) Possibility lift (subset construction). A component holds a set of values.
- 'Either A or B' sets {A,B}; 'not A' removes A.
- A question answers yes, no or maybe according to whether all, none or some of the worlds agree.
- A reading that empties every world is rejected. This is another teacher that needs no answers.

(2) Rules as wiring.
- 'If something is big it is red', 'all swans are white', 'thirsty -> kitchen' become learnt edges between attribute values.
- They are executed to a fixed point by the loop of loop.py: mass-conserving, with the answer as the fixed point, so depth generalisation comes by construction.
- Under the open-world reading, an underived attribute stays in the 'unknown' set, which is lift (1) again.
- 'Why' is the converse of a rule edge (bAbI 20, agents and goals).

(3) Kinds. A thing's attribute defaults to its kind's value, and members write to the kind (inheritance). This covers bAbI 16 and generics.

Meanings remain soft choices that are hardened and checked exactly.

**Первый опыт.** (a) bAbI 10, 15, 16 and 20 inside one.py's recall machine with the lifts, 30-40 stories each, CPU minutes. Current scores: 42.6, 100, 38.0 and 80.9%.

(b) ProofWriter: download the AllenAI release; open world, depth up to 5, plus birds-electricity and ParaRules.
- Teach on depth at most 2 (at most 2000 questions).
- Test at depth 3 and 5 and on the out-of-domain sets.
- The templated language goes through a rows.py-style reader; ParaRules goes through the idea-2 lexicon.
- CPU 1-3 h.
- Prediction: 98% or more at templated depth 5.

**Что опровергает.** Kill if bAbI 10 or 16 stays below 90% with at most 40 stories, or if ProofWriter depth-5 open-world accuracy is below 90% when taught on depth at most 2.

**Судьи.** generality 3: Possibility, rules-as-wiring and kinds are general constructions, but they are still hand-written parts added for bAbI 10/15/16/20 and ProofWriter. Variable binding will likely need yet another part. / feasibility 7: The bAbI 10/16/20 half is nearly done: tiny_isa.py already gets 98.2/100/100 from about 30 stories. ProofWriter depth-5 taught on depth 2 or less is a forward-chaining fixed point that the exact loop handles by construction, and its templated language is a modest reader job. It fits in 2-3 weeks on a CPU. The catch is that the ProofWriter reader may become one more hand-written part, and symbolic systems already saturate the templated sets, so the result is more feasible than striking. / evidence 5: ProofWriter is public, and teaching on depth 2 or less with depth-5, birds-electricity and ParaRules tests is a real out-of-distribution protocol. The literature is real: Johnson-Laird, van Emden & Kowalski, RuleTaker/ProofWriter. But templated ProofWriter falls to any symbolic prover once parsed, so a 'rows.py-style reader' is close to a hand-written parser, and experts will see 98% as parse plus prover, also matched by LLM+solver pipelines. Only ParaRules, read by a learnt lexicon, would be persuasive. bAbI 10 and 16 add little.

### 6. Маленький читатель и точный исполнитель на арифметике: лестница размеров, одни ответы, те же данные, что у соперника (red-team-4, сумма 15, недель 4)

**Что.** Непроверенная половина гипотезы — чтение. Арифметика — область, где исполнитель полон: любой ответ GSM8K — выражение над числами текста и несколькими константами. Значит, отдельную машину под задачу дописать нельзя, и вся новизна сидит в читателе. Читатель (тот же код, что слот --encoder в one.py) выдаёт типизированную программу на языке счетов этапа 3: ёмкости, передача, ставка, часть/целое, сравнение. Константы — из внешней таблицы (60 мин в часе, 7 дней в неделе, дюжина = 12); это и есть «внешняя база». Исполнение точное. Учение по одним ответам: перечисление согласных с ответом программ до глубины d с памятью найденных программ (MAPO), затем читатель учится на найденном. Чтение без ответов на неразмеченных задачах — аналог замыкания: целочисленность, неотрицательные количества, согласие единиц, подстановка ответа обратно. Никакого переходника под набор: числа выделяются только регулярным выражением. Оценка: лестница читателей 20M / 125M / 350M; учение на GSM8K train (7,5 тыс., только ответы) плюс MAWPS/ASDiv-A; тест — GSM8K test (1319) и SVAMP (1000). Контролируемый соперник — сеть того же размера, дообученная выдавать Python (PoT) на тех же 7,5 тыс. по тем же ответам; опубликованная точка — TinyGSM.

**Первый опыт.** Процессор, 2–3 дня: прежняя постановка этапа 2 (обучение на MAWPS+ASDiv-A, 3138 задач; тест — SVAMP), где читатель в 0,8M давал 68,5 %. Тот же конвейер с читателем на 20M и с RoBERTa-base — часы на процессоре, как на этапе 42; ожидания назвать до прогона. Затем Colab A100, 6–10 часов: GSM8K, лестница 20M/125M/350M x {читатель с исполнителем, PoT-соперник} x 3 зерна.

**Что опровергает.** При 125M и одних ответах GSM8K train: читатель с исполнителем не выше соперника PoT того же размера на тех же данных хотя бы на 10 пунктов. Или SVAMP ниже 80 % с RoBERTa-base. Или наклон лестницы говорит, что для 60 % на GSM8K нужно больше 1 млрд весов.

**Судьи.** generality 4: It sidesteps the circle by choosing a domain whose executor is already complete (arithmetic expressions), so no machine has to be written. It is a good test of the 0.2B-reader claim. But it is a PoT-style program reader and says nothing about who writes machines for new kinds of state. / feasibility 4: SVAMP with RoBERTa on the old stage-2 pipeline is a CPU job of a few days, and 80% would be striking. But GSM8K from answers only at 125M runs into search explosion at depth 4-8, a PoT rival trained on answers alone must be built, and contamination checks are needed. The GSM8K headline is unlikely in 4 weeks; the SVAMP half is possible. / evidence 7: Strongest by this criterion. It uses public benchmarks (GSM8K, SVAMP) where no per-task machine can be hand-written, and a matched rival: a PoT model of the same size on the same data. It also has a size ladder and falsifiers fixed in advance. The literature checks out: TinyGSM reports 63.1% at 125M without a verifier and 81.5% for 1.3B+1.3B. The bar is meaningful: the original SVAMP paper reports 43.8% for RoBERTa-based solvers trained on MAWPS+ASDiv-A, so 80% with RoBERTa-base would turn heads. RoBERTa predates both datasets, so contamination is low for that reader. Two weaknesses: training a PoT rival from answers alone is not well defined (it needs RL or self-training) and could end up a strawman, and the absolute GSM8K numbers will sit far below current LLMs.

### 7. Words as operators: a compositional reader in the same algebra; language as the glue between core systems (cognition-2, сумма 14, недель 3)

**Что.** For the register/cascade path, replace `text.reader` and the per-view rows of `text.Known` with a lexicon.
- Each word gets a soft choice over the generators of the chosen form, plus identity e (most words), flip (inverts the frame: 'has', 'of', 'from'), and an exponent marker (a numeral n repeats the next element n times).
- A clause's element is the ordered product of its words' elements. The frame is set by which name is said first, as `text.view` already marks it. A sentence is split into clauses at conjunctions and at 'meaning', each clause with its own name order.
- Learnt from answers through the exact machine, exactly as `Register` is: the gradient goes through the soft product, which is a convolution of distributions on the grid.
- `Register.close` is moved to the word level: its votes settle words rather than whole views. A word recurs in many views, so far fewer things are left unknown.
- Chunking: a view whose product keeps contradicting closure ('half past ten', 'nine o'clock') becomes one memorised unit.
- For kinship, kin words become elements of the idea-1 cascade (daughter = down + set female; '-in-law' = spouse bit). Names get a gender kept in exact memory across stories: a fact outside the weights.
- The lexicon is persisted per form (idea 4), so 'above/higher/older/before/north' arrive in a new family already read.

**Первый опыт.** CPU, 10-20 min:
- StepGame in other authors' words (anew.py, newtasks/phrasings: authors 1-8 teach, 9-12 test) and the original StepGame test with zero answers in its own words.
- Lexicon of about 2-3k words x about 8 elements.
- Report accuracy by words alone, before closure, at k=1 and k=10, against 77.4% and 59.4% for the current network reader in one.py. Then after word-level closure, against 100%.

Then a 'floors' family read with zero answers in its language, using the StepGame lexicon only.

Then CLUTRR text (gen23 and 1.3), with kin words as cascade elements: per-link reading against 80% for the small readers and 90% for RoBERTa-base. CPU 30-60 min.

**Что опровергает.** Kill if accuracy by words on unseen authors at k=1 is below 85%, or if per-link reading on CLUTRR 1.3 is below 83% (no better than the small readers).

**Судьи.** generality 4: Reading a new family with zero answers through a shared lexicon of word operators is 'the system produces the reading'. But that only works for words and algebras already learnt. Clause splitting, flip and exponent markers are hand-designed reading parts, and no machine is produced. / feasibility 6: It is all CPU, uses data already on disk (anew.py and the phrasings in newtasks), and existing closure does the reading without answers. Direction words compose, so beating 77.4% at k=1 by words looks reachable. The 85% bar on unseen authors (84.8% now with 20 authors) and 83% per link on CLUTRR (morphology, word order, distractor words) are real risks. StepGame can show a convincing result within 3 weeks; CLUTRR is less likely to. / evidence 4: The original StepGame test with zero answers is public, and the falsifiers are crisp. The lexical-semantics roots are real (Montague, Lambek, Baroni & Zamparelli 2010, Coecke et al. 2010, Boroditsky 2000, Hermer-Vazquez et al. 1999). But the 'other authors' phrasings come from one LLM generated inside the project, and StepGame and CLUTRR are near-solved by LLM+ASP pipelines. Compositional lexicons on templated text are old news, so a win reads as a good reader for crowdsourced templates.

### 8. Cumulative development: learning sets, overhypotheses, self-posed questions, and a sealed stream of families as the measure of leaving the circle (cognition-4, сумма 14, недель 4)

**Что.** (1) A persistent store (for example ground.json beside memory.json), updated after each family that is learnt. It holds:
- the idea-2 lexicon per form;
- Dirichlet counts over idea-1 presentations and law bits, together with the syntactic frames they came with;
- priors from frame to role ('AA is ___ of BB': ___ is a link element; a numeral before a noun is an exponent).
These become the prior and code lengths for the next family (hierarchical Bayes). A new family is first matched to the closest learnt family by relational signature (structure mapping: the same presentation under different words), and comparing two families that share a presentation sharpens that prior.

(2) Asking as experiment. Extend recall.py's asking (stage 13) to the cascade: among the top hypotheses (form x lexicon) that explain the answered stories equally well, request the answer of the unlabelled story, or a story composed from known sentence templates, on which they disagree most. Stop when one hypothesis is left and carry-over holds.

(3) The protocol.
- A stream of 12 or more families written by several outside authors (the owner, LLMs of different vendors, a few humans), with themes chosen by them, sealed.
- Code frozen at a hash.
- Each family offers 300 stories whose answers cost one 'ask' each.
- Record the asks needed to reach 95% on held-out long stories.
- Ablation: the same stream with the store wiped before every family.

**Первый опыт.** Retrospective first, 1 day on CPU, on families already on disk:
- StepGame; StepGame in foreign words; bAbI 17/18/19; the CLUTRR graph version; the idea-1 new families.
- Run them in a fixed order with the store kept and with it wiped, and count the asks needed to reach 95%.

Then the sealed stream: 12 families x 600 stories, about 2 weeks of other people's writing, about 30 min of CPU per family.

**Что опровергает.** Kill if any of these holds:
- the median asks-to-95% for families 9-12 is not at most half that for families 1-4;
- fewer than 8 of 12 families are learnt with the frozen code;
- with the store wiped, the system needs no more than 20% more asks than with it.

**Судьи.** generality 6: It gives the best operational definition of leaving the circle: a sealed stream of 12+ families by outside authors of several model families, code frozen at a hash, asks-to-95% falling over the stream, and an ablation that wipes the store. But the mechanism (priors over presentations, asking) only speeds up choice inside a space someone else built. It creates no machine, and it depends on ideas 1-2 existing. / feasibility 3: It depends on ideas 1 and 2 existing first. It needs 12 sealed families written by several people (about 2 weeks of others' writing), and its falsifiers ask for asks to halve by families 9-12. Only the retrospective day can land inside 4 weeks; the convincing part cannot. / evidence 5: The protocol is the cleanest operational test of 'out of the circle': a sealed stream by outside authors, frozen code, an asks-to-95% curve, and a wiped-store ablation as the control. The literature is real (Harlow 1949, Kemp, Perfors & Tenenbaum 2007, Angluin 1987). But the families are synthetic stories, not a public benchmark; the 12-family curve is noisy; LLM authors share a style (stage 32 already saw that); and everything rests on ideas 1 and 2.

### 9. Не выбирать алгебру, а выводить её из выученной таблицы: градуировки (null space), конгруэнции, каскад (unify-2, сумма 14, недель 2)

**Что.** Всегда начинать со свободной таблицы на коротких рассказах (FIN счётом, как `walk.Table`) — она не требует знать структуру. Потом из неё выводить алгебру:
(а) Градуировки = целочисленное ядро уравнений композиции g(ответ) = Σ знак·g(связь). Это линейная алгебра за секунду. Так счётчик (регистр) находится, а не выбирается. Для задач с ответом по знакам (StepGame) — малая ЦЛП по знаковым ограничениям или нынешний мягкий регистр.
(б) Конгруэнции = дробление разбиения по Муру / Майхиллу–Нероуду: символы, которые ведут себя одинаково с точностью до сдвига градуировки, сливаются в конечное управление ядра.
(в) Выход = SAY(clip z, q, r), где r — RESET-ячейка (например, пол последней связи), а q читает знак z (каскад).
Замерено здесь на CLUTRR по связям:
- Градуировка поколения найдена точно: ядро одномерное (отец, мать, дядя, тётя и родители супруга +1; деды ±2; сын, дочь, племянники, дети супруга −1; братья, сёстры, супруги 0). Выполняется на 9074 из 9074 рассказов, находится по 30.
- Таблица по N рассказам, тест k = 2…10 (1146 рассказов): 30 → 4,2 %; 60 → 27,4; 100 → 63,0; 200 → 89,4; 400 → 98,4.
- Найденный счётчик + управление из 3 состояний, читающее знак счётчика (форма Крона–Роудса), + ячейка пола; конечные части подгонялись градиентом и polish: 30 → 22,4; 60 → 63,4; 100 → 71,4 (при k = 10: 64,7 против 47,1); 200 (K = 4) → 79,1.
- Прямое произведение без каскада: 64,5 при N = 100.
Итог: на малых данных выигрыш, на 200 проигрыш, потому что градиент находит неверное конечное управление. Значит, ядро надо искать минимизацией или перебором с MDL, а не градиентом.

**Первый опыт.** Скрипты: brainstorm/after_stage_45/scripts/unify/grading.py, grading_few.py, holes.py, cascade_wreath.py (папка временная, её может стереть).
CLUTRR gen23 по связям, N ∈ {30, 60, 100, 200, 400} рассказов из 2–3 звеньев, тест k = 2…10. Сравнить:
(i) таблицу;
(ii) найденный счётчик + ядро, найденное дроблением разбиения, с выбором K по MDL;
(iii) регистр one.py.
CPU, минуты. Потом та же процедура на StepGame (таблица на 1–2 звеньях → знаковая ЦЛП → должен выйти Z²) и на bAbI 19.

**Что опровергает.** Опровергает любое из:
- при N = 100 «счётчик + ядро» ниже 90 % по всем k (у таблицы 63 %);
- при N = 400 ниже 98 %;
- на StepGame процедура не находит двумерную градуировку из рассказов в 1–2 звена — тогда регистр так и остаётся машиной, выбранной руками.

**Судьи.** generality 5: Here the system infers the algebra rather than choosing it. It starts from a free table, finds gradings from the null space, then congruences by Moore/Myhill-Nerode, so the counter is found, not hand-picked. That is genuinely 'the system produces the machine'. But it is measured negative at N=200 (the gradient finds the wrong control), it only helps where an abelian part exists, and every test is on sets already won. / feasibility 6: It is cheap (CPU minutes) and the scripts exist. The generation grading is found exactly from 30 stories. The fix it names, enumeration or minimization by MDL instead of the gradient, is small for CLUTRR (gender plus about 3 control states). Risks: the 98% bar at N=400 may sit above the structured model's ceiling (bicyclic x flags measured 95-98%), and StepGame needs a sign-constraint ILP. It is narrow but could be convincing within 2 weeks. / evidence 3: A neat, honest result (an exact generation grading on 9074 of 9074 CLUTRR graph stories), with real references (Myhill-Nerode, Moore, Smith normal form, HodgeRank 2011). But it is limited to CLUTRR's graph version and StepGame, and the measured result already loses to the plain table at N=200. A positive result would be a curiosity about kinship composition, not general evidence.

### 10. Measure the reader surface: weights x words of plain text, with public checkpoints in the existing Encoder slot (reader-2, сумма 14, недель 1)

**Что.** text.Encoder already loads any Hugging Face AutoModel, and one.py passes --encoder. Run the same protocol (by links 6000x32 steps; by answers 2500 steps inside one.py) with checkpoints that vary size and pretraining words separately. All exist on Hugging Face, checked today.
- **Size axis:** google/bert_uncased_L-2_H-128_A-2 (4.4M), L-4_H-256_A-4 (11M; done today: links 84.4/85.9%, answers 73.9/77.8%, 37 min CPU), L-4_H-512_A-8 (29M), L-8_H-512_A-8 (41M). All on about 3.3B words.
- **Words axis at fixed architecture:** nyu-mll/roberta-med-small-1M-1, roberta-base-10M-1, -100M-1, -1B-1.
- **Child-scale corpora:** babylm/ltgbert-10m-2024 and -100m-2024.
- **Anchor:** roberta-base (about 30B words; 90.1% known).

Readouts: link accuracy and answers on gen23, gen234 and CLUTRR 1.3, plus the stage-41 cross-author StepGame curve per encoder. Each model is also run on the path view from idea 1. The result says where the 0.2B goes: into the grammar module, or nowhere.

**Первый опыт.** - **Colab A100:** 10 checkpoints, by links, on gen23 + gen234 + CLUTRR 1.3, about 10-30 min each, about 4 h in all.
- **Then:** the three most informative by answers inside one.py, 3 seeds each, about 3 h.
- **CPU fallback:** the 4-11M models run on CPU in about 40 min each, as BERT-mini did today.

**Что опровергает.** - **Refuted:** if every model with 100M words or less, at 45M weights or less, stays at 82% links or below on gen23 (BERT-mini on 3.3B words: 84.4%; scratch: 80.5%). Then the reader's gain comes from billions of words, and the "little data" half of the bet fails for reading.
- **Holds:** if 100M words gives 84% or more at 30M weights or less, with about 7x margin, and 0.2B is too much for reading in these worlds.

**Судьи.** generality 1: A measurement of reader size against pretraining words, which explicitly adds no machine. It informs the 0.2B claim, not the circle. / feasibility 8: It is the most certain to deliver. The Encoder slot already loads any Hugging Face model, the checkpoints exist, and BERT-mini is already measured (84.4%). It takes about 1 week on Colab and a CPU. Either outcome is a convincing, publishable measurement of the README's own missing curve, and it settles where the 0.2B goes. It is a measurement, not a new mechanism. / evidence 5: A clean, cheap measurement with public checkpoints that are real: Turc et al.'s compact BERTs, the Zhang et al. 2021 MiniBERTas, and the BabyLM ltgbert-10m-2024 model, which exists on Hugging Face. It answers the README's own open question. But the deciding thresholds (82 vs 84% link accuracy) are 2 points on noisy CLUTRR 1.0 labels, with single seeds and differing tokenizers, so experts will want confidence intervals. A positive result settles the narrow pretraining-words question, not the architecture.

### 11. Одно исчисление состояния вместо трёх машин: регистр как выученная алгебра, форма выбирается по грамматике (red-team-2, сумма 14, недель 4)

**Что.** В one.py классы Register и Tabled (и счёт/«у кого» из Recalled) заменяются одним классом. Тип регистра берётся из маленькой грамматики: Z^n (n=1..3; набор шагов выучивается из [-3..3], а не задан как {-1,0,1}); циклическая Z_m (m до 24); конечный моноид с выученной таблицей Кэли (|M| до 32: мягкий тензор T[a,b,c] — вероятность того, что a·b=c, затем затвердение и polish, как в walk.Walk.polish); полный порядок; последовательность (свободный моноид — для ответов вида «s,e» в bAbI-19); прямые и каскадные произведения этих типов. Ответ — выученное факторотображение регистра; это обобщение нынешнего «по знакам чисел». Register.close переписывается для любого типа: два пути между одними вещами обязаны дать один и тот же элемент, то есть получается функтор из графа рассказа в алгебру, или согласованность путей. Тогда чтение без ответов работает и для таблицы (родство), где сейчас его нет (оговорка 2 этапа 44). Тип выбирается переносом с коротких путей на длинные плюс биты типа и таблиц. Оценка — «алгебры, скрытые от проектировщика». Третья сторона (владелец или сессия без доступа к репозиторию) пишет по запечатанной одностраничной спецификации около 6 генераторов рассказов о двух вещах со связями, которые складываются вдоль пути. Проектировщик видит только формат. Примеры того, что может выбрать автор: дни недели, 3D, расстояния в шагах, отношения интервалов, другая система родства, курс и ходы робота.

**Первый опыт.** Процессор. Шаг A (1–2 дня, только свои данные): грамматика обязана сама повторить выборы one.py — StepGame -> Z^2 (не ниже 95 % при k=10), граф CLUTRR -> моноид (не ниже 99 % на 10 звеньях), bAbI 17/18/19 — с точностью до пункта. Шаг B: 6 запечатанных генераторов по 10 тыс. рассказов, учение на k не больше 3, проверка на k=6..10; каждый прогон занимает минуты (моноид |M|=32 — это 32 тыс. чисел).

**Что опровергает.** На запечатанных семействах меньше 4 из 6 дают не ниже 95 % при k=10, или тип выбран неверно хотя бы на двух. На шаге A: StepGame при k=10 ниже 93 % или граф CLUTRR ниже 98 % — тогда грамматика стоит точности там, где отдельные машины работали.

**Судьи.** generality 5: The families are hidden properly: a third party writes generators from a sealed one-page spec without seeing the grammar, and the type is chosen by the system. The idea admits it is generalisation inside 'relations of two things composed along a path'. Allen/RCC families break the monoid, and Cayley tables learnt from answers are underdetermined. / feasibility 5: It overlaps with cognition-1, plus soft Cayley tables (|M| up to 32) that may not be identifiable from answers alone. If a sealed author picks intervals or RCC, it fails by design (composition there is a relation, not a function). Step A is feasible, and 4 of 6 sealed families is plausible but not likely, given the 4-week budget and outside generators. / evidence 4: Leakage is handled better than in cognition-1: a third party writes 6 generators from a sealed spec without knowing the grammar. The literature is real (Kemp & Tenenbaum 2008, Allen 1983, RCC-8, TransE and RotatE). But all of it is synthetic 'two things linked along a path' composition, not a public benchmark, and the idea admits that Allen/RCC-style families need relation algebras, not monoids. A positive result shows generality only within one kind of task.

### 12. Указательная машина с выученным управлением: CLRS вне обученного размера, семейства алгоритмов скрыты (red-team-3, сумма 14, недель 5)

**Что.** Ядро loop.py переходит от «одна команда — одна выученная маршрутизация» к машине модификации памяти с маленьким конечным управлением. Строка таблицы: состояние q (до 8) x вид узла под указателем x знак регистра -> действие (идти по полю, поставить метку, положить в очередь или стек или взять оттуда), операция регистра, следующее q, останов. Таблицы мягкие во время учения; градиент идёт сквозь точное исполнение с сохранением массы (как в loop.py); затем затвердение, дискретный polish и перезапуски. Примитивы памяти фиксируются и замораживаются до того, как открыт хоть один алгоритм CLRS: граф с полями, метки на узлах, одна очередь, один стек, целый регистр. Учение двумя способами: по одним выходам и, отдельно, с подсказками CLRS (промежуточные состояния) как более сильным надзором. Ответ — неподвижная точка, поэтому размер любой, а дрейфа нет; это прямой ответ на вопрос владельца о многих оборотах петли.

**Первый опыт.** Процессор, неделя: пять процедур на случайных списках и графах из генератора loop.py — длина списка, максимум, обратный обход по обратным ссылкам, расстояние BFS от источника (очередь), порядок открытия DFS (стек). Учение на графах до 8 узлов, по одним выходам, по 2 тыс. примеров; проверка на 32 и 128 узлах. Затем CLRS-30 (данные готовит A100 в Colab, учение ядра — процессор): до открытия набора правилом выбрать 4 алгоритма для разработки и 8 скрытых (например, все графовые и поиск), проверка на n=64.

**Что опровергает.** На пяти процедурах: для двух и больше управление не найдено (не выше 90 % на 128 узлах) за 1 CPU-час на каждую. На скрытых алгоритмах CLRS: среднее вне обученного размера ниже опубликованного у Triplet-GMPNN на тех же алгоритмах, или ниже 90 % на половине и больше.

**Судьи.** generality 5: The control table is learnt over frozen pointer memory, the memory primitives are frozen before CLRS is opened, and the hidden algorithm families are split by a rule. So the system would write the algorithm itself. The risks are the TerpreT outcome (soft control tables get stuck) and dynamic programming needing memory the frozen set lacks. It also has nothing to do with reading. / feasibility 3: TerpreT-style soft control tables tend to stick. Beating Triplet-GMPNN on hidden CLRS algorithms needs decoders and data pipelines. The five toy procedures might work in a week, but the convincing CLRS part probably will not fit 5 weeks, let alone 4. / evidence 6: CLRS-30 is a respected public benchmark, the hidden algorithm families are chosen by a rule before the set is opened, and the out-of-distribution test is the standard n=64. A learnt controller that is exact out of distribution from outputs alone would impress the neural algorithmic reasoning community. Against it: the baseline is Triplet-GMPNN from 2022 (I could not find the 74.14% average in search snippets; it is plausible but should be checked in the paper's table), and later methods are stronger. The queue and stack primitives are picked with BFS and DFS in mind, which leaks the designer's knowledge of CLRS. The hint-supervised variant proves less. Feasibility is the main risk (the TerpreT outcome), not convincingness.

### 13. Monotone recursion: a TRM whose only carried state is an option set that can only shrink (recurrent-2, сумма 13.5, недель 4)

**Что.** A TRM-sized core: 2 layers, 1–5M weights, the input given anew every turn (TRM/Huginn recipe). The state carried between turns is not a free vector z. It is, per cell, a vector m in [0,1]^V of values still possible.

Each turn the network reads x, its cell's m and its neighbours' m (attention restricted to a few learnt relations: hard addressing) and outputs a removal r = sigmoid(f). The update is m <- m * (1 - r).
- m is monotone non-increasing and bounded, so even the soft dynamics must converge.
- Hardened, it reaches a fixed point in at most cells×values turns; further turns cannot change it, so overthinking is impossible by construction.
- A fixed point with more than one option left: branch on the cell with fewest options (DPLL). The network only ranks branches.
- An empty option set means contradiction: backtrack.
- For distance-like tasks the lattice is upper bounds and the update is min-plus, also monotone.

After training, the learnt propagator is hardened into a rule table (local pattern of option sets -> removals) and checked exactly on all training solutions. The network then has written a machine that can be verified.

In code: a new lattice.py, the monotone sibling of loop.py's Machine (which carries mass over (node, number)). kr/mine.py and kr/maze.py are the exact ceiling and controls.

**Первый опыт.** The controls are done: kr/mine.py and kr/maze.py, with the data in sudoku_data/ and maze_data/ under brainstorm/after_stage_45/scripts/. About 15 CPU-minutes.

Next, Colab A100, about 1 day:
1. Train TRM from its official code, and monotone-TRM at the same weight count, on Sudoku-Extreme-1k and Maze-Hard-1k with the same augmentation. Give no mined invariants: the network must learn the propagator.
2. Report test accuracy at 16, 64, 256, 1024 and 4096 turns.
3. For mazes, also train on 15×15 and test on 30×30 and 60×60, generated with the convention just recovered (the Bansal easy-to-hard protocol).
4. Extract the hardened rule table and count the training solutions it explains.

**Что опровергает.** Any of these kills the idea:
- Monotone-TRM is below our TRM reproduction at equal weights and data.
- Its accuracy drops by more than 1 point from 64 to 4096 turns.
- The extracted rule table explains fewer than 99% of training solutions.
- On 60×60 mazes after 15×15 training it is below 90%.

**Судьи.** generality 3.5: A network-extracted rule table is a machine written by the system, and monotonicity answers the drift question. But both benchmarks are already solved by hand-written mining and tie-break search. The held-out family (jigsaw Sudoku, Futoshiki, KenKen) appears only under risks, not in the falsifier. / feasibility 4: The controls are done, but they show both benchmarks are solved with no network, which undercuts the claim. Reproducing TRM can itself take days of GPU, monotone training with branching gives a sparse signal, and it must also beat TRM at equal weights. A convincing result within 4 weeks of Colab is unlikely. / evidence 6: A head-to-head with TRM from its official code at equal weights and data, on public Sudoku-Extreme and Maze-Hard. The falsifiers are clean: stability from 64 to 4096 turns, 15x15 to 60x60 extrapolation, and rule extraction that must explain at least 99% of training solutions. The TRM and HRM numbers and Bansal 2022 are real. Experts will discount it because the monotone option-set bias is constraint propagation built in, and the idea's own controls show both benchmarks fall to search with no network, so a win shows stability, not generality. The maze tie-break observation is a good, checkable side result.

### 14. Impasse, subgoal, chunk: the system takes over the step I have been doing (Soar/GPS outer loop) (forgotten-old-4, сумма 13, недель 4)

**Что.** A driver (self.py) around one.py, in six steps. (1) Run the library: the three machines, the cells of idea 1, the algebras of idea 2, and algebra.py's composed questions. (2) An impasse is a kind of question that is not explained or not 'sure'; the learner already knows this, being 99.5% right where it says it understood (stage 22). (3) Means–ends diagnosis by algebraic property tests on the teaching stories, a few lines each. Swapping two sentences changes the answer: non-abelian, so a permutation cell. A sentence twice equals nothing: an involution, Z2. A period m: Z_m. Only the last sentence of a class matters: reset. The answer is a set or 'maybe': a disjunctive algebra. The question names two things: algebra; one thing over time: cascade. The answer word occurs only in other things' sentences: in-story induction (bAbI 16, 38%). Nothing fits: 'not mine'. (4) Subgoal: search in the grammar picked, cheapest first, asking for minimal pairs where the top two candidates disagree. (5) Chunk: store the machine found with its price in bits. memory.py already keeps readings; extend it to machines, and feed them to guess.py's dreams as first proposals for the next family. (6) A blackboard: all machines read and write one shared memory, so a compound question is a chain of machines (Hearsay-II).

**Первый опыт.** 1 week on CPU, after idea 1's cascades work. Write the six property tests and the dispatcher, run them on the six single-cell families, the turtle and two BBH tasks, and report the confusion between diagnosis and true structure. Then the sealed sequence, frozen code, one run each, about 40 teaching examples per task: BBH web_of_lies, navigate, tracking_shuffled_objects_3/5/7, date_understanding, logical_deduction_3/5/7, temporal_sequences, then two fresh Haiku-written families. Report coverage at 99% accuracy, silent errors, and machines made by the system.

**Что опровергает.** Kill it if the diagnosis picks the wrong grammar on more than 1 impasse in 3. Kill it if any family in the sealed sequence needs a commit to reach its number. Kill it if coverage at 99% accuracy on the second half of the sequence is not higher than on the first half, since chunks must pay off.

**Судьи.** generality 5: It names the outer loop (impasse, subgoal, chunk) and gives the right metric: machines made by the system against machines made by me, with a kill on any commit. But the diagnosis is a hand-written list of property tests mapped to grammars, including task-shaped ones such as 'in-story induction', which is a list of parts one level up. It depends on ideas 1-2 covering the families. / feasibility 3: It assumes the cascades of idea 1 already cover the families. The sealed BBH sequence with zero commits and wrong diagnosis on 1 impasse in 3 at most is a strong claim, and BBH tasks need new readers. The property tests are cheap, but a convincing result needs the whole stack within 4 weeks, which is unlikely. / evidence 5: A sealed sequence of public BBH tasks with frozen code and the 'zero commits' falsifier is a clean, outside-legible measure of leaving the circle, and the Soar, GPS and Hearsay-II roots are real. But the property tests and the diagnosis order are written by someone who has already mapped each BBH task to its structure, so the 'system-made machines' are partly pre-chosen. BBH tasks are also routinely solved by LLM+program methods, so the external bar is low unless the system is small and has no pretraining.

### 15. fold.py: одна машина = ВЫБОРКА × СВЁРТКА(алгебра) × SAY; минимальный набор команд вместо трёх классов one.py (unify-1, сумма 13, недель 3)

**Что.** Один интерпретатор, в который превращаются Register, Tabled, Recalled и loop.Machine.
Память (0 весов): журнал событий (t, символ, слоты-вещи и числа) — его уже дают `one.story()` и `rows.Reader.rows()`; граф пар — `text.together` и `text.near`.
Выборки (3 + 1):
- TIME[g, ключ] — события группы символов в порядке рассказа, с хранилищем на каждую вещь-ключ (= `recall` stores / 'at');
- PATH(x, y) — кратчайший путь (`walk.way`, `text.surest`);
- ORBIT(поле, x, до вида) — `loop.Machine.hard`, `recall.chain` и `reach`;
- SORT(ключ) для времени (`recall` 'when').
Алгебры свёртки (5):
- RESET — последнее значение, значение = вещь или множество; покрывает set, clear, to, lift, swap;
- Zⁿ с φ: символ → {−2..2}ⁿ (`Walk`);
- NAT{+1, +значение, max} (`loop`);
- FIN(k) — выученная таблица k × символ → k (`Table`, но как автомат);
- SET с add/remove (bAbI 7, 8).
Комбинаторы (2): произведение × и каскад ⋉ (вторая часть читает состояние первой; ключ по вещи — частный случай).
Запросы к свёрнутому состоянию (≈5): GET, INV, OTHER/последний, COUNT, EXTREME/MEMBER, плюс SCAN (история состояний) для asof/was/prior.
Выход (1): SAY — таблица «проекция состояния (знаки, clip(z), q, вещь, статус) → слово», как `Walk.says` и `recall.grade`.
У каждой алгебры есть мягкая форма, сохраняющая массу: для Zⁿ свёртка torch.roll (уже есть в `Walk.forward`), для FIN стохастические матрицы, для RESET смесь «оставить / записать» (`soft.py`), для указателя scatter_add (`loop.Machine.forward`). Произведения и каскады таких отображений тоже сохраняют массу: дрейфа нет, ответ — неподвижная точка.
Что учится на каркасе: выборы на символ (градиент, затвердение, общий `Walk.polish`). Каркас выбирается типизированным перебором по размеру с оценкой в битах (MDL) и переносом «короткие → длинные» — это критерий one.py.
Развёртка по наборам:
- bAbI 1, 2, 3, 6, 9, 11–13 = TIME + RESET по ключу + GET / GET∘GET / SCAN-prior;
- 4, 5 = хранилище связей + GET/INV;
- 7, 8 = SET + COUNT;
- 10 = RESET(множество) + MEMBER-статус + SAY;
- 14 = SORT;
- 15 = GET∘GET;
- 16 = GET∘OTHER∘INV∘GET;
- 17, 18 = PATH + Zⁿ;
- 19 = PATH + SAY(последовательность);
- 20 = GET + SAY-таблица;
- StepGame = PATH + Z²;
- CLUTRR = PATH + FIN или Z⋉FIN;
- петля = ORBIT + NAT.

**Первый опыт.** Выразимость уже замерена: brainstorm/after_stage_45/scripts/unify/tiny_isa.py — хранилища «последнее значение», группы видов по словам слотов, программы GET / GET2 / SAME / STATUS + SAY по счёту. bAbI 16: 100 % (10 программ), bAbI 10: 98,2 % (3), bAbI 20: 100 % (63), bAbI 1: 100 %; всё со 150 вопросов (~30 рассказов), меньше секунды. Это не вслепую: команды выбраны после того, как я посмотрел на задачи.
Следующий шаг — fold.py (~700 строк), протокол one.py одним поиском на 24 наборах: bAbI 1–20 (1k), StepGame, CLUTRR gen23 по связям и по тексту, задача `loop.py`. CPU, 4 ядра, 3–4 часа.
Затем замороженный код:
- на bAbI 'shuffled' (буквы перемешаны — те же задачи без английского);
- на старых пулах Haiku I–Q, базой служат числа `recall.py` и `records.py`.

**Что опровергает.** Опровергает любое из:
- на любом наборе, где one.py ≥ 95 %, падение больше чем на 2 пункта;
- bAbI 4/10/16/20 замороженным языком ниже 95 %;
- верного каркаса нет среди первых 200 перебранных хотя бы на 3 наборах из 24;
- на пулах I–Q взято (≥ 95 %) меньше задач, чем у `recall.py` и `records.py` вместе.

**Судьи.** generality 4: It consolidates three machines into about 16 commands, which reduces the list but does not remove the designer. The commands were picked after looking at the tasks, as the idea admits. Its blind tests are shuffled bAbI (the same tasks) and the I-Q pools the designer has already seen. The risk of 'the same list in another notation' is real. / feasibility 6: The expressiveness evidence is real but not blind (tiny_isa: bAbI 16/10/20/1 at 100/98/100/100 in under a second). About 700 lines and one CPU run over 24 sets is doable in 3 weeks. The risks: a soft cascade sticking in mixtures (measured 67.5% on CLUTRR), and the falsifier on pools I-Q, which asks it to match the coverage of records.py's many stages. Consolidation plus the shuffled bAbI run is probably achievable; beating the old learners on I-Q is not certain. / evidence 3: The instruction set was designed after looking at the tasks (the idea says so), and the results are on bAbI, StepGame and CLUTRR, which the designer has seen and which are already solved, plus in-house Haiku pools. bAbI 'shuffled' is a real public variant but only renames tokens. The test is about expressiveness, not about leaving the circle.

### 16. Grammar x lexicon: one shared reader of structure, meanings in an exact lexicon (reader-1, сумма 13, недель 2)

**Что.** 1. **New view.** Replace text.view(told,x,y,True), which today is every word of every sentence naming both people, with a path view. The path view is the shortest dependency path between the two mentions: the lemmas and arc labels on it, plus poss/case words hanging off it (her, 's). Names are written <0>/<1>/<o> as now. Each story is parsed once with spaCy en_core_web_sm (12 MB); the 9k+1k CLUTRR stories take 40 s on CPU, measured. The 5% of pairs that cross sentences take the pronoun's antecedent by the subject rule, or fall back to the old view.
2. **Reader output independent of the machine.** It says which relation word or phrase links the pair, which side (as said, or inverse), or that nothing is said. The sex of each person comes from an exact per-story register (pronouns, kin appositions, what the read links imply), not from the reader. The 20-way kin-with-sex label becomes (kind) x (side) x (sex from the register).
3. **Meaning in an exact lexicon.** It is keyed by the path signature, which generalizes text.Known rows: today they are keyed by whole views and almost never fire on the test. Back-off goes to (relation lemma, first/last arc), then to the small GRU on the path words. Each row maps a phrase id to a machine choice: an offset for the Register, a symbol for Tabled, a slot for recall. Rows are learned by counting through the hardened machine, exactly as walk.Table is now. The inverse of each relation word is learned by counting pairs read both ways.
4. **One reader for all three machines in one.py.** Recall's templates are replaced by the same path view.

**Первый опыт.** Add `text.py --paths` and the same flag in one.py. Data: CLUTRR 1.0 gen23 (working set), gen234 (second test), and the four CLUTRR 1.3 files.
- **Three readers:** (a) the counting lexicon alone (0 weights); (b) the small GRU (text.Reader, about 50k weights) on the path view; (c) BERT-mini on the path view. All three use the exact sex register.
- **Teaching:** by links and by answers.
- **Cost:** CPU, about 2 h for (a) and (b), 40 min for (c).
- **Then:** StepGame with the 24 Haiku authors' phrasings through readers.py (stage 41 setting: 20 authors learnt, 4 unseen), path view against word view.
- **Starting point:** brainstorm/after_stage_45/scripts/reader_lens/paths.py and sex.py.

**Что опровергает.** Named in advance. The idea is refuted if:
- the small GRU on path views, taught by links, reads 82% or less of links on gen23 (bag of words 80.5%), AND gives under 86% of answers on CLUTRR 1.3 (now 83.2%); or
- inside one.py by answers it stays under 60% on gen23 (now 56.1%); or
- on unseen StepGame authors it reads under 87% (now 84.8% with 20 authors).

**Судьи.** generality 3: A shared reader, with dependency paths plus a lexicon learnt by counting, makes reading machine-independent. That is useful, but it is a reader upgrade tested on known sets, with a borrowed parser, and it says nothing about who writes the machine. / feasibility 6: spaCy parsing is measured at 40 s for all CLUTRR, path signatures recur 58% against 4.7%, and paths.py and sex.py exist. The falsifier bars are modest (82% links, 86% on 1.3, 60% in one.py, 87% on unseen StepGame authors). It fits in 2 weeks on a CPU. The gain may be small and the borrowed parser invites objection, but it is a solid shared-reader result. / evidence 4: A credible, well-instrumented engineering step (shortest dependency paths, DIRT, Bunescu & Mooney 2005, all real), with numbers fixed in advance on public CLUTRR 1.0 and 1.3. But the gains at stake are a few points over the project's own readers. It borrows spaCy's trained parser, and the StepGame test uses the in-house Haiku authors. Experts would see a relation-extraction improvement, not a breakthrough.

### 17. One machine for every 'what is X to Y': a learnt relation algebra with path consistency, so closure works for any machine (forgotten-old-2, сумма 12, недель 3)

**Что.** Tabled (walk.Table plus text.py) and, as a special case, Register are replaced by a learnt qualitative calculus. Atoms are the answer words plus spares, as now. The composition table is over sets of atoms (disjunctions), counted from answers the way walk.Table counts. The converse comes from the sign of the link, plus an identity. The answer comes from path consistency over the whole story graph instead of a chart on one shortest way: a single atom gives the word, several atoms give 'maybe' or abstain. Reading without answers generalises: an unknown sentence kind's relation is the intersection of compositions along every other path. Register.close in one.py is the group special case, and one.py caveat 2 says closure exists today only for the register. For order in time, events become nodes, with a point algebra {<,=,>} plus metric constraints ('a day after' = simple temporal problems). Propagation then replaces rows.py's enumeration of all event orders, which took 11–150 minutes per task, and J7 never finished. Cyclic days use a Z7 cell from idea 1 or a cyclic order algebra; E12 failed for lack of exactly that.

**Первый опыт.** 2–3 days on CPU. (1) StepGame via walk.py: replace Table.hard (chart on the shortest way) with PC-2 over the whole story graph using the learnt disjunctive table. Compare with Table (67.2% in one.py's selection run) and Register (95% at k=10). (2) walk.py --hold 0.5: unknown kinds read by PC-closure, against the register's closure (101–102 of 102). (3) rows.py on the order-in-time tasks J1–J12 and I9–I12: point-algebra propagation in place of the all-orders worlds; record accuracy and minutes. (4) bAbI 10 inside one.py.

**Что опровергает.** Kill it if, on StepGame k=10, the learnt algebra with PC scores below 90% (register: 95%). Kill it if PC-closure reads fewer than 90% of the held-out kinds the register reads. Kill it if on the J tasks accuracy falls more than 2 points below the worlds enumeration, or time is not at least 10 times lower. Kill it if bAbI 10 stays below 90%.

**Судьи.** generality 4: A generic path-consistency propagator over a learnt disjunctive algebra closes real holes: order in time, honest 'maybe', closure for the table. But it is one more designer-written machine for the binary-relation genre, and all its tests are existing tasks. / feasibility 5: Point-algebra propagation replacing the all-orders enumeration in rows.py is a feasible, measurable win in about 10x time. bAbI 10 is already at 98% via tiny_isa. But the headline falsifier, a qualitative calculus with PC at 90% or more on StepGame k=10, will very likely fail: disjunctive cardinal composition cannot count, which is why the table gets 67% and the register 95%. The win is mostly internal, not convincing from outside. / evidence 3: The classic literature is real (Montanari 1974, Mackworth 1977, Allen 1983, Vilain & Kautz 1986). But the tests are StepGame, where the register already gets 95% and LLM+ASP pipelines are near 100%, plus the project's own J/I tasks and bAbI 10. A positive result is consolidation and a speed-up on in-house sets, not evidence an outsider would cite.

### 18. Машину пишет система: поиск Левина/OOPS по языку fold.py с растущей библиотекой и подсказчиком (unify-3, сумма 12, недель 4)

**Что.** Для нового набора:
1. Типы берутся из данных: вопрос о двух названных вещах? ответ — вещь, слово, да/нет или число? (это `two` в one.py и `shape` в `recall.learn`).
2. Каркасы перебираются в порядке «априорная вероятность × время»; каждый подгоняется: мягкая свёртка, затвердение, polish.
3. Каркас принимается по MDL в битах и по переносу на длинные рассказы, с ответом «понял / не понял» при 99 % — это уже есть.
4. Сжатие: частые подпрограммы (GET∘OTHER∘INV∘GET, Z⋉FIN, PATH + SAY-последовательность) становятся словами библиотеки с повышенным приором. Их мягкие формы получаются автоматически: композиция отображений, сохраняющих массу, сохраняет массу.
5. Подсказчик (наследник `guess.py` и `route.py`) учится на снах, порождённых библиотекой, и по сырому тексту 20 рассказов называет каркас.
Протокол последовательный: наборы идут по одному (bAbI → StepGame → CLUTRR → петля → пулы Haiku → внешние), код заморожен, меняются только библиотека и подсказчик.

**Первый опыт.** Последовательный прогон по 24 наборам идеи 1: язык заморожен после первого, библиотека растёт. На каждом наборе мерить: сколько каркасов перебрано до успеха, время, точность. Потом тот же прогон в обратном порядке. CPU, сутки.
Colab A100: подсказчик (1,5–12 млн весов) на миллионе снов из библиотеки. Метрика: верный каркас в top-5 на реальных наборах, которые подсказчик не видел.

**Что опровергает.** Опровергает любое из:
- число перебранных каркасов или время не падает хотя бы втрое от первой трети последовательности к последней;
- хотя бы одному набору нужна новая команда;
- у подсказчика top-5 ниже 50 % на невиданных реальных наборах.

**Судьи.** generality 6: Levin/OOPS search with a growing library and a learnt proposer replaces the designer's step. The kill rule is 'any set needs a new command'. But the sequence is mostly sets the fold.py language was designed on. Stage 32 found dreams cover few real tasks (19/87). The base language is hand-picked after looking at the tasks, so the circle can come back as 'add a command'. / feasibility 3: It needs fold.py working first. A proposer trained on dreams has failed before (stage 32: 19 of 87 tasks in the dreams' support). The falsifiers need a 3x drop in search and zero new commands over 24 sets. Four weeks is not enough to reach a convincing result. / evidence 3: Real roots (Levin, OOPS, DreamCoder, Stitch). But the sequential run is over 24 datasets the designer has seen, and the order and the dreams are under the designer's control. The 'unseen real datasets' for the proposer are not named, and stage 32 found that dreams covered only 19 of 87 tasks.

### 19. Coupled reading over unlabeled text: exact invariants as the teacher (reader-3, сумма 12, недель 2)

**Что.** Add a consistent() pass in one.py that generalizes Register.close. Over many stories without answers, a reading of a link becomes a lesson only if it passes exact checks that do not depend on the reader:
- (a) one reading per phrase kind across the whole corpus (stage 43's +6, now as a lesson rather than only at inference);
- (b) reading(x,y) and reading(y,x) are inverses under the learned table;
- (c) one sex per person across all mentions, from the register in idea 1;
- (d) relations learned to be functional stay functional: one mother, one father; in bAbI one place per thing at a time. The flag is counted from teaching (how many fathers does a person ever have), not written by hand;
- (e) every cycle closes (Register.close), and table triangles hold: if (x,y), (y,z) and (x,z) are all read, then table(x,y) composed with table(y,z) equals reading(x,z);
- (f) two independent views agree: the bag-of-words reader and the path-lexicon reader from idea 1 (co-training).

Links that pass and are confident become lessons for both readers and rows of the lexicon. Run 2-3 rounds, and keep adding unlabeled text: CLUTRR test halves, the rob_* files, CLUTRR 1.3 stories, StepGame halves.

**Первый опыт.** - **Teaching:** readers by links on gen23 teaching, as in stage 42.
- **Unlabeled text:** the first half of gen234 plus the rob_* train files, answers never opened.
- **Checks:** (a), (b), (c), (e), and (f) once idea 1 exists; one round.
- **Test:** answers on the second half of gen234 and on CLUTRR 1.3.
- **Audit only:** the precision of the accepted lessons against the links given beside the stories.
- **Cost:** CPU, about 3 h.

**Что опровергает.** - Under +3 answer points on the untouched second half of gen234, over "the same sentence read the same" recomputed on that half (72.1% on the whole of gen234); or
- audited lesson precision under 90% (drift).

Either means chain-shaped text carries too few invariants, and weights (idea 2) are the only lever left for free text.

**Судьи.** generality 3: Coupled self-training of readers from invariants improves reading of known families. The functional flag (d) may be hand-coded domain knowledge. No machine is produced. / feasibility 5: Invariants have worked three times before (stages 39-40, 43), and the checks are cheap to code. But CLUTRR chains carry few invariants, triangles are rare, and +3 on an untouched half with lessons 90% precise is uncertain. It is a 2-week CPU job with odds of a clear result near a coin flip. / evidence 4: Constraint-driven self-training (Yarowsky, Blum & Mitchell, Chang et al. 2007, NELL, all real) with a held-out half and a precision audit is a sound protocol. But the target is +3 points on noisy CLUTRR 1.0 chains that carry few invariants, and the baseline is the project's own earlier variant. Only a modest semi-supervised result is possible.

### 20. TRM's outer loop over programs, not answers: proposer, exact executor and a library, on ARC-AGI-1 (recurrent-3, сумма 12, недель 10)

**Что.** The carried state of the recursion is a discrete program in a frozen language, built from:
- the parts of ideas 1 and 2;
- grid things: connected components by colour, found the way one.py finds things;
- fields: 8 directions, 'same colour', 'inside';
- loop.py's instruction 'follow a field until a node of a kind and say where, how many, the sum or the greatest';
- a cascade fold along a path, a monotone fill, a colour-map table, and copy/translate/reflect.

Each outer step:
1. The program is executed exactly on the demo pairs.
2. The network (TRM-sized, 7M) sees the demos, the current program and the cell-by-cell difference grid.
3. It proposes an edit or a whole new program.
4. The loop stops at the fixed point 'all demos match exactly' (a verified program), which is then run on the test input.

Teaching:
- dreams: random programs on random grids give input/output pairs, as recall.dreams, guess.py and route.py did for stories;
- hindsight relabelling: whatever a wrong program outputs is the correct answer to some other task (CodeIt);
- expert iteration on solved training tasks, as AlphaProof does.

Between rounds, frequent subprograms of verified solutions become new primitives (DreamCoder/Stitch). They are kept only if they shorten the total description in bits, the 'one score in bits' the README already asks for.

A new machine is then a verified, stored program, not my code.

**Первый опыт.** 1. CPU, 3–4 days, no network: an audit of how much the frozen language covers. Enumerate programs up to depth 3 with an exact check on the demo pairs for all 400 ARC-AGI-1 training and 400 public eval tasks. Count tasks solved, and tasks where a program fits the demos but fails the test (spurious).
2. Colab A100, one session of about 6 hours: a 7M proposer on 1–3M dreams; 1000 sampled programs per task, each checked exactly; two attempts per task as ARC scores it, on ARC-AGI-1 public eval.
3. One library round, then rerun.
4. Fair rule: demo pairs of eval tasks may be used at test time (TRM's numbers include that), and this must be stated next to the score.

**Что опровергает.** Any of these kills the idea:
- The depth-3 audit solves fewer than 8% of ARC-AGI-1 eval tasks.
- The proposer adds fewer than 5 points over the audit.
- A library round adds fewer than 2 points.
- Headline: fewer than 15% on ARC-AGI-1 eval after two rounds, i.e. not above CodeIt's 14.75% with a 7M proposer.

**Судьи.** generality 6: This is the purest form of the test: every ARC task is a new machine, primitives added after the freeze must stay at zero, and the score is measured against CodeIt. Proposer, exact executor and library is the right loop. Against it: the grid DSL is the designer's list, depth-3 enumeration is icecuber territory, it is 10 weeks long, and it is far from the project's story families and reading. / feasibility 2: It is budgeted at 10 weeks, and ARC needs a broad DSL plus a proposer. Beating CodeIt's 14.75% after two library rounds will not happen within 2-4 weeks. Only the depth-3 coverage audit fits the window. / evidence 4: ARC-AGI-1 is the most respected public benchmark here, the fair-use rule is stated, and CodeIt (14.75%), TRM (44.6%) and AlphaProof are real references. But the headline bar of beating 15% is below known DSL brute-force search (icecuber's 2020 Kaggle win at about 20% on the hidden set) and far below TRM, so a 'positive' result convinces no one. The designer's prior exposure to the public eval tasks leaks into the DSL, and the plan takes 10 weeks.

### 21. A reader that asks: consult the external base only where views and invariants disagree; distil structure, not answers (reader-4, сумма 11, недель 1.5)

**Что.** 1. **Rank phrase kinds.** For each phrase kind (path signature) in unlabeled text, score frequency x disagreement, where disagreement is between the word reader, the lexicon and the invariants of idea 3.
2. **Ask about the top N only.** Ask a teacher: an LLM through the API (Haiku already wrote the authors), or the owner by hand for about 100. The question is in the machine-independent format of idea 1, for example: "In '<0> went to dinner with her daughter <1>', which word relates them and from whose side?"
3. **Where answers go.** Into exact lexicon rows (the external base), and as lessons for the small reader. Never as a per-dataset label.
4. **Baseline.** Plain distillation: the teacher labels 20k random pairs from many free texts in the same open format (relation phrase + side + coreference): TinyStories, Simple Wikipedia, the free/stories paraphrases, the 24 authors. The student is the best encoder of 30M weights or less from idea 2.

**Первый опыт.** - **Data:** CLUTRR 1.3, gen234, and StepGame unseen authors.
- **Compare:** targeted N in {50, 200, 1000} against random N, against 20k distilled labels.
- **Reader:** the path-lexicon plus small reader from idea 1.
- **Cost:** CPU, plus 1-2k teacher calls, or a Colab A100 hour for the 20k student.

**Что опровергает.** - 200 targeted questions give under +4 link points over idea 1 alone; or
- random 200 give as much as targeted 200, so asking adds nothing; or
- 20k distilled labels into a student of 30M weights or less stay under 88% of links on gen234.

Then the teacher route does not make a small reader good.

**Судьи.** generality 2: It outsources the hard readings to a teacher LLM. Useful as an 'external base', but it moves the designer's role to the teacher and produces no machine. / feasibility 6: It is short (1.5 weeks). Distilling 20k labels from an LLM teacher into a 30M or smaller student, aiming at 88% links, is plausible given RoBERTa-base's 90%. Haiku access was used before. Active learning beating random sampling by a margin is often marginal, and the masloжor-moved-into-the-teacher objection limits how convincing it is. It can be done, but the headline is modest. / evidence 3: Active learning against random sampling is well studied, with typically modest gains (Lewis & Gale 1994, Settles 2009). Using an LLM teacher invites the decisive objection that the data-hungry model has simply moved into the teacher. The falsifier thresholds are small, and the benchmarks are noisy CLUTRR and in-house StepGame authors.

### 22. Законы алгебры как учитель без ответов: распространить замыкание с одного счётчика на все машины (unify-4, сумма 10, недель 2)

**Что.** У каждой алгебры языка есть законы, которым обязан подчиняться текст без ответов:
- Zⁿ (группа) — сумма по любому циклу равна нулю (нынешний `Register.close`);
- FIN — согласие разбора при любой расстановке скобок (ассоциативность) и обратные пары (если a есть R для b, то b есть conv(R) для a);
- RESET — функциональность (одно значение за раз: предложение, дающее два значения сразу, прочитано неверно) и постоянство до следующей записи;
- NAT и количества — сохранение при «передал»;
- SORT — транзитивность и антисимметрия.
Законы используются:
(а) чтобы читать невиданные формулировки во всех машинах — сейчас только у счётчика (оговорка 2 этапа 44);
(б) чтобы выбирать алгебру или каркас без ответов — по тому, какие законы непомеченные рассказы выполняют лучше;
(в) чтобы учить читателя на непомеченном тексте (круг этапа 40) за пределами StepGame.
Приём этапа 43 «одинаковые предложения читаются одинаково» — это и есть закон функциональности RESET, только переодетый.

**Первый опыт.** CLUTRR по тексту: one.py-таблица плюс Z-градуировка из идеи 2. На тестовых рассказах без ответов сводить виды предложений законами «цикл даёт ноль в Z-части» и «ассоциативность таблицы». Наборы: rob_sup (поддерживающие факты дают второй путь) и gen23. Сравнение с нынешними 56–59 % (1.0) и 87/75/77/71 (1.3). CPU, 1–2 часа.

**Что опровергает.** Опровергает любое из:
- меньше +3 пунктов на тексте CLUTRR после чтения теста без ответов;
- законами определено меньше половины новых формулировок.

**Судьи.** generality 3: Using algebraic laws as a teacher without answers is a real mechanism for reading new phrasings. It produces no machines, and CLUTRR chains have almost no cycles to use. / feasibility 4: It is a short CPU experiment, but CLUTRR chains have almost no cycles: closure only bites in the rob_sup noise. The Z part covers only generation. The main loss (14.7% of links read as 'nothing said') is untouched by laws. A +3 gain is uncertain, and stage 44's caveat already predicts this. / evidence 3: Real roots (Chang et al. 2007, Ganchev et al. 2010, Blum & Mitchell). But the test is +3 points on CLUTRR text, whose chains have almost no cycles (the idea admits it), against the project's own baseline. Too small and too noisy to convince.

### 23. Discrete thoughts: a looped reader that writes one exact record per turn (Coconut/Huginn with the latent replaced by memory) (recurrent-4, сумма 10, недель 5)

**Что.** The reader is the only weighted part and the measured weak point: 80% per link with small readers, 90% with RoBERTa-base.

Make it recurrent in the Huginn/TRM way, with the story given anew every turn, but carry nothing in a vector between turns. The carried state is the exact memory: the records written so far, plus the machine's verdict (consistent, contradiction at link k, or an answer).

Each turn:
1. The input is the story tokens plus the exact state rendered as tokens.
2. The output is one hard pointer pair into the story (two thing positions) and one choice from the few relation options, as one.py's reading does. It is written as a record.
3. The exact machine (the walk now, the cascade of idea 1 later) recomputes the closure.
4. A contradiction is shown at the next turn, so the reader re-reads that pair: TRM's refinement loop with the exact machine as critic.

The loop halts at a fixed point: no new or changed record.

Teaching from answers only: the existing gradient through the exact machine for the relation choice (stage 42, text.mind), plus a MAPO-style buffer of verified readings for the pointers. Start with the same RoBERTa-base (125M); later a decoder of about 0.2B parameters, the owner's size.

**Первый опыт.** CPU code, then Colab A100 for about 6 hours.

CLUTRR 1.0 (two tests) and CLUTRR 1.3 inside one.py, answers only:
- RoBERTa-base as a looped reader against the current one-pass RoBERTa reader: 62.7% and 64.8% on 1.0; 89, 77, 76, 79% on 1.3. DSR-LM: 61.0 ± 2.6.
- 3 seeds.

Then two benchmarks the project has not tried:
- ProofWriter depth-5;
- the free-text paraphrases of bAbI already in newtasks/.

**Что опровергает.** Any of these kills the idea:
- CLUTRR 1.0 test 1, answers only, below 66% (less than +3 over 62.7%).
- Link accuracy with RoBERTa-base below 92% (it is 90% now).
- Accuracy changes by more than 1 point between 4 and 32 turns. Then the loop is not using the machine's verdicts, and the idea has failed even if the score is fine.

**Судьи.** generality 3: A looped reader with the exact machine as critic is about reading inside CLUTRR, where the critic rarely fires. The machine is still given. / feasibility 3: Five weeks. Learning pointers from answers alone is hard credit assignment, CLUTRR chains rarely trigger the critic, and the 1.0 ceiling is noisy. The falsifier asks for +3 and 92% links with a measurable effect of the loop depth. The odds are low within the window. / evidence 4: Measuring 4 vs 32 turns is a good mechanistic check. The references hold up: Huginn and Ouro are real, and Parcae is real (arXiv 2604.12946, ICLR 2026), though the idea cites it only through another paper's id. But the headline is +3 points on CLUTRR 1.0 against DSR-LM's 61 +- 2.6, within the noise and the label errors. Experts will file it with PAL- and ToRA-style tool use unless the fixed-point critic is shown to cause the gain.

### 24. Hard cleanup inside the loop: exact machines as a neural layer (Hopfield/VSA), drift-free by construction (forgotten-old-3, сумма 9, недель 2)

**Что.** This is the form a ~0.2B model would take: the exact state lives inside the network. After every turn of a looped block (loop.py's Looped), the state is snapped to a codebook by a modern Hopfield read, s <- softmax(beta * s * C^T) * C, with beta annealed towards hard. C holds node identities and register values bound by HRR/VSA (node bound with value), so the loop's state is always close to a legal discrete state. That is loop.py's pointer-and-register invariant, learnt rather than hand-coded. Second, today's soft register in one.py and walk.py keeps a distribution over (2k+1)^n cells. With fractional-power (Fourier) encoding, adding an offset multiplies characteristic functions, so a distribution over Z^n costs d numbers for any n, exact up to aliasing. That makes 3-D worlds or multi-attribute kin cheap on a GPU. The cells of idea 1 all have VSA forms: counter = fractional power, cyclic = roots of unity, permutation = slot permutation, reset = unbind and rebind.

**Первый опыт.** 1 day on CPU: loop.py's small task (--nodes 24 --fields 1 --kinds 2 --whats 1) without the --every hint, Looped plus cleanup after every turn, 5 seeds of 40 minutes as in stage 34. That rival got one seed in five to 16 steps even with the hint, and lost points at 64 turns. Then the full loop.py task on a Colab A100, where the plain Looped never learnt a single step. Separately, re-implement walk.Walk.forward in the frequency domain: check equality on StepGame with n=2, then time it at n=4 and n=6.

**Что опровергает.** Kill it if, with cleanup and without the hint, fewer than 4 of 5 seeds reach 95% at 16 steps. Kill it if 64 turns lose more than 1 point against 8 turns (overthinking stays). Kill it if on the full task the A100 run still does not learn one step. For the Fourier register, kill it if the hardened answers differ from the exact soft register on StepGame.

**Судьи.** generality 2: A Hopfield/VSA cleanup layer addresses drift and the neural form of the machine. It does not address who designs the machine. It is tested on one loop.py task. / feasibility 4: The Fourier register equivalence check is easy, but convinces no one by itself. The core bet, that Hopfield cleanup makes a looped net learn loop.py's routing without the hint in 4 of 5 seeds, is shaky: stage 34's looped net failed at learning, not only at drift, and cleanup does not fix credit assignment. It is fast to try (2 weeks) but has a low chance of a convincing positive. / evidence 3: The references are real (Ramsauer 2021, Plate, Frady et al. 2020, Bansal 2022), but the test task is loop.py's own synthetic task, and the rival is the project's Looped transformer, which never learnt it. That is a broken baseline, so a win proves little. The Fourier-register check is only an equality test.

## Что упустили все идеи (судьи)

### generality

1) At least five ideas are the same Krohn-Rhodes cascade: cognition-1, forgotten-old-1, recurrent-1, red-team-2, and unify-1/unify-2 in part. Their bottleneck is not expressibility, which the completeness argument already covers. It is identifiability, and that is already measured here: a dependent cascade (turtle) is not learnt from final answers at 80-400 stories; on CLUTRR, K=6 fits 100% of teaching and gets 67.5% on test; at N=200 the gradient finds the wrong finite control. Learning theory says the same thing: minimal automata from given examples is hard, and membership queries (Angluin's L*) make it tractable. So asking for minimal pairs and intermediate supervision must be the core of the learner. Only forgotten-old-1 and cognition-4 treat asking as more than an add-on.

2) The circle has two halves, the machine and the reader/adapter. Every cascade or algebra is complete only over a symbol alphabet the reader supplies: sentence kinds plus argument slots. Those readers are still per-family (templates in recall, networks elsewhere). A sealed family written by outsiders in free text needs event segmentation and slot binding done by the system with zero code. Every sealed protocol must count reader and adapter edits as commits too; red-team-1 alone notes that even a Dyna-bAbI adapter is an edit.

3) Every hypothesis space (cell grammar, fold.py, algebra.py, the ARC DSL) is designed by the designer after seeing the tasks, so the circle just moves up a level, to 'add a primitive'. The missing shared step 0 is to build now, before implementing any idea, the stage-22 thermometer that still does not exist: a standing sealed pool of new families written by outside authors (several model vendors and humans) who never read these ideas. The pre-registered metric is commits per new family and primitives added after the freeze, and the space may be extended only by the system's own library or invention, never by a commit. Each idea now invents its own protocol on mostly already-seen sets.

4) The known blind-loss boundary is outside the finite-state basis: order in time as conditions (about two thirds of blind losses), stacks and recursion, variable binding, quantifiers. Library learning by composition cannot invent a new type of state, as red-team-1 admits. No idea has the system grow its own state type when it says 'not mine'. The calibrated 'sure / not sure' signal (99.5% accuracy where sure) is the asset to build that on.

5) No idea uses the designer's own trace as training data. That trace is the about 12 post-blind patch commits, the 70 moves and the 3 machines. A small code-writing proposer trained on it would literally learn to do the designer's step: write a candidate machine in an open, Turing-complete language (not a fixed DSL), with exact verification and carry-over as the filter. red-team-1 uses this history only for evaluation.

### feasibility

1. Five ideas are one idea: a Krohn-Rhodes cascade or a single algebra replacing the machines (cognition-1, forgotten-old-1, unify-1, recurrent-1, red-team-2). They share one measured blocker. Learning a dependent (wreath) cascade or a finite control from final answers alone fails: turtle.py and downstream.py fail at 80-400 stories, and in unify-2 K=6 explains all teaching and gives 67.5% on test. Only forgotten-old-1 (exact enumeration, intermediate questions, asking) and unify-2 (minimization by MDL instead of the gradient) attack it. That should be the first work item, done once, before any merge.

2. No idea starts the sealed blind pool on day 1. Every 'out of the circle' claim needs families written by people other than the designer, by authors from different model families (stage 32 found twelve 'authors' were one model). That costs calendar time, not CPU, so it should start in parallel now. The 'thermometer' (a sealed pool of 200+ tasks) is still unbuilt.

3. The 'external base' half of the owner's picture gets no experiment. Facts by lookup (retrieval or a calculator inside the reasoning loop) is covered only obliquely, by reader-4's teacher and the constants table in red-team-4.

4. The rival is still broken (rival.py at 18% at k=1). A convincing claim to an outside reader needs a working baseline on the new families: a reproduced known recipe, or a small pretrained LM fine-tuned or prompted few-shot (scratchpad/brainstorm2/outside_icl.py is a start). No idea budgets that, except as a footnote.

5. The algebra ideas are tested on symbols handed to them or on templated text. The measured bottleneck on free text (about 80% per link) sits only in the reader ideas. Nothing tests an algebra idea and a reader idea together on one new human-written family.

6. Nobody trains even a small end-to-end model of the owner's form (one network reads, the exact state is carried, at about 0.2B). Only forgotten-old-3 and recurrent-4 gesture at it, and both score low on feasibility.

### evidence

1) None of the ideas first repairs the rival. All 22 ideas rest on published numbers from other protocols or on own transformers that are broken (18% at k=1 on StepGame against the paper's 45%). No 'imba' claim will survive review until one size- and data-matched neural baseline reproduces a published number, such as UT on StepGame or TRM on Sudoku. That reproduction should be step zero. 2) Leakage through the designer's knowledge. The designer already knows bAbI, CLUTRR, StepGame, BBH, ARC and CLRS, so 'frozen code' still carries primitives fitted to them. Convincing tests need benchmarks released after the code-freeze hash (fresh ARC-AGI-2/3 tasks, new human-written sets), pre-registration of the falsifiers in a commit before the run, and a count of every adapter and reader edit as a 'patch'. 3) The 'external base' half of the owner's idea is almost untested. Only red-team-4 touches a calculator; nothing tests retrieval or tools over facts outside the weights. That needs a public open-domain multi-hop QA benchmark with retrieval, against a small RAG baseline matched in size. 4) Compositional generalisation in language (SCAN, COGS) and reading free, non-templated text are where 'small and exact' is most contested. Most ideas stay on templated or LLM-written stories, which experts consider solved by LLM+ASP pipelines. 5) Statistics: single seeds, 2-point thresholds on noisy CLUTRR 1.0 labels, and no confidence intervals. 6) No idea gives a single cross-benchmark measure an outsider can read. 'Human patches per new family, on families released after the freeze' should be reported next to accuracy and the number of training examples. Paths: /home/user/MX-Project/research_m5/README.md; the 11 red-team-1 commits were checked with git log in /home/user/MX-Project and all exist."

## Сведённый план (как вернул синтез)

### Круг

Что это за круг. Каждый новый набор до сих пор выигрывал я, проектировщик: смотрел на формат, писал новую точную машину или часть (регистр под StepGame, таблицу под CLUTRR, вспоминание под bAbI, двенадцать заплат после слепых прогонов: 7d790a7, 45722bc, 62098af и другие). Система выбирает только из того, что я написал (one.py выбирает из трёх машин), и между наборами учится программист, а не ученик (это записано ещё на этапе 22). Отсюда сильные, но «узкие» числа. Слепые наборы восемь раз подряд дают 86–92 % и примерно два новых вида провала на набор. Измерено, что круг держат три вещи. (1) Пространства машин нет: есть список точек. (2) Читатель свой у каждой машины, и выученное о языке не переносится. (3) Ничего не сохраняется от набора к набору. Есть и четвёртое: пятая идея из пяти «одна алгебра вместо трёх машин» упирается не в выразимость, а в узнаваемость. Каскад, в котором шаг зависит от другого состояния (черепаха с курсом), по одним конечным ответам не находится (turtle.py, downstream.py). На CLUTRR управление из 6 состояний объясняет 100 % учебных рассказов и даёт 67,5 % на тесте.

Выход одной фразой: машину должна строить сама система, а не я. Новая машина строится поиском в маленьком полном базисе (счётчики и сбросы Крона–Роудса, их каскады, указатель по точной памяти, монотонная неподвижная точка) сквозь точное исполнение. Неоднозначность решается вопросом о минимальной паре рассказов, а не градиентом. Найденное копится в общей библиотеке. Мерило успеха — число правок человека (включая читателя и переходник) на семейство, написанное чужими руками после заморозки кода.

### Архитектура

Идея владельца в одной фразе: маленький «умный человек с внешней базой». Логика стоит ноль весов и исполняется точно. Факты и значения слов лежат во внешней точной базе. Веса нужны только чтению, и их число меряется, а не назначается (0,2 млрд — гипотеза, а не цель). Части такие.

1. Точная память (0 весов). Это и есть «внешняя база»: записи (t, символ, вещи, числа), граф связей, числа; рядом инструменты (калькулятор, таблица констант вроде «60 минут в часе», позже поиск). Это нынешние records/recall/text.together, сведённые в один журнал.

2. Читатель (единственное место с весами, общее для всех машин). (а) Грамматический модуль: кто с кем связан, каким словом, с какой стороны, кореференция, разбиение рассказа на события. Для начала это путь в дереве зависимостей (spaCy, 12 МБ) плюс маленькая предобученная сеть в слоте text.Encoder; её размер подбирается по лестнице 4M–125M. (б) Значения слов: строки точного словаря с ключом по подписи пути. Словарь растёт и хранится между наборами (сейчас whole-sentence view совпадает с учебным в 4,7 % случаев, подпись пути — в 58 %). (в) Точные регистры по рассказу: пол и тождество людей. Выход читателя не зависит от машины: символ события и его аргументы.

3. Одно исполнитель вместо трёх машин. Машина = ВЫБОРКА × СВЁРТКА × ВЫХОД.
- Выборки: TIME по ключу-вещи, PATH между двумя вещами, ORBIT «идти по полю до вида».
- Свёртка в каскаде до 3 клеток: счётчик Z (по знаку), кольцо Z_m, перестановки S_k, сброс/триггер над значениями или над указателем, множество. Действие нижней клетки выбирается по паре (символ, состояние верхней): это сплетение, которое даёт черепаху и календарь.
- Запросы: GET / INV / COUNT / EXTREME / SCAN.
- Выход: таблица «проекция состояния → слово».
Нынешний Register — это каскад (счётчик, счётчик). Tabled — счётчик поколения плюс триггеры пола и супруга (cog/kin_cascade3.py: 22 состояния, 95–98 % на 4–10 звеньях). Recalled — банк сбросов с ключом по вещи. loop.py — указатель плюс NAT.

4. Как учится новое семейство (основное ноу-хау).
- Сначала свободная таблица на коротких рассказах.
- Из неё алгебра выводит структуру: целочисленное ядро уравнений композиции даёт счётчики (поколение CLUTRR нашлось на 9074 из 9074 рассказов, хватает 30), дробление разбиения по Муру даёт конечное управление.
- Верхние клетки перебираются точно (символов мало), нижние подгоняются прежним рецептом: мягко, затвердить, polish.
- Кандидаты сравниваются одной оценкой в битах (MDL) и переносом с коротких рассказов на длинные.
- Если два лучших кандидата одинаково объясняют данные, система строит кратчайший рассказ, на котором они расходятся, и спрашивает только его ответ (L*, механизм спрашивания этапа 13). Промежуточные вопросы, где они есть в данных (bAbI), идут как плотный надзор.
- Логика, общая для всех форм, пишется один раз как «подъёмы»: множество возможных миров (да/нет/может быть), правила как связи, исполняемые до неподвижной точки, наследование от вида.

5. Внешний цикл, который сейчас у меня в голове.
- Тупик: калиброванное «не понял» (там, где ученик говорит «понял», точность 99,5 %).
- Диагноз: дешёвые алгебраические пробы на учебных рассказах. Перестановка двух предложений меняет ответ — значит, нужны перестановки; два раза одно и то же ничего не меняет — Z2; период m — Z_m; важно только последнее — сброс.
- Поиск в выбранной части базиса.
- Если ничего не подошло: попробовать добавить одну безымянную клетку (счётчик или флаг) и оставить её, только если она укорачивает описание в битах.
- Иначе честное «не моё».
- Найденное сжимается в библиотеку: частые подпрограммы становятся новыми словами с бо́льшим приором.

6. Где живут веса: только в читателе (и, возможно, позже в подсказчике, который по 20 рассказам называет каркас). Всё остальное — дискретные выборы и точные таблицы. Тысячи чисел учатся за секунды–минуты на процессоре.

7. Устойчивость на многих шагах. Состояние никогда не свободный вектор. Это либо распределение по конечным состояниям, двигаемое построчно-стохастическими матрицами (масса сохраняется, как в loop.py), либо множества вариантов, которые могут только сужаться (неподвижная точка гарантирована, по Кнастеру–Тарскому). Ответ — неподвижная точка, дрейфа и «передумывания» нет по построению. Сеть только предлагает, точная часть проверяет (схема AlphaProof на каждом масштабе).

### Шаги

#### 0. Термометр, запечатанный пул и починка соперника (параллельно со всем остальным) (недель: 1)

В первый же день пишется одностраничная спецификация формата «рассказ → вопрос → ответ» без слов о клетках и машинах. Её получают авторы, которые этих идей не читали: владелец, 2–3 живых человека, LLM трёх разных поставщиков. Каждый сам выбирает тему и пишет генератор (или 600 рассказов) на одно семейство: 300 учебных рассказов длиной не больше 3 звеньев, тест на 5/10/20 звеньях. Хотя бы два семейства — свободный текст, написанный людьми. Всё складывается запечатанным, я формат вижу, содержание нет. В отдельный коммит заранее записываются пороги и метрика: правки человека на семейство (читатель и переходник считаются), покрытие при точности не ниже 99 %, молчаливые ошибки, число спрошенных ответов, все зёрна. Одновременно чинится соперник: rival.py (сейчас 18 % при k=1) должен повторить опубликованное число UT на StepGame (около 45 % при k=1). Запасной вариант — scratchpad/brainstorm2/outside_icl.py, малая предобученная LM с дообучением или few-shot.

Проверка: К 4-й неделе запечатано не меньше 12 семейств не меньше чем от 3 разных источников; хеш спецификации и порогов закоммичен до первого прогона. Соперник даёт не меньше 40 % при k=1 на StepGame, не дальше 5 пунктов от статьи.

Провал: Если соперник не дотягивает до 40 % за неделю, своих трансформеров в сравнениях больше нет: только опубликованные числа плюс дообученная малая LM, и это пишется рядом с каждым сравнением. Если чужих семейств к 4-й неделе меньше 8, шаг 7 откладывается: без них заявить «вышли из круга» нельзя вовсе.

#### 1. Узнаваемость прежде всего: каскад, зависимый от состояния, по ответам плюс вопросы (недель: 2)

Это ядро победившей идеи и измеренное узкое место. Пишется cells.py: счётчик Z, кольцо Z_m (m≤12), S_k (k≤5), сброс/триггер, каскад до 3 клеток со сплетением. Узнавание идёт четырьмя путями. (а) Градуировки ищутся как целочисленное ядро уравнений композиции (unify/grading.py). (б) Верхняя клетка перебирается точно (8^|символов|, с отсевом по перенумерации состояний), нижняя подгоняется градиентом регистра walk.py из нескольких начал и полируется. (в) Выбор между кандидатами по битам (MDL) и переносу, а не по градиенту: на CLUTRR при N=200 градиент нашёл неверное управление. (г) L*-вопросы: строится кратчайший рассказ, на котором два лучших каскада расходятся, и спрашивается только его ответ (не больше 40). Семейства: черепаха/navigate, одометр (день недели Z7 с переносом недели), граф CLUTRR (учение на 2–3 звеньях, тест на 10), шесть семейств с одной клеткой из kr/cascade.py.

Проверка: Черепаха и одометр: не меньше 99 % на длине 100 при не больше 400 коротких рассказах и не больше 40 вопросах. Граф CLUTRR без рукописной таблицы: не меньше 90 % по всем k при N=100 (у таблицы 63 %) и не меньше 95 % при k=10 при N=400 (у двуциклического каскада измерено 95–98 %; поправки на соглашения CLUTRR 1.0 о родственниках через брак хранятся как запомненные исключения, «двойной путь»). Шесть семейств с одной клеткой остаются на 100 % на длине 200.

Провал: Если черепаха ниже 90 % на длине 100 даже с 40 вопросами, каскад из ответов и вопросов не узнаётся. Тогда пространство машин сводится к одиночным клеткам плюс композиции через промежуточные вопросы, и обещание «система сама собирает зависимые машины» снимается. Это записывается, план идёт дальше с урезанной заявкой.

#### 2. Одно исполнитель в one.py вместо Register, Tabled и Recalled (недель: 2)

Каскад ставится в one.py за флагом. Число состояний растёт 1, 2, 3, 4, 6, 8, 12, потом добавляется счётчик, потом вторая клетка, пока затвердевшая форма не объяснит все учебные рассказы. Ничьи решает перенос с коротких на длинные и биты. Register.close обобщается на любую выученную алгебру: два пути между одними вещами обязаны дать один элемент. Так чтение без ответов работает и для родства, где его сейчас нет (оговорка 2 этапа 44). Вспоминание (latest value, how many, chain end) переходит в банки каскадов с ключом по вещи. Внешняя проверка: публичный генератор Delétang et al. (регулярные задачи) и перестановки S5 у Liu et al.

Проверка: На каждом наборе one.py, где сейчас не меньше 95 %, потеря не больше 2 пунктов: StepGame k=10 не ниже 93 %, bAbI 17/18/19 в пределах пункта, все 20 задач bAbI по среднему не хуже. Выбор машины совпадает с нынешним one.py не меньше чем на всех наборах, кроме одного. Регулярные задачи Delétang не ниже 99 % на самой длинной тестовой длине с 8 началами.

Провал: Потеря больше 2 пунктов на любом наборе, где отдельная машина давала не меньше 95 %: тогда единое исполнитель стоит точности, и три машины остаются, а не сливаются. S5 не выучен за 30 CPU-минут: тогда утверждение о масштабе снимается до тех пор, пока S5 не собран из малых множителей.

#### 3. Составные запросы и логические «подъёмы» (недель: 2)

Над свёрнутым состоянием одна библиотека запросов: GET, INV, OTHER, COUNT, EXTREME/MEMBER, SCAN. Программы ищутся перебором до глубины 3–4 (tiny_isa.py уже взял bAbI 10/16/20 на 98,2/100/100 % со 150 вопросов). Подъёмы пишутся один раз и наследуются любой формой. (а) Множество миров: «или», «не», ответ да/нет/может быть, а прочтение, которое опустошает все миры, отвергается. (б) Правила как связи, исполняемые петлёй до неподвижной точки, при открытом мире. (в) Виды с наследованием. Порядок во времени — точечная алгебра {<,=,>} с распространением (PC-2) вместо перебора всех порядков в rows.py (11–150 минут на задачу, J7 не закончилась).

Проверка: bAbI 4/10/16/20 не ниже 95 % с не больше 40 рассказами; ни один набор не теряет больше 2 пунктов. ProofWriter (шаблонный, открытый мир): учение на глубине не больше 2, тест на глубине 5 не ниже 95 %, ParaRules отдельно, без порога, только записать. Задачи J и I9–I12: точность не ниже перебора миров минус 2 пункта, время ниже хотя бы вдесятеро.

Провал: Если верный каркас не находится среди первых 200 перебранных хотя бы на 3 наборах из 24, язык запросов не компактен и снова превращается в список. Если ProofWriter глубины 5 ниже 90 %, подъём правил не даёт глубины по построению.

#### 4. Один читатель для всех машин: путь, словарь, инварианты (недель: 2)

text.view заменяется видом по пути в дереве зависимостей (paths.py и sex.py из scratchpad/reader_lens). Читатель выдаёт символ, сторону и «ничего не сказано», а не выбор конкретной машины. Пол и тождество берёт точный регистр. Значения слов — строки словаря с ключом по подписи пути и откатом к (лемма, крайняя дуга) и затем к маленькой GRU; строки учатся счётом сквозь затвердевшую машину. Для счётчиков слова становятся операторами: элемент предложения — произведение элементов слов, замыкание голосует за слова, а не за целые предложения. Шаблоны recall.py заменяются тем же читателем. Учение без ответов идёт инвариантами: один пол на человека, обратные пары, функциональность (одна мать), замкнутые циклы, согласие двух независимых видов. Принимается только то, что прошло все проверки.

Проверка: Маленькая GRU по пути, учение по связям: больше 82 % связей на gen23 (сейчас 80,5) и не меньше 86 % ответов на CLUTRR 1.3 (сейчас 83,2). Внутри one.py по одним ответам не ниже 60 % на gen23 (сейчас 56,1). Невиданные авторы StepGame по словам не ниже 87 % (сейчас 84,8). bAbI без шаблонов в пределах 2 пунктов. Точность уроков, принятых инвариантами, не ниже 90 % по проверочной разметке.

Провал: Выполнено любое из условий провала reader-1: связи не выше 82 % и при этом ответы на 1.3 ниже 86 %, либо в one.py ниже 60 %, либо невиданные авторы ниже 87 %. Тогда общий читатель без весов не тянет, и всё чтение свободного текста ложится на шаг 5 (веса и слова предобучения).

#### 5. Сколько весов и слов нужно читателю, и внешняя база на арифметике (Colab) (недель: 1.5)

Лестница размеров в одном и том же слоте Encoder и по одному протоколу: bert L-2/H-128 (4,4M), L-4/H-256 (11M, уже 84,4 % связей), L-4/H-512 (29M), L-8/H-512 (41M); ось слов при той же архитектуре: roberta-med-small-1M, roberta-base 10M, 100M, 1B; BabyLM ltgbert-10m и -100m; опора — roberta-base (90,1 %). Всё на gen23, gen234, CLUTRR 1.3 и StepGame с невиданными авторами, по 3 зерна, с доверительными интервалами. Вторая половина: SVAMP. Учение на MAWPS+ASDiv-A (3138 задач) по одним ответам; читатель выдаёт программу на языке счетов; константы берутся из внешней таблицы; исполнение точное; числа выделяются только регулярным выражением. Сравнение с читателем этапа 2 (68,5 %) и с RoBERTa-решателями из статьи SVAMP (43,8 %).

Проверка: Модель на 100M слов и не больше 30M весов даёт не меньше 84 % связей на gen23. SVAMP с RoBERTa-base плюс точный исполнитель не ниже 80 %. Ожидания для каждой точки лестницы записываются до прогона.

Провал: Если все модели не больше 45M весов и не больше 100M слов остаются на 82 % и ниже, чтению нужны миллиарды слов, и половина гипотезы «мало данных» для чтения неверна; это пишется в «Рабочую гипотезу». Если SVAMP ниже 75 %, «читатель плюс точная часть» на арифметике не даёт заявленного, и GSM8K не трогается.

#### 6. Внешний цикл: тупик → диагноз → поиск → вопрос → библиотека, с проверкой на моей же истории (недель: 2)

self.py оборачивает one.py. Тупик — калиброванное «не понял». Диагноз — алгебраические пробы на учебных рассказах. Поиск идёт в выбранной части базиса, дешёвое сначала, с L*-вопросами. Рост: одна безымянная клетка, которая остаётся, только если укорачивает описание в битах. Иначе «не моё». Сжатие: частые подпрограммы решённых задач (например, «крайний из множества по ключу», написанный руками 14 раз) антиунификацией становятся словами библиотеки (не больше 20), принимаются по битам и переносу на рассказы в 3 и 10 раз длиннее. Всё хранится в ground.json: словарь, счёт форм, приоры. Проверка — откат по git на 12 заплат после слепых прогонов (7d790a7, 45722bc, 62098af, 15656cf, bb0bf77, cefc224, b084029, 9b44fe9, 8fddfe6, 7147602, 5f04428 и ещё одна). Для каждой берётся код родителя, цикл запускается на задачах, ради которых делалась правка. Заранее, вручную, каждая заплата помечается: «композиция существовавшего» или «новый тип состояния».

Проверка: Восстановлено не меньше 6 заплат из 12, не больше 30 CPU-минут на каждую. Тест на их задачах не дальше 2 пунктов от исправленного кода. 104 рабочие задачи не теряют больше 0,5 пункта. Восстановлена хотя бы одна заплата из помеченных «новый тип состояния» (через рост безымянной клетки).

Провал: Восстановлено 3 и меньше: внешний цикл не заменяет меня. Если восстановлены только заплаты-«композиции», круг закрыт наполовину: новые типы состояния по-прежнему требуют человека, и это честная граница, которая записывается до шага 7.

#### 7. Заморозка и один выстрел по запечатанному пулу (недель: 2)

Код замораживается по хешу. Ни одной правки, включая читателя и переходник. Прогон: 12+ запечатанных семейств шага 0, Dyna-bAbI (сделан, чтобы ломать такие системы; советовался на этапах 22 и 32 и так и не был запущен), BBH web_of_lies, navigate, tracking_shuffled_objects_3/5/7 и date_understanding (по ~40 учебных примеров), отрицательные контроли dyck_languages и boolean_expressions. Семейства идут по одному, в двух порядках. Абляция: то же с библиотекой, стираемой перед каждым семейством. Каждое число сопровождается правками человека (0), числом учебных примеров, спрошенных ответов и CPU-минут.

Проверка: Не меньше 8 из 12 запечатанных семейств дают не меньше 95 % на 10 звеньях при учении на не больше 3, при нуле коммитов. Там, где система говорит «понял», точность не ниже 99 %. На dyck/boolean «не уверен» не меньше чем на 90 % вопросов. Медиана спрошенных ответов на семействах 9–12 не больше половины медианы на семействах 1–4. Без библиотеки вопросов нужно хотя бы на 20 % больше.

Провал: Меньше 6 из 12 при нуле коммитов: это не общая архитектура, а хорошая библиотека машин для одного жанра, и так и пишется. Если библиотека ничего не даёт (абляция в пределах 20 %), «накопление как у человека» не подтверждено. Молчаливые ошибки больше 1 % там, где сказано «понял», лишают системы главного козыря, честного «я выучил».

#### 8. Colab: турнир против соперников при равных данных (недель: 1)

Тот же запечатанный пул на A100 против соперников (подробности в поле colab). Это единственный шаг, который даёт внешнему читателю сравнение «мало данных и точная логика против статистики».

Проверка: Описан в поле colab; пороги записаны в коммите шага 0.

Провал: Если дообученная малая LM с 40-кратными данными догоняет систему (в пределах 5 пунктов на 10 звеньях) на половине семейств и больше, преимущество «меньше данных — больше глубины» не подтверждено.

### Colab

«Запечатанный турнир на длину». Один прогон, пороги закоммичены до него (шаг 0).

Данные: 12+ запечатанных семейств чужих авторов (среди них хотя бы 2 семейства свободного текста, написанные людьми), плюс StepGame как контроль, что соперник починен. Учение: 300–500 рассказов длиной не больше 3 звеньев. Тест: 5, 10 и 20 звеньев, по 1000 рассказов.

Участники:
- (A) замороженная система (каскадное исполнитель плюс общий читатель плюс библиотека), не больше 40 спрошенных ответов, процессор;
- (B) декодер-трансформер с нуля, примерно 0,2 млрд весов, на тех же рассказах, по известному рецепту; засчитывается, только если на учебной длине даёт не меньше 95 % и если на StepGame повторил UT (не меньше 40 % при k=1);
- (C) малая предобученная LM (SmolLM2-360M или Qwen2.5-0.5B) с дообучением на тех же 500 рассказах;
- (D) та же LM, дообученная на 20 тысячах рассказов, то есть в 40 раз больше данных;
- (E) looped-вариант (Huginn/Ouro-рецепт, 0,2 млрд) на тех же 500, ответ на вопрос владельца о дрейфе: его точность при 4, 16 и 64 оборотах.

Три зерна, доверительные интервалы.

Числа, названные заранее:
- (A) не меньше 95 % на 10 звеньях на не меньше чем 8 из 12 семейств, ноль правок кода, не меньше 99 % там, где сказано «понял»; падение с 10 до 20 звеньев не больше 2 пунктов;
- (B) и (E) на 10 звеньях не выше 60 % в среднем; у (E) теряется больше 3 пунктов между 16 и 64 оборотами;
- (C) не выше 70 %;
- (D) не выше 85 %.

Победа засчитывается, если (A) выше (D) хотя бы на 10 пунктов на 10 звеньях на не меньше чем 8 семействах, при 40-кратно меньших данных и компьюте на порядки ниже.

Провал засчитывается, если (D) в пределах 5 пунктов от (A) на половине семейств и больше, или если (A) берёт меньше 6 из 12 без правок.

Отдельно в таблицу (не входит в вердикт): лестница читателя шага 5 на двух семействах свободного текста. Ответ на вопрос, где на самом деле нужны 0,2 млрд.

### Честные пределы

- Не «как человек» и не с нуля. Базис клеток (счётчики, кольца, перестановки, сбросы, указатель, монотонная точка) пишу я. Теорема Крона–Роудса гарантирует полноту только для конечных автоматов со счётчиками, а не для всего мышления. «Логика выучена» здесь значит «выбрана и собрана из рукописного базиса сквозь точное исполнение», а не выведена из сырых данных.
- Вне базиса остаётся то, что даёт около двух третей слепых потерь: порядок во времени как условие с выводом в обе стороны (точечная алгебра закрывает его лишь частично), стеки и рекурсия, связывание переменных, кванторы. Рост безымянной клетки добавляет только счётчик или флаг, а не новый тип состояния. Если шаг 6 это подтвердит, человек на новые типы состояния останется нужен.
- Узнаваемость зависимых каскадов из одних ответов пока отрицательна. Весь план опирается на то, что вопросы о минимальных парах это исправят. Если шаг 1 провалится, заявка сужается до одиночных клеток.
- Чтение свободного текста — 80–90 % на связь. На 10 звеньях 0,9^10 ≈ 35 %, поэтому «перенос на любую длину по построению» верен только при почти безошибочном чтении. Читателю нужны веса и предобучение, сколько именно — неизвестно, и шаг 5 может показать, что нужны миллиарды слов.
- 0,2 млрд — гипотеза, измеренная только на CLUTRR. GSM8K, ARC-AGI, SCAN/COGS, CLRS, открытый многошаговый поиск фактов (проверка внешней базы с поиском) в этот план не входят. На ARC эта система, скорее всего, близка к нулю без новых частей, и честно было бы поставить его как контроль «ноль без новой машины».
- Фронтирные LLM этот план не побеждает и не собирается: его козырь — данные и компьют на порядки ниже, точная длина и калиброванное «я выучил», а не широта знаний.
- Протечка остаётся: я знаю bAbI, CLUTRR, StepGame и BBH, и базис подобран не вслепую. Её смягчают только запечатанные чужие семейства, а LLM-авторы похожи друг на друга по стилю (этап 32).
- Статистика шумная: CLUTRR 1.0 с ошибками разметки, пороги в 2 пункта; нужны зёрна и интервалы, иначе выводы хрупкие.
- Срок: около 14 недель последовательной работы. Пул чужих авторов стоит календарного времени, а не процессора.

### Привито от других идей

- Из red-team-1: откат по git на 12 реальных заплат как ключ ответов, который нельзя подогнать задним числом (шаг 6); разметка заранее «композиция / новый тип состояния»; единица измерения «правки человека на семейство», включая переходник и читателя.
- Из red-team-общего и судей: запечатанный пул чужих авторов разных поставщиков с первого дня, хеш кода и порогов до прогона, все зёрна, Dyna-bAbI наконец запустить; починить соперника до любых сравнений (шаг 0).
- Из unify-2: искать счётчик как целочисленное ядро уравнений композиции (поколение CLUTRR — 9074/9074, по 30 рассказам), конечное управление — дроблением по Муру; выбирать по MDL и перебору, а не градиентом.
- Из recurrent-1: рост числа состояний 1→12 → счётчик → вторая клетка до первого объясняющего; обобщение Register.close на любую выученную алгебру; внешняя проверка генератором Delétang и S5.
- Из cognition-1 и cognition-4: «двойной путь» — запомненные исключения поверх каскада (соглашения CLUTRR о родстве через брак); рост безымянной клеткой (Carey); накопительная библиотека ground.json с абляцией «стёрто перед каждым семейством» и кривой вопросов на 95 %.
- Из cognition-3: логические подъёмы, написанные один раз (множество миров, правила как связи до неподвижной точки, виды); ProofWriter глубина 5 при учении на 2.
- Из unify-1: язык ВЫБОРКА × СВЁРТКА × ВЫХОД и запросы GET/INV/COUNT/EXTREME/SCAN; программы глубины не больше 3 на bAbI 10/16/20 (tiny_isa.py).
- Из forgotten-old-2: точечная алгебра времени с распространением вместо перебора всех порядков в rows.py.
- Из forgotten-old-4: тупик → диагноз алгебраическими пробами → подцель → кусок (Soar/GPS) как внешний цикл; отрицательные контроли dyck/boolean на «не моё».
- Из reader-1, cognition-2, reader-3: вид по пути зависимостей, точный словарь по подписи пути (это и есть «внешняя база» значений), слова как операторы, точный регистр пола, учение без ответов инвариантами с аудитом точности уроков.
- Из reader-2 и red-team-4: лестница размера и слов предобучения читателя; SVAMP с точным исполнителем и таблицей констант как первая проверка внешней базы и инструментов.
- Из recurrent-2: множества вариантов, которые только сужаются, как второй вид точного состояния с гарантированной неподвижной точкой. Sudoku/Maze — только как проверка устойчивости, не общности (их берёт поиск без сети).
- Из recurrent-3: на будущее, после шага 7, ARC-AGI-1 как аудит покрытия базиса (глубина 3, точная проверка на демонстрациях), а не как заявка.
- Отложено, но записано (замечание судей): учить предлагающую сеть на моей собственной истории (12 заплат, 70 ходов, 3 машины), чтобы она писала кандидатов-машины на открытом языке, а точная проверка и перенос фильтровали.
