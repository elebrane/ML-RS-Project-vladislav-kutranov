# 10 исследовательских вопросов: метод, расчёт, плюсы, минусы, источник

Общее для всех: данные FAR-Trans [1], топ-10 на клиента, метрика nDCG@10 по новым для клиента активам (допущение до утверждения протокола), бейзлайн — популярность как в [1].
Баллы (1–5, больше — лучше): Задание · Новизна · Выполнимость · Информативность · Бенчмарк · Риск. Это экспертная оценка, не измерение.

---

## 1. LLM-реранкинг против порядка популярности — 26/30 (5·4·5·4·4·4) · Ranker
**Вопрос.** We test whether zero-shot LLM reranking of the top-30 popularity candidates improves nDCG@10 over their original popularity order for FAR-Trans customers with transactions before the cut, counting only assets new to the customer.
- **Метод:** zero-shot listwise-реранкинг: LLM получает пользователя и список кандидатов и выдаёт их порядок, без обучения [2], [3].
- **Как считается:** в промпт — история клиента (названия и категории активов) и 30 пронумерованных кандидатов; модель возвращает 10 номеров по порядку; nDCG@10 сравнивается с теми же 30 кандидатами в порядке популярности.
- **Почему хорош:** не нужны обучение и GPU; модель использует знания о компаниях и связи между ними, которых нет в счётчиках. В поиске листовой LLM-реранкинг конкурирует с обученными ранкерами [3].
- **Слабые стороны:** смещение по позиции в списке и в сторону популярного; в [2] zero-shot LLM уступает обученному SASRec. Потолок качества — полнота кандидатов. Стоимость и время на каждого клиента. Контаминация.
- **Источники:** [2] LLMRank, [3] RankGPT.

## 2. LLM против LightGBM-ранкера с теми же признаками — 25/30 (5·4·3·5·4·4) · Ranker
**Вопрос.** We test whether LLM reranking improves nDCG@10 over a LightGBM learned ranker that receives the same history and asset features, on the same candidates, for the common evaluation users.
- **Метод:** learning to rank, LambdaMART — бустинг деревьев, где градиент каждой пары «релевантный/нерелевантный» взвешивается тем, насколько их перестановка меняет nDCG [4]; реализация LightGBM [5].
- **Как считается:** на каждую пару (клиент, кандидат) — признаки: популярность, совпадение категории и сектора с историей, давность последней покупки и т. п.; модель учится на прошлых срезах и ранжирует кандидатов на следующем.
- **Почему хорош:** самое честное сравнение «метод против метода» при одинаковой информации; быстрый инференс; прямо оптимизирует nDCG.
- **Слабые стороны:** нужны инженерия признаков и обучение на прошлых окнах; в статье FAR-Trans гибрид на LambdaMART (Hybrid-nDCG, 0.231) уступил популярности (0.271) [1] — его надо хорошо настроить.
- **Источники:** [4] Burges, [5] LightGBM, [1] FAR-Trans.

## 3. Настоящие названия против обезличенных описаний — 26/30 (5·4·5·5·2·5) · Ranker
**Вопрос.** We test whether showing real asset names to the LLM reranker, instead of anonymised category-and-sector descriptors, changes nDCG@10 for the same customers and candidates.
- **Метод:** анонимизация сущностей для оценки look-ahead bias: тот же промпт, но названия компаний заменены нейтральными метками [6].
- **Как считается:** вариант A — «MYTILINAIOS SA (stock, industrials)»; вариант B — «Asset 17 (stock, large cap, industrials, Athens)»; разница nDCG@10 на тех же клиентах и кандидатах.
- **Почему хорош:** меняется одна строка промпта; любой исход информативен; прямо измеряет контаминацию, которую иначе пришлось бы просто признать риском.
- **Слабые стороны:** одна абляция не отделяет законное знание (чем занимается компания) от знания будущего. В [6] анонимизация на периоде обучения модели даже повысила доходность стратегии: знание о компании может и мешать («distraction effect»). Сама по себе не поднимает бенчмарк.
- **Источники:** [6] Glasserman & Lin.

## 4. С историей клиента против без неё — 24/30 (5·3·5·4·2·5) · Ranker
**Вопрос.** We test whether including the customer's purchase history in the reranking prompt improves nDCG@10 over a prompt with the candidates only, for the same customers and candidates.
- **Метод:** абляция контекста в LLM-реранкинге [2].
- **Как считается:** два промпта, отличающихся только блоком истории; разница nDCG@10.
- **Почему хорош:** отвечает, персонализирует ли модель вообще или просто ранжирует по известности; дёшево.
- **Слабые стороны:** в [2] модели слабо используют порядок истории; у половины клиентов в истории один актив — сигнал небольшой; новизна ниже, частично изучено в [2].
- **Источники:** [2] LLMRank.

## 5. Перемешивание кандидатов в несколько проходов против одного — 21/30 (4·2·4·3·3·5) · Ranker
**Вопрос.** We test whether aggregating three LLM reranking passes over shuffled candidate orders improves nDCG@10 over a single pass in popularity order, for the common evaluation users.
- **Метод:** permutation self-consistency: список кандидатов перемешивают несколько раз, ранжируют каждый вариант и сводят результаты в общий порядок [7]; похожий приём «bootstrapping» — в [2].
- **Как считается:** 3 перестановки кандидатов → 3 ранжирования → сведение (например, по среднему месту) → топ-10.
- **Почему хорош:** снимает смещение по позиции; в [7] прирост до 7–18% для GPT-3.5 и до 34–52% для Mistral в задачах ранжирования.
- **Слабые стороны:** в K раз больше вызовов — хуже время, а время решает ничьи в бенчмарке; приём уже изучен [2], [7] — новизна низкая.
- **Источники:** [7] Tang et al., [2] LLMRank.

## 6. Последние 5 покупок против всей истории — 20/30 (4·2·4·3·2·5) · Ranker
**Вопрос.** We test whether a prompt with only the customer's last five purchases improves nDCG@10 over a prompt with the full history, for customers with at least six purchases before the cut.
- **Метод:** recency-focused prompting [2].
- **Как считается:** два промпта с разной длиной истории для одних и тех же клиентов с длинной историей.
- **Почему хорош:** в [2] длинная история ухудшала результат, а акцент на последних покупках помогал.
- **Слабые стороны:** клиентов с ≥6 разными покупками мало — узкая популяция и шумный результат; повторяет вывод [2].
- **Источники:** [2] LLMRank.

## 7. Текстовый эмбеддинг истории против ID в kNN — 20/30 (4·3·2·4·3·4) · Encoder
**Вопрос.** We test whether an LLM text embedding of a customer's purchase history improves nDCG@10 over an ID-based history representation in the same nearest-neighbour recommender, for customers with at most three distinct assets.
- **Метод:** текстовые представления вместо ID: история превращается в текст и кодируется языковой моделью [8], [9].
- **Как считается:** «купил: National Bank of Greece (банк), …» → эмбеддинг → косинусная близость клиентов → кандидаты оцениваются по покупкам ближайших соседей; контроль — тот же kNN на бинарных векторах покупок.
- **Почему хорош:** для коротких историй ID-векторы почти пустые, а текст переносит смысл; в [8] текстовые представления сильнее всего помогали редким (long-tail) товарам.
- **Слабые стороны:** нужна open-source модель эмбеддингов (из вашей среды не скачать — Kaggle или Colab); текст об активах бедный; выигрыш в [8] неустойчив по метрикам.
- **Источники:** [8] UniSRec, [9] Sentence-BERT.

## 8. Эмбеддинги описаний активов против категорий и сектора — 19/30 (4·3·3·3·3·3) · Encoder
**Вопрос.** We test whether LLM embeddings of asset descriptions improve nDCG@10 over one-hot category and sector features in the same content-based recommender, for customers who bought exactly one asset before the cut.
- **Метод:** контентная фильтрация с плотными текстовыми эмбеддингами вместо one-hot признаков [9], [8].
- **Как считается:** текст актива (название, тип, сектор, рынок) → эмбеддинг; профиль клиента — среднее эмбеддингов купленного; кандидаты ранжируются по косинусной близости; контроль — тот же метод на one-hot векторах.
- **Почему хорош:** работает там, где коллаборативному методу не на что опереться (половина клиентов купила один актив); активов всего ~320 — считать дёшево.
- **Слабые стороны:** у фондов сектор пустой — текст почти только название; названия несут контаминацию; та же зависимость от внешней модели.
- **Источники:** [9] Sentence-BERT, [8] UniSRec.

## 9. Новости как контекст реранкера (RAG, гипотеза H1) — 20/30 (4·5·2·4·3·2) · Ranker + RAG
**Вопрос.** We test whether adding news headlines published in the 30 days before the cut to the reranking prompt improves nDCG@10 over the same reranker without news, on the same candidates.
- **Метод:** retrieval-augmented generation — перед ответом найти документы и подать их модели в контекст [10]; гипотеза опирается на эффект внимания: частные инвесторы покупают акции, попавшие в новости [11], тон новостей несёт информацию [12].
- **Как считается:** для каждой компании-кандидата — заголовки за 30 дней до среза (фильтр по дате) → в промпт → реранкинг; контроль — тот же реранкер без новостей и популярность, пересортированная по числу упоминаний.
- **Почему хорош:** самая высокая новизна; прямая опора в финансовой литературе; это и есть тема Session 3.
- **Слабые стороны:** нет источника датированных новостей по греческим компаниям; утечка, если дата публикации определена неточно; контаминация; новости есть только у акций.
- **Источники:** [10] RAG, [11] Barber & Odean, [12] Tetlock.

## 10. Энкодер плюс ранкер против каждого по отдельности — 19/30 (4·3·1·4·4·3) · Hybrid
**Вопрос.** We test whether combining LLM-embedding retrieval with LLM reranking improves nDCG@10 over encoder-only and reranker-only variants and over popularity, for the common evaluation users.
- **Метод:** двухэтапная схема «отбор кандидатов → LLM-ранжирование» [13].
- **Как считается:** энкодер (вопрос 7 или 8) отбирает кандидатов, LLM их переставляет; минимум 4 прогона: бейзлайн, только энкодер, только ранкер, оба.
- **Почему хорош:** потенциально лучший результат для бенчмарка; в [13] ранкер давал прирост 4–12% над отборщиком при быстром инференсе.
- **Слабые стороны:** самый трудоёмкий; качество ранкера ограничено полнотой отбора [13]; обязательна абляция, разделяющая вклад частей.
- **Источники:** [13] LlamaRec.

---

## Источники
Статус на 2026-11-13: по тексту статьи прочитаны [1], [2], [6], [7], [8], [13]; по странице с аннотацией — [3]; [4], [5], [9], [10], [11], [12] найдены, библиографические данные сверены, но сами тексты не открывались. Цитировать в отчёте можно только то, что вы откроете сами.
[1] J. Sanz-Cruzado, N. Droukas, R. McCreadie, "FAR-Trans: An investment dataset for financial asset recommendation," Fin-RecSys @ IJCAI, 2024. https://arxiv.org/abs/2407.08692
[2] Y. Hou et al., "Large language models are zero-shot rankers for recommender systems," ECIR, 2024. https://arxiv.org/abs/2305.08845
[3] W. Sun et al., "Is ChatGPT good at search? Investigating large language models as re-ranking agents," EMNLP, 2023, pp. 14918–14937. https://aclanthology.org/2023.emnlp-main.923
[4] C. J. C. Burges, "From RankNet to LambdaRank to LambdaMART: An overview," Microsoft Research Technical Report, 2010. https://www.microsoft.com/en-us/research/?p=159784
[5] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," NeurIPS, 2017. https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
[6] P. Glasserman, C. Lin, "Assessing look-ahead bias in stock return predictions generated by GPT sentiment analysis," arXiv:2309.17322, 2023. https://arxiv.org/abs/2309.17322
[7] R. Tang, C. Zhang, X. Ma, J. Lin, F. Ture, "Found in the middle: Permutation self-consistency improves listwise ranking in large language models," NAACL, 2024, pp. 2327–2340. https://aclanthology.org/2024.naacl-long.129
[8] Y. Hou et al., "Towards universal sequence representation learning for recommender systems," KDD, 2022. https://arxiv.org/abs/2206.05941
[9] N. Reimers, I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," EMNLP-IJCNLP, 2019. https://arxiv.org/abs/1908.10084
[10] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," NeurIPS, 2020. https://arxiv.org/abs/2005.11401
[11] B. M. Barber, T. Odean, "All that glitters: The effect of attention and news on the buying behavior of individual and institutional investors," Rev. Financial Studies, 21(2), 785–818, 2008.
[12] P. C. Tetlock, "Giving content to investor sentiment: The role of media in the stock market," J. Finance, 62(3), 1139–1168, 2007.
[13] Z. Yue et al., "LlamaRec: Two-stage recommendation using large language models for ranking," arXiv:2311.02089, 2023. https://arxiv.org/abs/2311.02089
