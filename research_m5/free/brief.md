You are given a JSON file with short stories. Each story is a list of sentences; a sentence has "s" (the text) and "a" (the answer if it is a question, otherwise null).

Your job: reword EVERY sentence in free, natural, varied English and save the result.

Rules:
1. The meaning must stay exactly the same; a question must keep exactly the same answer. One sentence in, one sentence out, same order, same number of sentences in every story.
2. Keep every name of a person, thing, place, team or colour and every number EXACTLY as written (the same word in the same form; digits stay digits). Do not add any new name or number.
3. Change everything else freely: other verbs, another word order, passive voice, small harmless words (then, now, just, later, in the end), another way to ask the question. Do NOT reuse one pattern: similar sentences must be reworded in different ways across the file.
4. Use only letters, digits and spaces, with a final full stop for a statement and a final question mark for a question. No apostrophes, hyphens, commas or quotes. At most 14 words per sentence.
5. Do not use a pronoun in place of a name (no he, she, it, they).

Output: write a JSON file at the output path given in your instructions, with exactly this shape:
{"task": "<the same task name>", "stories": [{"part": "...", "id": N, "sentences": [{"s": "<original>", "p": "<your rewording>", "a": <the same answer or null>}, ...]}, ...]}
Every story and every sentence of the input must be present. Write the file with a small Python script that holds your rewordings in a list and checks that the counts match, rather than typing the JSON by hand. When the file is written and the counts match, reply with one line: the output path and the number of sentences.
