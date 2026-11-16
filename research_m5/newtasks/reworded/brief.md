You are given a JSON file with sentence templates taken from small story tasks. In a template, every `*` stands for a
name of a person, a thing or a place, and every `#` stands for a number. Statements are plain sentences; questions are
templates of questions about the story.

For EVERY template (statements and questions of every task in the file) write 6 different rewordings that say the
same thing in other words.

Hard rules for a rewording:
1. It has exactly as many `*` and `#` as the template, and they stand for the SAME things IN THE SAME ORDER (the
   first `*` of your rewording is the first `*` of the template, and so on). So no rewording that swaps the roles
   (not "* was given the * by *" for "* gave the * to *").
2. Lowercase English words only, no punctuation, no question mark, plain everyday words; 2 to 12 words besides the
   `*` and `#`.
3. Use different verbs, prepositions and word order where the order of the `*` allows; do not just add "really" or
   "now". The six should differ from each other.
4. Keep the meaning exact: do not make it more general or more specific, keep negation, keep "before"/"after",
   "more"/"less" as they are in meaning.

Write the result as ONE JSON file: a list with an object for every task of the input, in the same order:
{"id": the task id, "statements": {template: [six rewordings]}, "questions": {template: [six rewordings]}}.
Every template of the input must be a key, copied exactly. Check with python that the file is valid JSON, that every
template is present and that every rewording has the same number of `*` and of `#` as its template; fix what is not.
Your final reply must be only the path of the file you wrote.
