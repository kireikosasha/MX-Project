Write ONE self-contained Python 3 script that generates a synthetic reasoning task given as RECORDS, not as language: every line is a short row of codes and numbers (like a log, a ledger or a table), and some rows are queries with answers. The kind of reasoning to test and the path to save the script are given in your instructions.

Requirements:
1. `python gen.py N SEED SIZE` prints N episodes to standard output; SIZE is 1, 3 or 10 (default 1 when it is not given). All randomness comes from random.Random(SEED). Standard library only; no files, no network.
2. SIZE says how big an episode is. At SIZE 1 an episode has 3 to 12 records and 1 to 5 queries. At SIZE s it has about s times more records and, where the reasoning allows it, longer chains and bigger numbers - but exactly the same codes, the same record formats and the same pool of entity codes as at SIZE 1, and still 1 to 5 queries. Nothing may appear at SIZE 3 or 10 that cannot appear at SIZE 1 except longer episodes, longer chains and bigger numbers.
3. Output format, exactly: one row per line; each line starts with its number and one space; numbers restart at 1 for every episode. Tokens are separated by single spaces. A record ends with a full stop. A query ends with a question mark, then ONE TAB character, then the answer. Example of the format only:
1 put k1 b3.
2 put b3 s2.
3 at k1?<TAB>s2
4. No natural language. A row is 2 to 5 tokens: first an operation code, then entity codes and/or numbers. An operation code is 2 to 4 lowercase letters of your own invention and must NOT be an English word. An entity code is one lowercase letter followed by one digit (k1, b3); use one letter per kind of entity. Numbers are written in digits. No other tokens at all: no articles, no prepositions, no filler. Use at most 6 different operation codes for records and at most 3 for queries, and at most 8 entity codes of each kind.
5. A query is answered only from the records above it in the same episode. Answers are computed by your program from a simulation, never written by hand, and must be unambiguous. An answer is one entity code, or yes / no, or a number in digits, or several entity codes separated by commas without spaces when the answer is a list (in a fixed order, e.g. sorted).
6. The task must need reasoning over several records (not looking up one record), and episodes must differ from each other.
7. Do not copy the bAbI tasks.

Then run `python gen.py 5 1` and `python gen.py 2 1 10`, check the format and that the answers are right, fix any problems. Your final reply must be only: the path of the script and one sentence describing the task.
