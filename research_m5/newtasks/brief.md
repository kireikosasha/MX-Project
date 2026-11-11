Write ONE self-contained Python 3 script that generates a synthetic reading-comprehension task: short stories made of simple English sentences, with questions and answers, in the general style of the bAbI tasks but with your own world, words and sentence patterns. The kind of reasoning to test and the path to save the script are given in your instructions.

Requirements:
1. `python gen.py N SEED` prints N stories to standard output; all randomness comes from random.Random(SEED). Standard library only; no files, no network.
2. Output format, exactly: one sentence per line; each line starts with its number and one space; numbers restart at 1 for every story. A statement ends with a full stop. A question ends with a question mark, then ONE TAB character, then the answer. Example of the format only:
1 The key is in the red box.
2 The red box is on the shelf.
3 Where is the key?<TAB>shelf
3. Every story has 3 to 12 statements and 1 to 5 questions (in the middle or at the end); a question is answered only from the statements above it in the same story.
4. Answers are computed by your program from a simulation of the world, never written by hand, and must be unambiguous. An answer is one lowercase word, or yes / no, or a number in digits, or several lowercase words separated by commas without spaces when the answer is a list.
5. Keep the language small and regular: at most about 40 different words in the whole task; every kind of fact is said with 1-3 fixed sentence patterns and every kind of question with 1-2 fixed patterns; names and things vary inside the patterns. Only letters, digits, spaces, '.' and '?' (no apostrophes, hyphens, or commas inside sentences).
6. The task must need reasoning over several sentences (not looking up one sentence), and stories must differ from each other (random names, things, order, length).
7. Do not copy the wording of the original bAbI tasks.

Then run `python gen.py 5 1`, check the format and that the answers are right, fix any problems. Your final reply must be only: the path of the script and one sentence describing the task.
