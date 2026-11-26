# One family of sequence tasks (for an author who has not seen the learner)

Invent one rule that reads a sequence of tokens and gives an answer.

- Take the theme you are given and make the rule about it (a kitchen, a ferry, a card table...). Do not make it a
  well-known textbook exercise.
- `ALPHABET`: 2 to 8 tokens, short words.
- `answer(seq)`: for ANY list of tokens from the alphabet — any order, any length from 1 to 1000 — returns a short
  string. It must be deterministic, defined for every such list, and fast (one pass over the list is enough).
- The answer should be something a person could keep track of in their head while reading the sequence token by token,
  and it should depend on the sequence as a whole, not only on its last token and not only on its length. On sequences
  of length 20 drawn at random it must take at least two different values.
- Make it a rule a person needs to keep several things in mind for: at least two quantities that change
  independently of each other, and at least one of them can grow without a fixed limit (a count, a queue, who came
  first...). Keep it a rule a careful person could still follow token by token.
- Write `family.py` with `ALPHABET`, `answer` and a docstring that says the rule in one paragraph. Test it with a few
  sequences of your own.

How it will be used: a learner sees short sequences with their answers, may ask the answers of some sequences it chooses,
and is then tested on much longer ones. Nothing else about the learner is told to the author.
