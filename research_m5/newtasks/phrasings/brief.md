# Sentence patterns for a spatial dataset

You write sentence patterns. Each pattern is one English sentence that says where one object stands
relative to another on a flat board seen from the front: "above" is up, "below" is down, "left" and
"right" are as the viewer sees them.

There are 8 relations. A relation always gives the position of **AA relative to BB**:

- `left` — AA is directly to the left of BB (same height)
- `right` — AA is directly to the right of BB (same height)
- `above` — AA is directly above BB (same column)
- `below` — AA is directly below BB (same column)
- `upper-left` — AA is diagonally up and to the left of BB
- `upper-right` — AA is diagonally up and to the right of BB
- `lower-left` — AA is diagonally down and to the left of BB
- `lower-right` — AA is diagonally down and to the right of BB

Write **20 patterns for each of the 8 relations** (160 in all).

Rules:

1. Every pattern contains the placeholders `AA` and `BB` exactly once each, and no other names.
2. Vary the wording and the structure of the sentence as much as you can. In about half of the patterns
   mention BB first. Example for `left`: "BB has AA on its left side." or "BB stands to the right of AA."
   (the second says the same fact from the side of BB — that is fine and wanted).
3. Be exact. No vague patterns ("near", "next to", "beside" alone). A pattern must not fit any other of the
   8 relations. A diagonal must state both of its parts clearly.
4. Write in your own style, given below. Do not reuse a sentence frame more than twice across relations
   with only the direction word changed.
5. Plain words only: letters, digits, spaces, commas, apostrophes, and a full stop at the end.

Save the result as JSON: an object whose keys are the 8 relation names and whose values are lists of 20
strings.

When the file is written, read every pattern again: put BB in the middle, place AA as the sentence says,
and check that the relation of AA to BB is the one the pattern is filed under. Correct what is wrong and
save the file again. Reply with one line: how many patterns you corrected.
