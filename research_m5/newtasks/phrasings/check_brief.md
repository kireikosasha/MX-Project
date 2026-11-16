# Reading sentence patterns

Each file you are given is a JSON list of English sentences about two objects, AA and BB, on a flat board
seen from the front ("above" is up, "below" is down, left and right as the viewer sees them; on a compass
north is up and east is right; on a clock face 12 is straight up and 3 is to the right).

For every sentence decide where **AA stands relative to BB**. The answer is one of:

`left`, `right`, `above`, `below`, `upper-left`, `upper-right`, `lower-left`, `lower-right`

(`left` = AA directly to the left of BB at the same height; `upper-left` = AA diagonally up and to the left
of BB; and so on). If a sentence does not say it clearly, or fits none of the eight, answer `unclear`.

Work carefully, sentence by sentence: put BB in the middle and place AA as the sentence says. Mind sentences
that speak from the side of BB ("BB is to the right of AA" means AA is to the left of BB).

Write the answers as a JSON list of strings, in the same order and of the same length as the input list.
