"""Two job trays at a university computing center.

The computing center keeps two trays of punched-card jobs waiting for the
mainframe, one for the physics department and one for the chemistry
department; both trays start empty. The token "phys" drops one physics job
into the physics tray, and "chem" drops one chemistry job into the chemistry
tray. The token "run" is one operator shift on the mainframe: if both trays
hold jobs, the operator runs one job from each tray; if only one tray holds
jobs, the operator runs two jobs from that tray (or its single job if it holds
only one); if both trays are empty, nothing happens. After the whole sequence
has been read, the answer is "phys" if more physics jobs than chemistry jobs
are still waiting, "chem" if more chemistry jobs are waiting, and "even" if
the two trays hold the same number of jobs.
"""

ALPHABET = ["phys", "chem", "run"]


def answer(seq):
    phys = 0
    chem = 0
    for tok in seq:
        if tok == "phys":
            phys += 1
        elif tok == "chem":
            chem += 1
        elif tok == "run":
            if phys > 0 and chem > 0:
                phys -= 1
                chem -= 1
            elif phys > 0:
                phys = max(phys - 2, 0)
            elif chem > 0:
                chem = max(chem - 2, 0)
        else:
            raise ValueError("unknown token: %r" % (tok,))
    if phys > chem:
        return "phys"
    if chem > phys:
        return "chem"
    return "even"
