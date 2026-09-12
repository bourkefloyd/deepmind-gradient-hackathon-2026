"""Word Hunt scoring: 3=100, 4=400, 5=800, 6=1400, 7+=1800."""

SCORES = {3: 100, 4: 400, 5: 800, 6: 1400}


def score_word(word: str) -> int:
    n = len(word)
    if n < 3:
        return 0
    return SCORES.get(n, 1800)
