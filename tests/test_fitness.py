import evomatic.fitness as fitness


def test_compare_candidates():
    A = {"rank": 0, "crowding": 1}
    B = {"rank": 0, "crowding": 2}
    C = {"rank": 1, "crowding": 0}

    assert fitness.compare_candidates(A, B) == B
    assert fitness.compare_candidates(A, C) == A
    assert fitness.compare_candidates(B, C) == B
