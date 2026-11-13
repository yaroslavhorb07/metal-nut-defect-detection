from nutdetect.predict import is_uncertain


def test_low_threshold_does_not_flag_confident_good():
    # threshold 0.06: a 0.005 score is clearly good, 0.12 is borderline
    assert not is_uncertain(0.005, 0.06, 1.0)
    assert is_uncertain(0.12, 0.06, 1.0)
    assert not is_uncertain(0.99, 0.06, 1.0)


def test_band_is_symmetric_around_half():
    assert is_uncertain(0.6, 0.5, 1.0)
    assert is_uncertain(0.4, 0.5, 1.0)
    assert not is_uncertain(0.9, 0.5, 1.0)
    assert not is_uncertain(0.1, 0.5, 1.0)


def test_extreme_probabilities_do_not_crash():
    assert not is_uncertain(0.0, 0.06, 1.0)
    assert not is_uncertain(1.0, 0.06, 1.0)
