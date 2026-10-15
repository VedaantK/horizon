from concordance.engines.hft_bridge import self_test


def test_hft_environment_self_test():
    out = self_test()  # raises unless every hand-computed hftbacktest case passes
    assert out.count("PASSED") >= 6
