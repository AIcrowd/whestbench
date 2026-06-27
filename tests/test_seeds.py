from whestbench.seeds import derive_estimator_seed, derive_seed_streams


def test_derive_estimator_seed_matches_known_anchor():
    # Anchor from the public eval dataset MLP "patricia-hawkins".
    assert derive_estimator_seed(6717184059027789272) == 2861657444


def test_derive_seed_streams_is_self_consistent_and_deterministic():
    w0, s0, est0 = derive_seed_streams(12345)
    w1, s1, est1 = derive_seed_streams(12345)
    # estimator seed is the third spawned stream, stable across calls
    assert est0 == est1 == derive_estimator_seed(12345)
    # the three streams are distinct substreams of the same root
    assert int(w0.generate_state(1)[0]) != int(s0.generate_state(1)[0])
