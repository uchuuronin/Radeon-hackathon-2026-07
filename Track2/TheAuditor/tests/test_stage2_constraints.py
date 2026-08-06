from route import route_decision


def test_verified_high_agreement_auto():
    assert route_decision(
        verify_pass=True,
        agreement=0.98,
        tau_hi=0.95,
        tau_lo=0.70,
    ) == "auto"


def test_failed_verification_escalates():
    assert route_decision(
        verify_pass=False,
        agreement=0.99,
        tau_hi=0.95,
        tau_lo=0.70,
    ) == "escalate"


def test_low_agreement_escalates():
    assert route_decision(
        verify_pass=True,
        agreement=0.50,
        tau_hi=0.95,
        tau_lo=0.70,
    ) == "escalate"


def test_middle_zone_is_conservative():
    assert route_decision(
        verify_pass=True,
        agreement=0.80,
        tau_hi=0.95,
        tau_lo=0.70,
    ) == "escalate"


def test_threshold_boundary():
    assert route_decision(
        verify_pass=True,
        agreement=0.95,
        tau_hi=0.95,
        tau_lo=0.70,
    ) == "auto"
