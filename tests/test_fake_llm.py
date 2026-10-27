from research_assistant.llm.fake_provider import FakeLLM


def test_matches_by_marker_substring():
    llm = FakeLLM(responses={"ROLE: planner": "planner response"})
    assert llm.invoke("ROLE: planner\n\nsome prompt") == "planner response"


def test_falls_back_to_default_when_no_marker_matches():
    llm = FakeLLM(responses={"ROLE: planner": "x"}, default="fallback")
    assert llm.invoke("ROLE: unknown") == "fallback"


def test_scripted_list_advances_on_repeated_calls():
    llm = FakeLLM(responses={"ROLE: critic": ["first", "second"]})
    assert llm.invoke("ROLE: critic call 1") == "first"
    assert llm.invoke("ROLE: critic call 2") == "second"


def test_scripted_list_holds_on_last_value_after_exhausted():
    llm = FakeLLM(responses={"ROLE: critic": ["only"]})
    assert llm.invoke("ROLE: critic") == "only"
    assert llm.invoke("ROLE: critic") == "only"


def test_records_every_call():
    llm = FakeLLM(responses={})
    llm.invoke("prompt one")
    llm.invoke("prompt two")
    assert llm.calls == ["prompt one", "prompt two"]
