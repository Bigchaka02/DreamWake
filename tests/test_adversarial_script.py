"""Exercise the adversarial dreams command with no network calls."""
import json
import runpy
from pathlib import Path

from dreamwake import designs, llm


def test_adversarial_script_writes_four_cases_and_table(tmp_path, monkeypatch, capsys):
    def chat(messages, tools=None, **kwargs):
        assert tools is not None
        assert designs.DREAM_SUFFIX in messages[1]["content"]
        args = {"summary": "done"}
        return {"content": "", "tool_calls": [{"name": "finish", "arguments": args, "raw": json.dumps(args)}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "seconds": 0.0,
                "finish_reason": "tool_calls"}

    monkeypatch.setenv("DW_MODEL", "fake")
    monkeypatch.setattr(llm, "chat", chat)
    script = Path(__file__).resolve().parents[1] / "scripts" / "adversarial_dreams.py"
    runpy.run_path(str(script))["main"](["--trials", "1", "--out", str(tmp_path)])

    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    cases = ["good", "wrong_scope", "wrong_scope_size", "inverted"]
    assert [row["case"] for row in rows] == cases
    table = capsys.readouterr().out.splitlines()
    assert "| Case | Trials | Control verified | Rules changed | First practice scored 0 |" in table
    for row in rows:
        assert set(row) == {"case", "trial", "record", "items_after", "changed"}
        assert row["trial"] == 0
        record = row["record"]
        [offered] = record["dreams_offered"]
        assert offered["rejected"] == "" and offered["control_score"] is not None
        assert row["items_after"] == record["items_before"] and row["changed"] is False
        verified = int(offered["control_score"] == 0)
        first = "-"
        if row["case"] == "good":
            first = int(bool(record["exercises"]) and record["exercises"][0]["score"] == 0)
        assert f"| {row['case']} | 1 | {verified} | 0 | {first} |" in table
    # A control that only finishes adds nothing: regeneration must not hand it the "_MS" check string.
    scores = {row["case"]: row["record"]["dreams_offered"][0]["control_score"] for row in rows}
    assert scores["good"] > 0 and scores["wrong_scope"] > 0 and scores["wrong_scope_size"] > 0
    assert "TIMEOUT_MS" in (tmp_path / "trial0" / "good" / "s2" / "src" / "settings.py").read_text(encoding="utf-8")
