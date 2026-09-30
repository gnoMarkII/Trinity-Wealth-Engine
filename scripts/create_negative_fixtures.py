import pathlib
import json

p = pathlib.Path("tests/fixtures/terminal_v2/phase4")
p.mkdir(parents=True, exist_ok=True)

(p / "cboe_corrupted_fixture.csv").write_text("INVALID_HEADER,FOO\nnot_a_date,not_a_num\n", encoding="utf-8")
(p / "sec_companyfacts_empty_fixture.json").write_text(json.dumps({"cik": 999999, "entityName": "Empty Corp", "facts": {"us-gaap": {}}}, indent=2), encoding="utf-8")
(p / "sec_form4_corrupted_fixture.xml").write_text("<?xml version=\"1.0\"?><malformedDocument>", encoding="utf-8")
(p / "treasury_auctions_empty_fixture.json").write_text(json.dumps({"data": []}, indent=2), encoding="utf-8")
(p / "google_news_empty_fixture.xml").write_text("<?xml version=\"1.0\"?><rss version=\"2.0\"><channel><title>Empty</title></channel></rss>", encoding="utf-8")

print("Negative fixtures created successfully in", p)
