.PHONY: smoke

smoke:
	@PYTHONPATH=. python -m tests.smoke_runner tests/smoke_scenario.txt --out smoke.log
