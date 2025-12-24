.PHONY: smoke

smoke:
	@SMOKE=1 PYTHONPATH=. python -m tests.smoke_runner tests/smoke_scenario.txt
