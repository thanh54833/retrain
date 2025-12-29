p_c:
	git pull &&git add . && git commit -m "Update Document" && git push

sync:
	@echo "Setting up cron job to run 'make p_c' every 5 minutes..."
	@(crontab -l 2>/dev/null | grep -v "make p_c"; echo "*/5 * * * * cd $(PWD) && make p_c >> $(PWD)/cron.log 2>&1") | crontab -
	@echo "Cron job installed! It will run every 5 minutes."
	@echo "Logs will be written to $(PWD)/cron.log"

schedule_30_s:
	@echo "Starting 30-second push scheduler..."
	@nohup sh -c 'while true; do cd $(PWD) && make p_c >> $(PWD)/schedule_30s.log 2>&1; sleep 30; done' > /dev/null 2>&1 & echo $$! > $(PWD)/.schedule_30s.pid
	@echo "30-second scheduler started! PID: $$(cat $(PWD)/.schedule_30s.pid)"
	@echo "Logs will be written to $(PWD)/schedule_30s.log"

unschedule_30_s:
	@if [ -f $(PWD)/.schedule_30s.pid ]; then \
		kill $$(cat $(PWD)/.schedule_30s.pid) 2>/dev/null || true; \
		rm -f $(PWD)/.schedule_30s.pid; \
		echo "30-second scheduler stopped!"; \
	else \
		echo "No scheduler running or PID file not found."; \
	fi

unschedule:
	@echo "Removing cron job for 'make p_c'..."
	@crontab -l 2>/dev/null | grep -v "make p_c" | crontab -
	@echo "Cron job removed!"


	
 
 