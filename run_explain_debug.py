import traceback
from datetime import datetime

try:
    import src.explain as ex
    ex.main()
except Exception as e:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"error_log_{timestamp}.txt"
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write(f"Error at {timestamp}\n\n")
        f.write(traceback.format_exc())
    print(f"Error logged to {log_filename}: {e}")
