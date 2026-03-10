import traceback
try:
    import src.explain as ex
    ex.main()
except Exception as e:
    with open('error_log.txt', 'w', encoding='utf-8') as f:
        f.write(traceback.format_exc())
