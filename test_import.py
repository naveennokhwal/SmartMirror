# test_import.py
try:
    from api.emotion import EmotionDetector
    print("Import successful!")
    detector = EmotionDetector()
    print("Initialization successful!")
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
