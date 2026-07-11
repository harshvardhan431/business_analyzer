from dotenv import load_dotenv
load_dotenv()
import os
print(load_dotenv())  # True = file found
print(repr(os.getenv("GEMINI_API_KEY")))