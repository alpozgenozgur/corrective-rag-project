import sys

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from graph.graph import app


if __name__ == "__main__":
    print(app.invoke(input={'question':'How can i make a hamburger'}))