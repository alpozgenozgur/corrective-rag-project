from dotenv import load_dotenv

load_dotenv()

from graph.graph import app


if __name__ == "__main__":
    print(app.invoke(input={'question':'How can i make a hamburger'}))