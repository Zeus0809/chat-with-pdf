import pymupdf as pd
import pprint


PATH = "/Users/illiakozlov/ChatWithPDF/chat-with-pdf/data/SOR-2001-269.pdf"

def get_text_dict():

    pdf = pd.open(PATH)
    text_dict = pdf[0].get_text("dict", sort=True)
    pprint.pprint(text_dict)

if __name__ == "__main__":
    get_text_dict()
