from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage


MAX_PAGES = 1500


class Preprocessor:
    def __init__(self, path):
        self.path = path
        self.pages = None
        self.start = 1
        self.end = MAX_PAGES
        self.paragraphs = None

    def __read_pdf(self):
        try:
            file = open(self.path, 'rb')
        except OSError:
            raise Exception("Can't open file: ", self.path)
        self.pages = PDFPage.get_pages(file, caching=True, check_extractable=True)

    def read_text(self):
        try:
            file = open(self.path, 'r')
        except OSError:
            raise Exception("Can't open file: ", self.path)
        text = file.read()
        self.paragraphs = text.split('\n\n')

    def set_start_page(self, start):
        self.start = start

    def set_end_page(self, end):
        self.end = end

    def page_by_page(self):
        self.__read_pdf()

        counter = 0
        for Page in self.pages:
            # check the page limits
            counter += 1
            if counter < self.start or counter > self.end:
                continue
            # if counter > self.end:
            #     return None

            # create the needed objects to read the page
            resource_manager = PDFResourceManager()
            file_handler = StringIO()
            converter = TextConverter(resource_manager, file_handler)
            page_interpreter = PDFPageInterpreter(resource_manager, converter)

            # read the page and convert it to text
            page_interpreter.process_page(Page)
            text = file_handler.getvalue()

            # return the text and start from here in the next call
            yield text

            # close the opened handlers
            converter.close()
            file_handler.close()

    def get_page(self, page_number):
        self.set_start_page(page_number)
        self.set_end_page(page_number + 1)

        for Page in self.page_by_page():
            return Page


# if __name__ == '__main__':
#     preprocessor = Preprocessor()

    # preprocessor.set_start_page(9)
    # preprocessor.set_end_page(81)
    # for page in preprocessor.page_by_page('inputs/modeling.pdf'):
    #     print(page)
    #     print()
    # print("Finished")

    # page = preprocessor.get_page('inputs/modeling.pdf', 23)
    # print(clean_text(page))

#     preprocessor.read_text('inputs/coreference.txt')
#     print("Coreference____________________________________________\n")
#     for paragraph in preprocessor.paragraphs:
#         print("Original: \n", paragraph)
#         print()
#         if need_segmentation(paragraph):
#             paragraph = word_segmentation(paragraph)
#             print("After Segmentation: \n", paragraph)
#             print()
#         paragraph = solve_coreference(paragraph)
#         print("After Coreference: \n", paragraph)
#         print()

#     preprocessor.read_text('inputs/wordsegment.txt')
#     print("Word Segmentation____________________________________________\n")
#     print("Original: \n", preprocessor.paragraphs[0])
#     print()
#     print("After Segmentation: \n", word_segmentation(preprocessor.paragraphs[0]))
#     print()