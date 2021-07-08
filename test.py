from pprint import pprint
#import nltk
#nltk.download('stopwords')
from Questgen import main
from Questgen.mcq.MCQGen import MCQGen

payload = {
            "input_text": "Sachin Ramesh Tendulkar is a former international cricketer from India and a former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket. He is the highest run scorer of all time in International cricket. I wanna be a cricketer",
            "max_questions": 3
          }

def testTF():
    qe= main.BoolQGen()
    output = qe.predict_boolq(payload)
    pprint(output)

def testMCQ():
    mcqGen = MCQGen()
    output = mcqGen.predict_mcq(payload)
    pprint(output)

def testFAQ():
    qg = main.QGen()
    output = qg.predict_shortq(payload)
    pprint(output)

def testParaphrasing():
    qg = main.QGen()
    output = qg.paraphrase(payload)
    pprint(output)

""" print("\nTrue/False::")
testTF() """
print("\nMCQ::")
testMCQ()
""" print("\nFAQ::")
testFAQ()
print("\nParaphrasing::")
testParaphrasing()
 """

""" answer = main.AnswerPredictor()
payload3 = {
    "input_text" : '''Sachin Ramesh Tendulkar is a former international cricketer from 
              India and a former captain of the Indian national team. He is widely regarded 
              as one of the greatest batsmen in the history of cricket. He is the highest
               run scorer of all time in International cricket.''',
    "input_question" : "What is Sachin Tendulkar profession?"
    
}
output = answer.predict_answer(payload3)
print(output)
 """

""" payload4 = {
    "input_text" : '''Sachin Ramesh Tendulkar is a former international cricketer from 
              India and a former captain of the Indian national team. He is widely regarded 
              as one of the greatest batsmen in the history of cricket. He is the highest
               run scorer of all time in International cricket.''',
    "input_question" : "Is Sachin tendulkar a former cricketer? "
}
output = answer.predict_answer(payload4)
print (output) """