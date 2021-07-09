from pprint import pprint
#import nltk
#nltk.download('stopwords')
from Questgen import main
from Questgen.mcq.MCQGen import MCQGen
from Questgen.boolean.BooleanGen import BoolGen

payload = {
            "input_text": "Sachin Ramesh Tendulkar is a former international cricketer from India and a former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket. He is the highest run scorer of all time in International cricket.",
            "max_questions": 3,
            "topics_num": 3
          }

def testTF():
    boolGen = BoolGen()
    output = boolGen.predict_tf(payload)
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
    return output

print("\nTrue/False::")
testTF()
print("\nMCQ::")
testMCQ()
print("\nFAQ::")
testFAQ()


print("\nParaphrasing::")
out = testParaphrasing()

answer = main.AnswerPredictor()
payload3 = {
    "input_text" : out['Paragraph'],
    "input_question" : out['Paraphrased Questions'][0]
}
output = answer.predict_answer(payload3)
print(output)


""" payload4 = {
    "input_text" : '''Sachin Ramesh Tendulkar is a former international cricketer from 
              India and a former captain of the Indian national team. He is widely regarded 
              as one of the greatest batsmen in the history of cricket. He is the highest
               run scorer of all time in International cricket.''',
    "input_question" : "Is Sachin tendulkar a former cricketer? "
}
output = answer.predict_answer(payload4)
print (output) """