import re
import time
import spacy
from nltk import FreqDist
from nltk.corpus import brown
from sense2vec import Sense2Vec
from similarity.normalized_levenshtein import NormalizedLevenshtein
from Questgen.mcq.mcq import tokenize_sentences, get_keywords, get_sentences_for_keyword, \
                             filter_phrases, sense2vec_get_words



class MCQGen:
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.s2v = Sense2Vec().from_disk('s2v_old')
        self.fdist = FreqDist(brown.words())
        self.normalized_levenshtein = NormalizedLevenshtein()

        
    def predict_mcq(self, payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions", 4)
        }

        text = inp['input_text']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)
        
        keywords = get_keywords(self.nlp,modified_text,inp['max_questions'],self.s2v,self.fdist,self.normalized_levenshtein,len(sentences) )

        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
        
        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet
   
        final_output = {}

        if len(keyword_sentence_mapping.keys()) == 0:
            return final_output
        else:
            try:
                generated_questions = self.generate_questions_mcq(keyword_sentence_mapping,self.s2v,self.normalized_levenshtein)

            except:
                return final_output
            end = time.time()

            final_output["statement"] = modified_text
            final_output["questions"] = generated_questions["questions"]
            final_output["time_taken"] = end-start
            
            return final_output


    def generate_questions_mcq(self, keyword_sent_mapping,sense2vec,normalized_levenshtein):
        answers = keyword_sent_mapping.keys()
        output_array = {}
        output_array["questions"] = []

        for index, val in enumerate(answers):
            individual_question ={}
            # The next line to replace with case insensitive
            context = re.sub(re.escape(val), "____", keyword_sent_mapping[val], flags=re.IGNORECASE)
            individual_question["context"] = context
            individual_question["question_type"] = "MCQ"
            individual_question["answer"] = val
            individual_question["id"] = index+1
            individual_question["options"], individual_question["options_algorithm"] = self.get_options(val, sense2vec)
            individual_question["options"] =  filter_phrases(individual_question["options"], 10,normalized_levenshtein)
            index = 3
            individual_question["extra_options"]= individual_question["options"][index:]
            individual_question["options"] = individual_question["options"][:index]
            
            if len(individual_question["options"])>0:
                output_array["questions"].append(individual_question)

        return output_array

    def get_options(self, answer,s2v):
        distractors =[]

        try:
            distractors = sense2vec_get_words(answer,s2v)
            if len(distractors) > 0:
                print(" Sense2vec_distractors successful for word : ", answer)
                return distractors,"sense2vec"
        except:
            print (" Sense2vec_distractors failed for word : ",answer)


        return distractors,"None"