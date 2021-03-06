import torch
from QAGen.utilities import tokenize_sentences, get_keywords, get_sentences_for_keyword
from QAGen.QGen import QGen


class ShortGen(QGen):

    def __init__(self, loader):
        QGen.__init__(self, loader)
            

    def predict_shortq(self, keywords, modified_text):
        
        sentences = tokenize_sentences(modified_text)
        
        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet

        #final_output = {}

        if len(keyword_sentence_mapping.keys()) == 0:
            print('ZERO')
            return []
        else:
            
            generated_questions = self.__generate_normal_questions(keyword_sentence_mapping,self.device,self.tokenizer,self.qg_model)
            
            
        #final_output["statement"] = modified_text
        #final_output["questions"] = generated_questions["questions"]
        
        if torch.device=='cuda':
            torch.cuda.empty_cache()

        #return final_output
        return generated_questions


    def __generate_normal_questions(self,keyword_sent_mapping,device,tokenizer,model):  #for normal one word questions
        batch_text = []
        answers = keyword_sent_mapping.keys()
        for answer in answers:
            txt = keyword_sent_mapping[answer]
            context = "context: " + txt
            text = context + " " + "answer: " + answer + " </s>"
            batch_text.append(text)
        
        encoding = tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True, return_tensors="pt", max_length=512, truncation=True)
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        with torch.no_grad():
            outs = model.generate(input_ids=input_ids,
                                attention_mask=attention_masks,
                                max_length=150)
            
        output_array = []
        #output_array["questions"] =[]
        
        for index, val in enumerate(answers):
            #individual_quest= {}
            out = outs[index, :]
            dec = tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            Question= dec.replace('question:', '')
            Question= Question.strip()

            #individual_quest['Question']= Question
            #individual_quest['Answer']= val
            #individual_quest["id"] = index+1
            #individual_quest["context"] = keyword_sent_mapping[val]
            
            #output_array["questions"].append(individual_quest)
            output_array.append((Question, val))
            
        return output_array