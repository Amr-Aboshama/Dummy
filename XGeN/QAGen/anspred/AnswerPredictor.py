#nltk.download('brown')
#nltk.download('stopwords')
#nltk.download('popular')


class AnswerPredictor:
          
    def __init__(self, loader):
        self.tokenizer = loader.tokenizer
        self.device = loader.device
        self.model = loader.ap_model

    def greedy_decoding (inp_ids,attn_mask,model,tokenizer):
        greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
        Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        return Question.strip().capitalize()

    def predict_answer(self,payload):
        inp = {
            "input_text": payload.get("input_text"),
            "input_question" : payload.get("input_question")
        }

        context = inp["input_text"]
        question = inp["input_question"]
        input = "question: %s <s> context: %s </s>" % (question,context)

        encoding = self.tokenizer.encode_plus(input, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)
        greedy_output = self.model.generate(input_ids=input_ids, attention_mask=attention_masks, max_length=256)
        Question =  self.tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
        output = Question.strip().capitalize()

        return output
