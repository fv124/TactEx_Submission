import spacy
from textblob import Word

def get_fruits_of_interest(user_input):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(user_input)
    words = [token.lemma_.lower() for token in doc if not token.is_punct]

    keywords = ['fruit', 'fruits']
    fruits = ['banana', 'apple', 'pear', 'peach', 'lemon', 'lime', 'tomato', 'patato', 'avocado', 'pepper', 'carrot', 'corn', 'zucchini', 'kiwi', 'strawberry', 'garlic', 'potato']
    fruits_of_interest = []
    pluralize = False

    key2 = None
    for i,token in enumerate(doc):
        if token.tag_ == "JJS":
            #if doc[i+1].lemma_.lower() in fruits:
            key2 = "JJS"

    for fruit in fruits:
        if fruit in words:
            fruits_of_interest.append(fruit)
            if Word(fruit).pluralize().lower() in user_input.lower():
                pluralize = True
    key = False
    for keyword in keywords:
        if keyword in words:
            key = True

    if len(fruits_of_interest)==0 and key==True:
        intent = "compare_all"
        fruits_of_interest = fruits
    elif len(fruits_of_interest)==0:
        intent = "N/A"
    elif pluralize==True or key2 == "JJS":
        intent = "multiple_2"
    elif len(fruits_of_interest)>1:
        intent = "multiple_1"
    else:
        intent = "single"
    
    return fruits_of_interest, intent


def get_property(user_input):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(user_input)
    words = [token.text for token in doc]

    prop = None
    for word in words:
        if "ripe" in word:
            prop = "ripeness"
            break
        elif "hard" in word:
            prop = "hardness"
            break
        elif "soft" in word:
            prop = "softness"
            break
    
    return prop

def structure_fruits(obj_list):
    if len(obj_list) == 1:
        return obj_list[0]
    elif len(obj_list) == 2:
        return f"{obj_list[0]} and {obj_list[1]}"
    else:
        return ", ".join(obj_list[:-1]) + f", and {obj_list[-1]}"

def chat_response(user_input):
    prop = get_property(user_input)
    fruits_of_interest, intent = get_fruits_of_interest(user_input)

    if prop==None or intent=='N/A':
        return ("Sorry, I didn’t understand your request. Please try asking more structured sentences like 'Identify if the banana is ripe' "
                "or 'Compare hardness of apple and orange. Remember I am only able to analyze fruits at the moment."), fruits_of_interest
    elif intent == 'multiple_1':
        answer = structure_fruits(fruits_of_interest)
        return (f"Okay, I'll check the {prop} of the {answer}."), fruits_of_interest
    elif intent == 'multiple_2':
        plural_obj = []
        for fruit in fruits_of_interest:
            plural_obj.append(Word(fruit).pluralize())
        answer = structure_fruits(plural_obj)
        return (f"I will compare the {prop} of the different {answer}."), fruits_of_interest
    elif intent == 'compare_all':
        return (f"I will check the {prop} of all the fruits i can identify. I will come back to you with a report."), fruits_of_interest
    else:
        return (f"Got it, I will check the {prop} level of the {fruits_of_interest[0]} on the table."), fruits_of_interest
    
def get_intent(user_input):
    prop = get_property(user_input)
    fruits_of_interest, intent = get_fruits_of_interest(user_input)
    sentence = ''

    for i, fruit in enumerate(fruits_of_interest):
        sentence += fruit + '.'
        if i != len(fruits_of_interest)-1:
            sentence += ' '

    return prop, sentence, intent
    
# Test the NLP logic by running this script
# print(chat_response("How hard is the banana?"))
# print(chat_response("I would like to know the ripeness level of the apples"))
# print(chat_response("Which of the bananas is harder?"))
# print(chat_response("I would like to know the ripeness level of the avocado and apple"))
# print(chat_response("What is the ripest fruit on the table"))
# print(chat_response("What is the ripeness of the avocados vs the apples"))
# print(chat_response("What is the ripest: oranges or apples"))
# print(chat_response("What is the ripest avocado?"))


