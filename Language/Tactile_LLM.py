import requests
import json
import numpy as np
import re

def structure_input_LLM(centroids, phrases, HA):
    if HA == None:
        HA = np.random.randint(45, 90, size=len(phrases))   

    F = []
    objects_structured = []

    for i,phrase in enumerate(phrases):
        if i <= len(centroids)-1:
            dic = {}
            F.append(phrase)
            count = F.count(phrase)


            dic["object"] = f"{phrase} {count}"
            dic['x'] = centroids[i][0]
            dic['y'] = centroids[i][1]
            dic['hardness'] = int(HA[i])
            print(dic['object'], dic['x'], dic['y'], dic['hardness'])

            objects_structured.append(dic)
        
    return objects_structured

def get_LLM_response(objects_structured):
    API_KEY = "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"   #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
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    You are a descriptive scene generator for a robot vision system that identifies fruits and vegetables. You will receive a list dictionaries corresponding to items, 
    each with positions (x, y):
    - x represents front/back position (negative = farther back, positive = closer to front)
    - y represents left/right position (negative = left, positive = right)

    You will also receive a hardness measurement (HA: hardness units) for each item. Your task is to describe locations, hardness values and ripeness concisely and naturally.

    Here are 10 rules to follow:
    1. Do not state the x,y positions. Use relative positions for objects close to one another (example: if 2 bananas are close to one another in the back of the scene, say: on the back, the most to the right banana has
    a hardness value of XX HA. Just slightly located to the left of it, the second banana has a hardness level of XX HA.). To get object locations out of x,y positions, follow next rules:
    
    For x, the first number in th
    if x <-170:
        fb = 'back'
    elif -170 <= x <= -20:
        fb = 'center'
    elif x > -20:
        fb = 'front'

    if y <-520:
        lr = 'left'
    elif -520 <= y <= -400:
        lr = 'center'
    elif y>-400:
        lr = 'right'

    So in the scene x is down-under location, y is left-right location. Make sure to check this properly. 
    
    2. State the hardness value for a fruit. Do say if some fruit is too soft or hard by looking at the ideal ripeness ranges, but don't take too much conclusion on individual ripenesss state. Remind that the range for the ideal ripeness is different for different types of fruits and vegetables
    Do not be too strict on these ranges, example:
    - Bananas: ripe if hardness is between 60 and 75 HA.
    - Zucchini: ripe between 75 and 95 HA.
    - Carrot, Patato: ripe between 85 and 100 HA.
    - Lime, Lemon: ripe between 65 and 80 HA.
    - Kiwi, Strawberry: ripe between 40 and 60 HA.
    - Pepper: ripe between 55 and 70 HA.
    - Tomato: ripe between 55 and 70 HA

    3. To interprete, use following rules (related to point 2):
    - Within ideal range: Don’t state firmness.
    - Above range: Say “firmer than ideal” or “unripe”.
    - Below range: Say “softer than ideal” or “overripe”.
    - If values are close to one another, don't be over interpretative in ripeness.
    4. You must compare the same fruits or vegetables in terms of ready to eat or ripeness, example given by saying if there are two bananas: the left one is the most ready to eat or the most frontis the ripest probably. If both are
    far outside ideal range, do state this that it is likely that no ripe item is present of that fruit. If both are within range, state that both are ripe. Focus on comparisons and not too much on ripeness individually,
    and state is something is really too soft for its range (overripe) or too hard (underripe).
    5. Do not compare different fruits or vegetables in terms of ready to eat or ripeness, so skip that. If you only have 1 fruit of a sort, just say hardness and if soft or hard, but not interpretation on ripeness. Don't state this info.
    6. Keep in mind location is just as support to talk about object ripeness and relative positions of similar positioned objects. It is just to refer to the user
    which iterm you are talking about. If x values are comparable, use y position to make distinction and vice versa.
    7. Firmer means above (>) ideal HA values, softer is lower (<) HA value in comparison. Check this properly.
    8. Report in one paragraph. Be very concise. No notes and repetitive words or sayings. Use short sentences.
    9. If a fruit or vegetable is present with a 0 hardness value, just say to the user that that item could not be found in the scene. Do not mention any location or hardness value of this item.
    10. Use fluent text, human-interpretable and interesting. Not over interpretative.Do not output your reasoning steps or any text inside <think> tags.



    Input:
    {json.dumps(objects_structured, indent=2)}

    Now write the description:
    """

    data = {
        "model": "deepseek-r1-distill-llama-70b",
        "messages": [
            {"role": "system", "content": "You turn object data into scene descriptions, explain and interprete tactile levels."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

    response_text = response.json()["choices"][0]["message"]["content"]
    clean_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    return clean_text


## test above logic
# centroids = [(-170, -550), (-200, -450), (-50, -450), (-50, -300)]
# phrases = ["lime", "lemon", "lime", "lime"]
# output = structure_input_LLM(centroids, phrases, None)

# response = get_LLM_response(output)
# print(response)