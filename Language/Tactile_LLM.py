import requests
import json
import numpy as np
import re

def structure_input_LLM(centroids, phrases, HA):
    if HA == None:
        HA = np.random.randint(45, 95, size=len(phrases))   
        if np.random.randint(4)==0:
            HA[0] = 0

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
            # print(dic['object'], dic['x'], dic['y'], dic['hardness'])

            objects_structured.append(dic)
        
    return objects_structured

def get_LLM_response(objects_structured):
    API_KEY = "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" #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  #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
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    You are a descriptive scene generator for a robot vision system that identifies fruits and vegetables. You will receive a list dictionaries corresponding to items, 
    each with positions (x, y):
    - x represents front/back position (negative = further back, positive = closer to front)
    - y represents left/right position (negative = left, positive = right)

    You will also receive a hardness measurement (HA: hardness units) for each item. Your task is to describe locations, hardness values (and in some cases interpret ripeness) concisely and naturally.

    Here are 10 rules to follow:
    1. Do not state the x,y positions. Use relative positions for objects close to one another (example: if 2 bananas are close to one another in the back of the scene, say: on the back, the most to the right banana has
    a hardness value of XX HA. Just slightly located to the left of it, the second banana has a hardness level of XX HA.). To get object locations out of x,y positions, follow next rules:
    
    x determinses the front or back side of the scene. The three cases are:
    - if x < -170: object is positioned in the back.
    - if x is between -170 and -20: object is positioned in the center.
    - if x > -20: object is positioned in the front.

    y determines the left or right side of the scene. The three cases are as follows:
    - if y < -520: object is positioned on the left.
    - if y is between -520 and -400: object is positioned in the center.
    - if y > -400: object is positioned on the right side of the scene.

    Make sure to check this well.

    2. State the hardness value for a fruit. Do also say for banana, lime and lemon if fruit is too soft or hard by looking at the ideal ripeness ranges:
    - Bananas: ripe if hardness is between 60 and 75 HA.
    - Lime, Lemon: ripe between 65 and 80 HA.

    To interprete the ripeness stage of the fruits in point 2, use following rules:
    - Within ideal range: Don’t state firmness.
    - Above range: Say “firmer than ideal” or “unripe”.
    - Below range: Say “softer than ideal” or “overripe”.
    - If values are close to one another, don't be over interpretative in ripeness.

    3. Begin always answer with an overview of the items that could be found.

    4. You must compare the same types of fruits or vegetables in terms of ready to eat or ripeness, example given by saying if there are two bananas: the left one is the most ready to eat or the most frontis the ripest probably. If both are
    far outside ideal range, do state this that it is likely that no ripe item is present of that fruit. If both are within range, state that both are ripe. Focus on comparisons and not too much on ripeness individually,
    and state if something is really too soft for its range (overripe) or too hard (underripe).

    5. Do not compare different types of fruits or vegetables in terms of ready to eat or ripeness, so skip that. 
    
    6. If you only have 1 fruit of a sort, just say hardness (and if soft or hard). Do never state that you do not have the ideal range.
    
    7. Keep in mind location is just as support to talk about object ripeness and relative positions of similar positioned objects. It is just to refer to the user
    which item you are talking about. If x values are comparable, use y position to make distinction and vice versa. If x and y are center, just say center.
    
    8. If a fruit or vegetable is present with a 0 hardness value (HA), just say to the user at end of answer that that item could not be found in the scene. Do never mention any location or hardness value of if hardness is zero.
    
    9. Use fluent text, human-interpretable and interesting. Not over interpretative. Report in one paragraph. Be very concise. No notes and repetitive words or sayings. Use short sentences.

    10. Do not output your reasoning steps or any text inside <think> tags.

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
    print("Requests per day limit:", response.headers.get('x-ratelimit-limit-requests'))
    print("Requests remaining today:", response.headers.get('x-ratelimit-remaining-requests'))
    print("Tokens per minute limit:", response.headers.get('x-ratelimit-limit-tokens'))
    print("Tokens remaining this minute:", response.headers.get('x-ratelimit-remaining-tokens'))
    print("Time to wait before retrying (seconds):", response.headers.get('retry-after'))
    response_text = response.json()["choices"][0]["message"]["content"]
    clean_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    return clean_text


# test above logic
# centroids = [(-170, -550), (-200, -450), (0, -430), (-200, -430)]
# phrases = ["strawberry", "corn", "banana", "banana"]
# # centroids = [(-170, -550), (-50, -450), (-200, -441)]
# # phrases = ["corn", "banana", 'banana']
# output = structure_input_LLM(centroids, phrases, None)
# print(output)

# response = get_LLM_response(output)
# print(response)

# # evaluation = evaluate_LLM_output(output, response)
# # print(evaluation)
