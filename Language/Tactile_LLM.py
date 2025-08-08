import requests
import json
import numpy as np

def structure_input_LLM(centroids, phrases, HA):
    if HA == None:
        HA = np.random.randint(45, 90, size=len(phrases))   

    F = []
    objects_structured = []

    for i,phrase in enumerate(phrases):
        dic = {}
        F.append(phrase)
        count = F.count(phrase)

        dic["object"] = f"{phrase} {count}"
        dic['x'] = centroids[i][0]
        dic['y'] = centroids[i][1]
        dic['hardness'] = int(HA[i])

        objects_structured.append(dic)
    
    return objects_structured

def get_LLM_response(objects_structured):
    API_KEY = "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"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
    You are a descriptive scene generator for a robot vision system, specialized in fruits and vegetables. You will receive a list of fruits and vegetables
    with positions (x, y) and hardness values. The goal is to communicate this information concisely and effectively to the user.

    Some rules to follow:
    1. Describe the object locations like front/center/right/left/back/... Do not state the x,y positions explicitly and use relative positions for similar 
    objects close to one another (example: if 2 bananas are close to one another in the back of the scene, say: on the back, the most to the right banana has
    a hardness value of XX HA. Just slightly located to the left of it, the second banana has a hardness level of XX HA.)
    2. To get object locations out of x,y positions, follow next rules:
    if x <-170:
        fb = 'back'
    elif -170 <= x <= -20:
        fb = 'center'
    elif x > 20:
        fb = 'front'

    if y <-520:
        lr = 'left'
    elif -520 <= y <= -400:
        lr = 'center'
    elif y>-400:
        lr = 'right'
    
    3. Use natural, human-readable text and interprete the hardness values. State if the hardness value for an item is what you would expect if it is ripe, unripe
    or overripe. Remind that the range for the ideal ripeness is different for different types of fruits and vegetables. Example:
    - Apples: ripe if hardness is between 30 and 45 HA.
    - Bananas: ripe if hardness is between 65 and 70 HA.
    - Avocados: ripe between 25 and 40 HA.
    - Mango: ripe between 25 and 40 HA.
    - Lemon: ripe between 35-50.
    - Tomato: ripe between 70-80 HA.
    4. All hardness levels are in shore A values. HA = Hardness Unit.
    5. You can compare the same fruits or vegetables, example given by saying if there are two bananas: the left one is the most ready to eat.
    6. Do not compare different fruits or vegetables in terms of ready to eat or ripeness, so skip that.
    7. Keep in mind location is just as support to talk about object ripeness and relative positions of similar positioned objects. It is just to refer to the user
    which iterm you are talking about.
    8. Firmer means above ideal HA values, softer is lower HA value.
    9. Overripe is typically also not what we want to eat. For hardness above the ripe range, describe as "likely unripe" or "firm." Below say "overripe" or "too soft", not unripe.
    If too low, do not say "not ready to eat" as it is actually ready but just a bit too long waiting already so not tasty.
    10. Report in one paragraph. Be concise.


    Input:
    {json.dumps(objects_structured, indent=2)}

    Now write the description:
    """

    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You turn object data into scene descriptions, explain and interprete tactile levels."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

    return response.json()["choices"][0]["message"]["content"]


### test above logic
centroids = [(-170, -550), (-200, -450), (-50, -450), (-50, -300)]
phrases = ["banana", "lemon", "lime", "corn"]
output = structure_input_LLM(centroids, phrases, None)

response = get_LLM_response(output)
print(response)