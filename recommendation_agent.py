import ollama


class RecommendationAgent:

    def __init__(self):
        pass

    def get_recommendation(self, disease):

        prompt = f"""
You are a dermatology recommendation assistant.

A patient has been diagnosed with: {disease}

Provide a SHORT recommendation in the following format only:

Disease:
<one line>

Symptoms:
- symptom 1
- symptom 2

Homeopathic Remedies:
- remedy 1
- remedy 2

Advice:
- advice 1
- advice 2

Maximum 80 words.
Do not give long explanations.
"""

        response = ollama.chat(
            model='llama3',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        return response['message']['content']


# Test
if __name__ == "__main__":

    disease = input("Enter disease name: ")

    agent = RecommendationAgent()

    print("\n===== RECOMMENDATION =====\n")

    recommendation = agent.get_recommendation(disease)

    print(recommendation)





