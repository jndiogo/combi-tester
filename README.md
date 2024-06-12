# Combi-tester

This is a simple way to test an LLM from combinations of prompts and expected results.

Requires [Sibila](https://github.com/jndiogo/sibila) to access local and remote LLMs. Install Sibila with:

```
pip install --upgrade sibila
```

Each test is driven by an YAML config file, like the following example of an LLM for NLP processing of user commands, controlling a system of four elements (A,B,C,D), with each having levels between 0 and 9. See the inlined comments for more info:


``` YAML
setup:
  # optional instruction/system message:
  inst: |-
    You will parse user commands and emit actions to control the state of a system, which is listed after "STATE:"
    The system has 4 elements named A, B, C, D, which can be set to a level between 0 and 9, where level 0 means off, while level 9 means maximum or full.
    If the user requests a non-existent element (not one of A, B, C, D), emit a special action setting any element to special level -1.
    If the user requests a level which is not between 0 and 9, emit a special action setting any element to special level -1.
    For example: 
    - if the user enters "set element A to level 7", you should emit an action with element=A, level=7
    - if the user enters "set element H to level 4", you should emit an action with element=A, level=-1, because H is not a valid element
    - if the user enters "set element B to level 16", you should emit an action setting element=A, level=-2, because level 16 is outside 0-9 range
    
  # the script responsible for doing the inference (whatever you put in the generate() function) and scoring/evaluation (evaluate()):
  script: |
    from typing import Any, Optional, Union, Literal
    from pydantic import BaseModel, Field
    
    Element = Literal["A","B","C","D"]
    
    class Action(BaseModel):
        #thinking: str = Field(description="Reasoning behind the action")
        element: Element = Field(description="Element to set")
        level: int = Field(description="Level to set, between 0 and 9, or special level -1")
    

    def generate(model, inst_text, in_text):
        return model.extract(list[Action], in_text, inst=inst_text)

    def evaluate(value: Action, expected: dict):
        sub_scores = []

        for field in expected:
             score = getattr(value,field) == expected[field]
             sub_scores.append(float(score))

        return sum(sub_scores) / len(sub_scores)
          
        
  vars: # vars defined here will be replaced into the "in" text for each test run:
    off_state: {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    valid_element*: ["A","B","C","D"] # vars ending with * will have their values combined to form each individual prompt
    invalid_element*: ["E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","X","Y","W","Z"]
    valid_level*: [0,1,2,3,4,5,6,7,8,9]
    invalid_level*: [10,11,12,13,14,15,24,36,41,59,67,73,84,99,563,999,-1,-5,-20]

tests: # the actual test input (aka user prompt):
  - in: |-
      STATE: {off_state}
      Set element {valid_element*} to level {valid_level*}
    expected: # the expected generated values, each key is used in the evaluate() function:
      element: "valid_element*"
      level: "valid_level*"

  - in: |-
      STATE: {off_state}
      Set {invalid_element*} to {valid_level*}
    expected:
      level: "-1" # direct value to compare with LLM result

  - in: |-
      STATE: {off_state}
      Set {valid_element*} to {invalid_level*}
    expected:
      level: "-1"

```

To test local GGUF models (llama.cpp based) place the files in a "models" folder. Or use any remote models supported by Sibila.

Run the test from the YAML config:


``` python
try: from dotenv import load_dotenv; load_dotenv()
except: ...

from tester import TestSet

from sibila import Models


Models.setup("models", clear=True)

models = [
    "llamacpp:Phi-3-mini-4k-instruct-q4.gguf",
    "llamacpp:Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    "llamacpp:openchat-3.6-8b-20240522-Q4_K_M.gguf",
    "llamacpp:Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
    "openai:gpt-3.5-turbo-0125",
    "openai:gpt-4o-2024-05-13",
]

test_path = "abcd4-mult.yaml"
tester = TestSet(test_path)

res = tester.run_tests_for_models(models, 
                                  options={"only_tests": [], # if set: the indices of the tests to run
                                           "first_n_runs": None, # if set: only do first n combinations
                                           "model_delays": {"anthropic": 5} # if model name in keys, delay n seconds before next call (for simple rate limiting) 
                                          })

with open("last_result.txt", "w", encoding="utf-8") as f:
  f.write(str(res))

print(TestSet.report(res))
```

Giving these (resumed) results:

```
Mean score for all models: 0.918
Model scores:
  llamacpp:Phi-3-mini-4k-instruct-q4.gguf: 0.877
  llamacpp:Meta-Llama-3-8B-Instruct-Q4_K_M.gguf: 0.930
  llamacpp:openchat-3.6-8b-20240522-Q4_K_M.gguf: 0.926
  llamacpp:Mistral-7B-Instruct-v0.3.Q4_K_M.gguf: 0.837
  openai:gpt-3.5-turbo-0125: 0.939
  openai:gpt-4o-2024-05-13: 0.998
Total 14 tests, 454 runs.

Incorrect answers per model:
= llamacpp:Phi-3-mini-4k-instruct-q4.gguf: 0.877 ===================
{'in_text': "STATE: {'A': 0, 'B': 0, 'C': 0, 'D': 0}\nSet E to 2", 'result': [Action(element='A', level=2)], 'expected': [{'level': -1}], 'score': 0.0}
{'in_text': "STATE: {'A': 0, 'B': 0, 'C': 0, 'D': 0}\nSet E to 3", 'result': [Action(element='A', level=3)], 'expected': [{'level': -1}], 'score': 0.0}
{'in_text': "STATE: {'A': 0, 'B': 0, 'C': 0, 'D': 0}\nSet E to 4", 'result': [Action(element='A', level=4)], 'expected': [{'level': -1}], 'score': 0.0}
{'in_text': "STATE: {'A': 0, 'B': 0, 'C': 0, 'D': 0}\nSet E to 5", 'result': [Action(element='A', level=5)], 'expected': [{'level': -1}], 'score': 0.0}
{'in_text': "STATE: {'A': 0, 'B': 0, 'C': 0, 'D': 0}\nSet E to 6", 'result': [Action(element='A', level=6)], 'expected': [{'level': -1}], 'score': 0.0}
(...)
```

## To do

- Load variables from files.
- Threads as input.
- An HTML results viewer.
- Results caching and persistence.
- Document compare functions.

