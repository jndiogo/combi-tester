try: from dotenv import load_dotenv; load_dotenv()
except: ...

from sibila import Models

from tester import TestSet


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

res = tester.run_tests_for_models(models[-1:], 
                                  options={"only_tests": [1], # if set: the indices of the tests to run
                                           "first_n_runs": 2, # if set: only do first n combinations
                                           "model_delays": {"anthropic": 5} # if model name in keys, delay n seconds before next call (for simple rate limiting) 
                                          })

with open("last_result.txt", "w", encoding="utf-8") as f:
  f.write(str(res))

print(TestSet.report(res))
