from typing import Any, Optional, Union, Callable, Annotated, Literal, get_origin, get_args
import os, time, copy

import yaml
from pprint import pprint

from tqdm.autonotebook import tqdm

from pydantic import BaseModel
from dataclasses import dataclass, is_dataclass

import logging
logger = logging.getLogger(__name__)

from sibila import (
    Model, Models,
    GenConf,
    Thread, Msg,    
)

from utils import (
    execx
)

from evalutils import (
    equal, same,
    str_sub, istr_sub
)

SCRIPT_GLOBALS = {
    "equal": equal,
    "same": same,
    "str_sub": str_sub,
    "istr_sub": istr_sub,
}




class TestSet():

    def __init__(self,
                 path: str):
        with open(path, "r", encoding="utf-8") as f:
            test_defs = yaml.safe_load(f.read())

        self.setup = test_defs.get("setup", {})
        self.tests = test_defs.get("tests", [])

        # parse setup.script if any
        self.script_globals = {}
        if "script" in self.setup:
            execx(self.setup["script"], 
                  self.script_globals,
                  description=f"script for test '{os.path.basename(path)}'")

    

    def run_tests_for_models(self,
                             model_names: list[str],
                             options: dict = {},
                             callables: Optional[dict] = None) -> dict:
        
        out = {}

        pbar = tqdm(model_names, desc="Models", leave=False)
        for model_name in pbar:
            pbar.set_description("Model " + model_name)

            model = Models.create(model_name)

            res = self.run_tests(model, 
                                 options,
                                 callables)
            out[model_name] = res

            model.close()

        return out
    



        
    def run_tests(self,
                  model: Model,
                  options: dict = {},
                  callables: Optional[dict] = None):
        
        """
        options:
            "only_tests": list[int] -> only run tests whit these indices
            "first_n_runs": int -> only run the first n runs of each test
        
        """

        only = options.get("only_tests", [])

        for_testing = []
        for index,test in enumerate(self.tests):
            if not only or index in only:
                for_testing.append(index)

        score_sum = 0.
        out = []
        for index in tqdm(for_testing, desc="Tests", leave=False):

            test_def = self.tests[index]
            result = self.run_test(model, 
                                   test_def,
                                   options,
                                   callables)

            score_sum += result["test_score"]
            out.append(result)

        testset_score = score_sum / (len(out) or 1)

        return {
                "testset_score": testset_score,
                "testset": out
                }



    def run_test(self,
                 model: Model,
                 test_def: dict,
                 options: dict = {},
                 callables: Optional[dict] = None) -> dict:
        """ 
        def generate(model: Model,
                     inst_text: str, 
                     in_text: str) -> Any

        Returns a dict of:
            "in_text_tmpl": in_text_tmpl,
            "test_score": score_mean,
            "test_runs": [{"result": ..., "score": ...}]
        """
                
        if callables is not None:
            generate = callables.get("generate")
            evaluate = callables.get("evaluate")
        else:
            generate = evaluate = None

        if generate is None:
            generate = self.script_globals.get("generate")
        if evaluate is None:
            evaluate = self.script_globals.get("evaluate")

        if generate is None:
            raise ValueError("Could not find generate() in setup.script. Please provide it in callables arg or define in YAML setup.script")
        if evaluate is None:
            # raise ValueError("Could not find evaluate() in setup.script. Please provide it in callables arg or define in YAML setup.script")            
            def stock_evaluate(value: Any, 
                               expected: dict):
                sub_scores = []

                for field in expected:
                    if isinstance(value, BaseModel) or is_dataclass(value):
                        field_value = getattr(value,field)
                    elif isinstance(value, dict):
                        field_value = value[field]
                    else:
                        field_value = value

                    score = same(field_value, expected[field])
                    sub_scores.append(float(score))

                return sum(sub_scores) / len(sub_scores)
            evaluate = stock_evaluate


        model_delay = None
        if 'model_delays' in options and isinstance(options['model_delays'], dict):
            model_desc = model.desc().lower()
            for name,delay in options['model_delays'].items():
                if name.lower() in model_desc:
                    model_delay = delay
                    break


        # resolve inst_text, in_text, check local, then setup's
        inst_text_tmpl = test_def.get("inst")
        if inst_text_tmpl is None:
            inst_text_tmpl = self.setup.get("inst", "")

        in_text_tmpl = test_def.get("in")
        if in_text_tmpl is None:
            in_text_tmpl = self.setup.get("in")
        if not in_text_tmpl:
            raise ValueError(f"IN text not found at {test_def}")
        
        # expected cascades fom setup
        expected_tmpl = test_def.get("expected")
        if expected_tmpl is None:
            expected_tmpl = self.setup.get("expected")
        if not expected_tmpl:
            raise ValueError(f"Could not find 'expected' key at test_def nor in its setup")


        # aggregate setup and local test vars
        all_vars = self.setup.get("vars", {}).copy()
        all_vars.update(test_def.get("vars", {}))
        logger.debug("All vars: {all_vars}")


        # filter-out unused vars in inst and in text
        all_text_tmpl = inst_text_tmpl + in_text_tmpl + str(expected_tmpl)
        used_vars = []
        used_combis = []
        for var_name in all_vars:
            if var_name in all_text_tmpl:
                if var_name.endswith("*"):
                    used_combis.append(var_name)
                else:
                    used_vars.append(var_name)

        combis = {}
        for name in used_combis:
            combis[name] = {
                "values": all_vars[name],
                "count": len(all_vars[name]),
                "index": 0
            }


        # for all combinations
        # gather variable values, including combi actual values
        var_vals = {name: all_vars[name] for name in used_vars}

        scores = []
        combi_index = 0
        while(True):

            for combi_name in combis:
                var_vals[combi_name] = combis[combi_name]["values"][combis[combi_name]["index"]]


            # expand inst and in messages            
            inst_text = prepare_render_fstring(inst_text_tmpl,
                                               var_vals,
                                               SCRIPT_GLOBALS)
            in_text = prepare_render_fstring(in_text_tmpl,
                                             var_vals,
                                             SCRIPT_GLOBALS)

            # print(in_text, expected_tmpl)

            # generate
            try:
                res = generate(model, inst_text, in_text)
            except Exception as e:
                model.close()
                raise e


            # score
            score_res = self.score_result(res, 
                                          expected_tmpl, 
                                          var_vals,
                                          SCRIPT_GLOBALS,
                                          evaluate)

            logger.debug(f"--------- {combi_index}: {var_vals} -> {score_res}")

            scores.append({"in_text": in_text,
                           "result": res,
                           "expected": score_res["expected"],
                           "score": score_res["score"]})


            if model_delay is not None:
                time.sleep(model_delay)

            if 'first_n_runs' in options and options['first_n_runs']:
                if combi_index + 1 >= options["first_n_runs"]:
                    break


            # increment combis' indices
            all_done = False
            rev_combi_keys = list(combis.keys())[::-1]
            carry = False
            for index, combi_name in enumerate(rev_combi_keys):
                entry = combis[combi_name]
                next_val = entry["index"] + carry
                if index == 0: # increment less significative = last
                    next_val += 1
                if  next_val >= entry["count"]:
                    if index == len(rev_combi_keys)-1: # done, last key is about to be incremented
                        all_done = True
                        break
                    entry["index"] = 0
                    carry = True
                else:
                    entry["index"] = next_val

            if all_done:
                break

            combi_index += 1


        score_mean = sum([entry["score"] for entry in scores]) / (len(scores) or 1)

        return {
            "in_text_tmpl": in_text_tmpl,
            "test_score": score_mean,
            "test_runs": scores
        }



    def score_result(self,
                     res: Any,
                     expected_tmpl: Union[dict,list[dict]],
                     var_vals: dict,
                     globals: dict,
                     item_evaluate_fn: Callable) -> dict:
        """
        res is the generated result, which can be a list of items or a single item
        expected_tmpl is a single dict or a list of dict with fields and expected results

        If res is:
            - list[item]: expected_tmpl will be a list and all values will be searched for exact match. More res values than expected=>0 score
            - item: 
                - if expected_tmpl is a list: will match any in the list and use the max score
                - if expected_tmpl is an item: will do an item match

        expected_tmpl items will be variable expanded

        Returns a dict of:
            "score": 0..1
            "expected":
        """


        # expand expected_tmpl vars
        expected = copy.deepcopy(expected_tmpl)
        if not isinstance(expected, list):
            expected = [expected]

        #print(expected)
        for exp_entry in expected:

            for name,tmpl in exp_entry.items():
                clean_tmpl,clean_var_vals = sanitize_format_vars(tmpl, var_vals)
                globs = globals.copy()
                globs.update(clean_var_vals)
                #print(name, tmpl,clean_tmpl,globs)

                try:
                    evald = eval(clean_tmpl, globs)
                except NameError: # pass as a literal
                    evald = clean_tmpl

                exp_entry[name] = evald


        if isinstance(res, list):
                
            # list to list: same size?
            if len(res) != len(expected):
                return {
                    "score": 0.,
                    "expected": expected
                }
            
            match1_count = 0
            for exp in expected:
                for r in res:
                    sc = item_evaluate_fn(r, exp)
                    if sc == 1.:
                        match1_count += 1
                        break

            return {
                "score": match1_count / len(res),
                "expected": expected
            }
        
        else: # res is an item: an any/or match, where max score is returned

            max_score = 0.
            for exp in expected:
                sc = item_evaluate_fn(res, exp)
                max_score = max(sc, max_score)
            return {
                "score": max_score,
                "expected": expected
            }



    @staticmethod
    def report(res: dict):

        testset_score_sum = 0.
        for model_name in res:
            model_res = res[model_name]
            testset_score_sum += model_res['testset_score']

        mean_testset_score = testset_score_sum / len(res)

        # find tests count and test_runs
        first_model_name = list(res.keys())[0]
        model_res = res[first_model_name]
        test_count = len(model_res['testset'])
        test_run_count = 0
        for test in model_res['testset']:
            test_run_count += len(test["test_runs"])

        out = f"Mean score for all models: {mean_testset_score:.3f}\n"
        out += "Model scores:\n"

        for model_name in res:
            model_res = res[model_name]
            out += (f"  {model_name}: {model_res['testset_score']:.3f}\n")

        out += f"Total {test_count} tests, {test_run_count} runs.\n"

        header_added = False
        for model_name in res:
            model_res = res[model_name]

            if model_res['testset_score'] < 1.:
                if not header_added:
                    out += "\nIncorrect answers per model:\n"
                    header_added = True

                out += f"= {model_name}: {model_res['testset_score']:.3f} ===================\n"

                for test in model_res['testset']:
                    for test_run in test["test_runs"]:
                        if test_run["score"] != 1.0:
                            out += str(test_run) + "\n"


        return out









def sanitize_format_vars(format: str,
                         var_vals: dict) -> tuple[str, dict]:
    
    out_var_vals = {}
    for name in var_vals:
        if name.endswith("*"):
            bare_name = name[:-1]
            out_var_vals[bare_name] = var_vals[name]
            format = format.replace(name, bare_name)

        else:
            out_var_vals[name] = var_vals[name]

    return format, out_var_vals



def prepare_render_fstring(format: str,
                           var_vals: dict,
                           globals: dict) -> str:
    
    format,var_vals = sanitize_format_vars(format, var_vals)
    glob = globals.copy()
    glob.update(var_vals)

    return render_fstring(format, glob, "literal")



def render_fstring(format: str,
                   globals: dict,
                   error_mode: str = "raise") -> str:

    """Split int text and {eval} groups. Only evals innermost {} blocks
    error_mode: what to do if an eval block errors -> raise, literal, remove
    """

    span= [] # array of tuples of (should_eval, text)
    remain = format
    while remain:
        
        pos = remain.find("{")
        if pos == -1:
            span.append((False,remain))
            break
        elif pos == 0: # don't emit text span
            remain = remain[pos+1:]
        else:
            span.append((False, remain[:pos]))
            remain = remain[pos+1:]
            
        # find closing }
        pos = remain.find("}")
        if pos == -1:
            raise ValueError(f"Missing closing '}}' at '{format}'")
        
        # find any opening '{' before next closing '}' -> this means it's not an eval
        open_pos = remain.find("{")
        if open_pos != -1 and open_pos < pos: # emit text span till open '{' and continue from there
            span.append((False, '{' + remain[:open_pos]))
            remain = remain[open_pos:]
        else:
            span.append((True, remain[:pos]))
            remain = remain[pos+1:]

    out = ""
    for is_eval, text in span:
        if is_eval:
            try:
                res = eval('f"{' + text + '}"',
                        globals)
                res = str(res)
            except Exception as e:
                if error_mode == 'raise':
                    raise e
                elif error_mode == 'literal':
                    res = "{" + text + "}"
                else:
                    res = ''

        else: # plain text
            res = text

        out += res

    return out





