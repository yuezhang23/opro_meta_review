import datetime
import functools
import os
import sys
from dotenv import dotenv_values
OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

from absl import app
from absl import flags
import numpy as np
import openai
from opro import prompt_utils
from opro.optimization import opt_utils
import pandas as pd

from liquid import Template

config = dotenv_values(os.path.join(OPRO_ROOT_PATH, ".env"))
ROOT_DATA_FOLDER_PATH = os.path.join(OPRO_ROOT_PATH, "data")

_SCORER = flags.DEFINE_string(
    "scorer", "gpt-4o-mini", "The name of the scorer LLM."
)

_OPTIMIZER = flags.DEFINE_string(
    "optimizer", "gpt-4o-mini", "The name of the optimizer LLM."
)

_DATASET = flags.DEFINE_string(
    "dataset", "metareview", "The name of dataset to search for instructions on."
)

_TASK = flags.DEFINE_string(
    "task",
    "train, test",
    "The name of task(s) within the above dataset to search for instructions on. Multiple tasks can be specified using comma separation (e.g. 'train,test').",
)

_INSTRUCTION_POS = flags.DEFINE_string(
    "instruction_pos",
    "Q_begin",
    "The position of the instruction to search for.",
) 

_INITIAL_PROMPTS = flags.DEFINE_string(
    "initial_prompts",
    "Given the reviews (Text), answer if a paper would be accepted (Yes) or not (No).", 
    "The initial instructions to search for.",
)

_META_PROMPT_TYPE = flags.DEFINE_string(
    "meta_prompt_type",
    "both_instructions_and_exemplars",
    "The type of meta-prompt: whether to have both previous instructions and"
    " dataset exemplars (often for fine-tuned optimizers), or to have only"
    " previous instructions (often for pre-trained optimizers).",
)

def main(_):
  openai_api_key = config["OPENAI_API_KEY"]
  scorer_llm_name = _SCORER.value
  optimizer_llm_name = _OPTIMIZER.value
  dataset_name = _DATASET.value.lower()
  task_name = _TASK.value
  meta_prompt_type = _META_PROMPT_TYPE.value

  assert dataset_name == "metareview", "Only metareview dataset is supported."

  assert scorer_llm_name in {
      "gpt-4.1-nano",
      "gpt-4o-mini",
  }, "Scorer must be either gpt-4.1-nano or gpt-4o-mini"
  
  assert optimizer_llm_name in {
      "gpt-4.1-nano",
      "gpt-4o-mini",
  }, "Optimizer must be either gpt-4.1-nano or gpt-4o-mini"
  
  assert meta_prompt_type in {
      "both_instructions_and_exemplars",
      "instructions_only",
  }

  instruction_pos = _INSTRUCTION_POS.value
  assert instruction_pos in {
      "before_Q",
      "Q_begin",
      "Q_end",
      "A_begin",
  }, (
      "The instruction position should be either before the question, or at the"
      " beginning of the question, at the end of the question, or at the"
      " beginning of the answer."
  )
  print(
      f"scorer: {scorer_llm_name}, optimizer: {optimizer_llm_name}, dataset:"
      f" {dataset_name}, task: {task_name}, instruction_pos: {instruction_pos}"
      f" initial_prompts: {_INITIAL_PROMPTS.value}"
  )

  # make sure the scorer and optimizer models are callable
  assert openai_api_key, "The OpenAI API key must be provided."
  openai.api_key = openai_api_key

  root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "review_data")

  # =================== create the result directory ==========================
  datetime_str = (
      str(datetime.datetime.now().replace(microsecond=0))
      .replace(" ", "-")
      .replace(":", "-")
  )

  save_folder = os.path.join(
      OPRO_ROOT_PATH,
      "outputs",
      "optimization-results",
      f"test-{task_name}-s-{scorer_llm_name}-o-{optimizer_llm_name}/",
  )
  result_by_instruction_folder = os.path.join(
      save_folder, "result_by_instruction"
  )
  os.makedirs(result_by_instruction_folder)
  print(f"result directory:\n{save_folder}")

  # ====================== scorer model configs ==============================
  scorer_gpt_max_decode_steps = 1024
  scorer_gpt_temperature = 0.0

  scorer_gpt_dict = dict()
  scorer_gpt_dict["max_decode_steps"] = scorer_gpt_max_decode_steps
  scorer_gpt_dict["temperature"] = scorer_gpt_temperature
  scorer_gpt_dict["num_decodes"] = 1
  scorer_gpt_dict["batch_size"] = 6
  scorer_gpt_dict["num_servers"] = 8

  scorer_llm_dict = {
      "model_type": scorer_llm_name.lower(),
  }
  scorer_llm_dict.update(scorer_gpt_dict)
  call_scorer_server_func = functools.partial(
      prompt_utils.call_openai_server_func,
      model=scorer_llm_name.lower(),
      max_decode_steps=scorer_gpt_max_decode_steps,
      temperature=scorer_gpt_temperature,
  )

  # ====================== optimizer model configs ============================
  optimizer_gpt_max_decode_steps = 1024
  optimizer_gpt_temperature = 1.35

  optimizer_llm_dict = dict()
  optimizer_llm_dict["max_decode_steps"] = optimizer_gpt_max_decode_steps
  optimizer_llm_dict["temperature"] = optimizer_gpt_temperature
  optimizer_llm_dict["batch_size"] = 1
  optimizer_llm_dict["num_decodes"] = 1
  call_optimizer_server_func = functools.partial(
      prompt_utils.call_openai_server_func,
      model=optimizer_llm_name,
      max_decode_steps=optimizer_gpt_max_decode_steps,
      temperature=optimizer_gpt_temperature,
  )

  # ====================== try calling the servers ============================
  print("\n======== testing the scorer and optimizer servers ===========")
  scorer_test_output = call_scorer_server_func(
      ["Does the sun rise from the north? Just answer yes or no.", "is the sky blue? just answer yes or no"],
      temperature=0.0,
  )
  print(f"number of scorer output decodes: {len(scorer_test_output)}")
  print(f"scorer test output: {scorer_test_output}")
  
  optimizer_test_output = call_optimizer_server_func(
      "Does the sun rise from the north? Just answer yes or no.",
      temperature=1.3,
  )
  print(f"number of optimizer output decodes: {len(optimizer_test_output)}")
  print(f"optimizer test output: {optimizer_test_output}")
  print("Finished testing the servers.")

  # ====================== read data ============================
  print("\n================ prompt optimization settings ==============")
  tasks_all = task_name.split(',')  
  multiple_choice_tasks = set()
  boolean_tasks = set(tasks_all)
  numerical_output_tasks = set()

  raw_data = pd.DataFrame()
  prediction_treat_as_number = False
  prediction_treat_as_bool = True

  for task_name in tasks_all:
    f_gsm = os.path.join(root_data_folder_path, f"{task_name}.csv")
    single_task_df = pd.read_csv(f_gsm, sep=";", header=None)
    raw_data = pd.concat([raw_data, single_task_df])

  raw_data = raw_data.sample(500)
  num_examples = len(raw_data)
  print(f"number of examples: {num_examples}")

  # ================get test ==========================
#   train_index = np.sort(
#       np.array(
#           np.random.choice(
#               num_examples, size=int(train_ratio * num_examples), replace=False
#           )
#       )
#   )
#   eval_and_test_index = np.sort(
#       np.array(list(set(np.arange(num_examples)) - set(train_index)))
#   )
  eval_index = np.sort(
      np.array(
          np.random.choice(
              eval_and_test_index,
              size=int(eval_ratio * num_examples),
              replace=False,
          )
      )
  )

  # ========== set other optimization experiment hyperparameters ==============
  old_instruction_score_threshold = 0.5
  extract_final_answer_by_prompting_again = True
  include_qa = False
  evaluate_in_parallel = True

  optimizer_llm_temperature = optimizer_llm_dict["temperature"]

  num_few_shot_questions_for_instruction_refinement = 3
  num_generated_instructions_in_each_step = 8
  num_search_steps = 100

  initial_instructions = [_INITIAL_PROMPTS.value]

  few_shot_qa_pairs = True
  few_shot_selection_criteria = 'current_most_frequent'
  evaluate_generated_ins_on_few_shot = True
  evaluate_old_ins_on_few_shot = False
  eval_interval = 4

  max_num_instructions = 6
  num_score_buckets = 100
  meta_prompt_instructions_before_exemplars = True

  # ===================== run prompt optimization ======================
  assert few_shot_selection_criteria in {
      "accumulative_most_frequent",
      "current_most_frequent",
      "random",
      "constant",
  }
  evolution_kwargs = {
      "num_search_steps": num_search_steps,
      "old_instruction_score_threshold": old_instruction_score_threshold,
      "scorer_llm_dict": scorer_llm_dict,
      "optimizer_llm_dict": optimizer_llm_dict,
      "extract_final_answer_by_prompting_again": (
          extract_final_answer_by_prompting_again
      ),
      "include_qa": include_qa,
      "evaluate_in_parallel": evaluate_in_parallel,
      "tasks_all": tasks_all,
      "train_ratio": train_ratio,
      "eval_ratio": eval_ratio,
      "test_ratio": test_ratio,
      "train_index": train_index,
      "eval_index": eval_index,
      "dataset_name": dataset_name,
      "task_name": task_name,
      "num_examples": num_examples,
      "root_data_folder_path": root_data_folder_path,
      "optimizer_llm_temperature": optimizer_llm_temperature,
      "initial_instructions": initial_instructions,
      "multiple_choice_tasks": multiple_choice_tasks,
      "raw_data": raw_data,
      "call_scorer_server_func": call_scorer_server_func,
      "call_optimizer_server_func": call_optimizer_server_func,
      "instruction_pos": instruction_pos,
      "prediction_treat_as_number": prediction_treat_as_number,
      "prediction_treat_as_bool": prediction_treat_as_bool,
      "result_by_instruction_folder": result_by_instruction_folder,
      "few_shot_qa_pairs": few_shot_qa_pairs,
      "num_score_buckets": num_score_buckets,
      "max_num_instructions": max_num_instructions,
      "meta_prompt_type": meta_prompt_type,
      "meta_prompt_instructions_before_exemplars": (
          meta_prompt_instructions_before_exemplars
      ),
      "few_shot_selection_criteria": few_shot_selection_criteria,
      "optimizer_llm_name": optimizer_llm_name,
      "num_generated_instructions_in_each_step": (
          num_generated_instructions_in_each_step
      ),
      "evaluate_generated_ins_on_few_shot": evaluate_generated_ins_on_few_shot,
      "num_few_shot_questions_for_instruction_refinement": (
          num_few_shot_questions_for_instruction_refinement
      ),
      "evaluate_old_ins_on_few_shot": evaluate_old_ins_on_few_shot,
      "eval_interval": eval_interval,
      "save_folder": save_folder,
      "verbose": True
  }

  opt_utils.run_evolution(**evolution_kwargs)

if __name__ == "__main__":
  app.run(main)
