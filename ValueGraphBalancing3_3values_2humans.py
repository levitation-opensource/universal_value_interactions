# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This code was developed based on research and ideas of Lenz
# https://github.com/ramennaut
# 
# Coded by Roland 
# https://github.com/levitation
#
# Repository: https://github.com/levitation-opensource/universal_value_interactions


import os
import numpy as np
from matplotlib import pyplot as plt
import yaml
import random


def init_matrix(negative_interaction_matrix_dict, positive_interaction_matrix_dict):

  # check that each value_name is represented in the interaction matrix
  for value_name in value_names:
    assert negative_interaction_matrix_dict.get(value_name) is not None
    assert positive_interaction_matrix_dict.get(value_name) is not None


  # check the interaction matrices for consistency
  for value1, value1_data in negative_interaction_matrix_dict.items():
    for value2, interaction in value1_data.items():
      assert negative_interaction_matrix_dict[value2][value1] == interaction
      assert positive_interaction_matrix_dict[value1].get(value2) is None

  for value1, value1_data in positive_interaction_matrix_dict.items():
    for value2, interaction in value1_data.items():
      assert positive_interaction_matrix_dict[value2][value1] == interaction
      assert negative_interaction_matrix_dict[value1].get(value2) is None


  # create numpy format interaction matrix
  interaction_matrix = np.zeros([num_value_names, num_value_names])
  positive_interaction_matrix = np.zeros([num_value_names, num_value_names])
  negative_interaction_matrix = np.zeros([num_value_names, num_value_names])

  for value1, value1_data in negative_interaction_matrix_dict.items():
    index1 = value_names.index(value1)   # do not use enumerate() here for case the value_names are in a different order
    for value2, interaction in value1_data.items():
      index2 = value_names.index(value2)   # cannot use enumerate() here since not all keys are present
      interaction_matrix[index1, index2] = interaction
      negative_interaction_matrix[index1, index2] = interaction

  for value1, value1_data in positive_interaction_matrix_dict.items():
    index1 = value_names.index(value1)   # do not use enumerate() here for case the value_names are in a different order
    for value2, interaction in value1_data.items():
      index2 = value_names.index(value2)   # cannot use enumerate() here since not all keys are present
      interaction_matrix[index1, index2] = interaction
      positive_interaction_matrix[index1, index2] = interaction

  assert np.array_equal(interaction_matrix, interaction_matrix.T)   # check that the matrix was populated correctly - the matrix has to be symmetric


  return interaction_matrix, positive_interaction_matrix, negative_interaction_matrix

#/ def init_matrix():


def init():
  (
    between_agents_interaction_matrix,
    between_agents_positive_interaction_matrix,
    between_agents_negative_interaction_matrix,
  ) = init_matrix(
    between_agents_negative_interaction_matrix_dict,
    between_agents_positive_interaction_matrix_dict,
  )

  (
    self_feedback_interaction_matrix,
    self_feedback_positive_interaction_matrix,
    self_feedback_negative_interaction_matrix,
  ) = init_matrix(
    self_feedback_negative_interaction_matrix_dict,
    self_feedback_positive_interaction_matrix_dict,
  )

  return (
    between_agents_interaction_matrix,
    between_agents_positive_interaction_matrix,
    between_agents_negative_interaction_matrix,
    self_feedback_interaction_matrix,
    self_feedback_positive_interaction_matrix,
    self_feedback_negative_interaction_matrix,
  )

#/ def init():


def prettyprint(data):
  print(yaml.dump(data, allow_unicode=True, default_flow_style=False))


def custom_sigmoid10(data):
  signs = np.sign(data)
  logs = np.log10(np.abs(data) + 1)     # offset by +1 to avoid negative logarithm values
  return logs * signs


def custom_sigmoid(data):
  signs = np.sign(data)
  logs = np.log(np.abs(data) + 1)     # offset by +1 to avoid negative logarithm values
  return logs * signs


def tiebreaking_argmax(arr):
  max_values_bitmap = np.isclose(arr, arr.max())
  max_values_indexes = np.flatnonzero(max_values_bitmap)

  if len(max_values_indexes) == 0:  # Happens when all values are infinities or nans. This would cause np.random.choice to throw.
    result = np.random.randint(0, len(arr))
  else:
    result = np.random.choice(max_values_indexes)  # TODO: seed for this random generator

  return result


def plot_agent_history(subplots, plots_column, agent_name, values_history, utilities_history):

  linewidth = 0.75  # TODO: config


  subplot = subplots[plots_column, 0]
  for index, value_name in enumerate(value_names):
    subplot.plot(
      values_history[:, index],
      label=value_name,
      linewidth=linewidth,
    )

  subplot.set_title(f"{agent_name} - Value level evolution")
  subplot.set(xlabel="step", ylabel="raw value level")
  subplot.legend()


  subplot = subplots[plots_column, 1]
  for index, value_name in enumerate(value_names):
    subplot.plot(
      custom_sigmoid10(values_history[:, index]),
      label=value_name,
      linewidth=linewidth,
    )

  subplot.set_title(f"{agent_name} - Sigmoid10 of Value level")
  subplot.set(xlabel="step", ylabel="custom_sigmoid10(raw value level)")
  subplot.legend()


  subplot = subplots[plots_column, 2]
  for index, value_name in enumerate(value_names):
    subplot.plot(
      utilities_history[:, index],
      label=value_name,
      linewidth=linewidth,
    )

  subplot.set_title(f"{agent_name} - Utilities evolution")
  subplot.set(xlabel="step", ylabel="utility level")
  subplot.legend()


  subplot = subplots[plots_column, 3]
  for index, value_name in enumerate(value_names):
    subplot.plot(
      custom_sigmoid10(utilities_history[:, index]),
      label=value_name,
      linewidth=linewidth,
    )

  subplot.set_title(f"{agent_name} - Sigmoid10 of Utilities")
  subplot.set(xlabel="step", ylabel="custom_sigmoid10(utility level)")
  subplot.legend()


  # TODO: std or gini index over values per timestep plot

#/ def plot_agent_history(values_history, utilities_history, utility_function_mode, rebalancing_mode):


def plot_history(values_history_dict, utilities_history_dict, utility_function_mode, rebalancing_mode):

  fig, subplots = plt.subplots(2, 4)


  fig.suptitle(f"Value graph balancing - utility function: {utility_function_mode} - rebalancing: {rebalancing_mode}")


  agent_name = agent_names[0]
  plot_agent_history(
    subplots,
    0,  # plots_column
    agent_name.upper(),
    values_history_dict[agent_name],
    utilities_history_dict[agent_name],
  )

  agent_name = agent_names[1]
  plot_agent_history(
    subplots,
    1,  # plots_column
    agent_name.upper(),
    values_history_dict[agent_name],
    utilities_history_dict[agent_name],
  )


  plt.ion()
  # maximise_plot()
  fig.show()
  plt.draw()
  plt.pause(60)  # render the plot. Usually the plot is rendered quickly but sometimes it may require up to 60 sec. Else you get just a blank window

  wait_for_enter("Press enter to close the plot")

#/ def plot_history(history):


def wait_for_enter(message=None):
  if os.name == "nt":
    import msvcrt

    if message is not None:
      print(message)
    msvcrt.getch()  # Uses less CPU on Windows than input() function. This becomes perceptible when multiple console windows with Python are waiting for input. Note that the graph window will be frozen, but will still show graphs.
  else:
    if message is None:
      message = ""
    input(message)


def compute_utilities(prev_actual_values, updated_actual_values, prev_utilities, utility_function_mode):

  value_changes = updated_actual_values - prev_actual_values

  positive_actual_values = np.maximum(updated_actual_values, 0) 
  negative_actual_values = np.minimum(updated_actual_values, 0) 

  # NB! this is not same as *_interaction_value_changes since here we filter by the sign of the change, not sign of the interaction
  positive_value_changes = np.maximum(value_changes, 0) 
  negative_value_changes = np.minimum(value_changes, 0) 

  if utility_function_mode == "linear":
    utilities = updated_actual_values

  elif utility_function_mode == "sigmoid":
    utilities = custom_sigmoid(updated_actual_values)

  elif utility_function_mode == "prospect_theory":   # sigmoid is applied to value CHANGES not to RESULTING values. ALSO: negative side is amplified.
    # NB! current logic amplifies LOSS, irrespective whether the resulting value is positive of negative. 
    change_utilities = custom_sigmoid(positive_value_changes) + custom_sigmoid(negative_value_changes) * 2   # TODO: config parameter
    # utilities = prev_utilities + change_utilities
    utilities = 0.5 * prev_utilities + change_utilities   # TODO: parameter for past utilities discounting

  elif utility_function_mode == "concave":   # positive side is logarithmic similarly to sigmoid, but negative side is treated exponentially
    # SFELLA formula: https://link.springer.com/article/10.1007/s10458-022-09586-2
    positive_updated_utilities = np.log(positive_actual_values + 1)
    negative_updated_utilities = 1 - np.exp(-negative_actual_values)
    utilities = positive_updated_utilities + negative_updated_utilities

  elif utility_function_mode == "linear_homeostasis":   # too much of an actual value reduces the subjective value (utility)
    diff_from_targets = np.abs(updated_actual_values - target_values)
    # diff_from_targets = np.power(diff_from_targets, 2)    # TODO: parameter
    utilities = -0.1 * diff_from_targets    # linear mode

  elif utility_function_mode == "squared_homeostasis":   # too much of an actual value reduces the subjective value (utility)
    diff_from_targets = np.abs(updated_actual_values - target_values)
    # diff_from_targets = np.power(diff_from_targets, 2)    # TODO: parameter
    utilities = -0.01 * diff_from_targets * diff_from_targets    # squared error mode

  else:
    raise Exception("Unknown utility_function_mode")

  return utilities

#/ def compute_utilities(actual_values):


def main(utility_function_mode, rebalancing_mode):

  (
    between_agents_interaction_matrix,
    between_agents_positive_interaction_matrix,
    between_agents_negative_interaction_matrix,
    self_feedback_interaction_matrix,
    self_feedback_positive_interaction_matrix,
    self_feedback_negative_interaction_matrix,
  ) = init()


  actual_values_dict = {}
  utilities_dict = {}
  values_history_dict = {}
  utilities_history_dict = {}

  for agent_name in agent_names:

    # TODO!!!: init prev values and utilities to be equal to initial actuals and utilities? It is not like the world suddenly jumped into existence and there was nothing before.
    prev_actual_values = np.zeros([num_value_names])
    prev_utilities = np.zeros([num_value_names])

    if utility_function_mode == "linear_homeostasis" or utility_function_mode == "squared_homeostasis":  # NB! in case of homeostatic utilities, the initial values cannot be too far off targets, else the system never recovers
      actual_values = homeostatic_utility_scenario_actual_values.copy()  # NB! copy since this matrix might be modified in place later
    else:
      actual_values = initial_actual_values.copy()  # NB! copy since this matrix might be modified in place later


    utilities = compute_utilities(prev_actual_values, actual_values, prev_utilities, utility_function_mode)

    values_history = np.zeros([experiment_length, num_value_names])
    utilities_history = np.zeros([experiment_length, num_value_names])

    actual_values_dict[agent_name] = actual_values
    utilities_dict[agent_name] = utilities
    values_history_dict[agent_name] = values_history
    utilities_history_dict[agent_name] = utilities_history

  #/ for agent_name in agent_names:


  for step in range(0, experiment_length):

    updated_utilities_dict = {}
    updated_actual_values_dict = {}

    for agent_index, agent_name in enumerate(agent_names):
      other_agent_name = agent_names[1 - agent_index]

      self_utilities = utilities_dict[agent_name]
      other_utilities = utilities_dict[other_agent_name]

      self_actual_values = actual_values_dict[agent_name]


      # compute between-agent-interactions

      interaction_matrix = between_agents_interaction_matrix
      positive_interaction_matrix = between_agents_positive_interaction_matrix
      negative_interaction_matrix = between_agents_negative_interaction_matrix

      # NB! the raw value level changes are computed based on interactions with utilities, not on interactions between raw value levels
      if not restrict_negative_interactions:
        value_changes1 = np.matmul(other_utilities, interaction_matrix) * value_interaction_rate
      else:
        positive_interaction_value_changes = np.matmul(other_utilities, positive_interaction_matrix) * value_interaction_rate
        negative_interaction_value_changes = np.matmul(np.maximum(other_utilities, 0), negative_interaction_matrix) * value_interaction_rate  # np.maximum: in case of negative interactions, ignore negative actual values
        value_changes1 = positive_interaction_value_changes + negative_interaction_value_changes


      # compute self-feedback-interactions

      interaction_matrix = self_feedback_interaction_matrix
      positive_interaction_matrix = self_feedback_positive_interaction_matrix
      negative_interaction_matrix = self_feedback_negative_interaction_matrix

      # NB! the raw value level changes are computed based on interactions with utilities, not on interactions between raw value levels
      if not restrict_negative_interactions:
        value_changes2 = np.matmul(self_utilities, interaction_matrix) * value_interaction_rate
      else:
        positive_interaction_value_changes = np.matmul(self_utilities, positive_interaction_matrix) * value_interaction_rate
        negative_interaction_value_changes = np.matmul(np.maximum(self_utilities, 0), negative_interaction_matrix) * value_interaction_rate  # np.maximum: in case of negative interactions, ignore negative actual values
        value_changes2 = positive_interaction_value_changes + negative_interaction_value_changes


      # compute utilities from updated actual values

      self_updated_actual_values = self_actual_values + value_changes1 + value_changes2

      self_utilities = compute_utilities(
        self_actual_values,
        self_updated_actual_values,
        self_utilities,
        utility_function_mode,
      )
      self_actual_values = self_updated_actual_values

      # do not broadcast the updates until both agents have computed their updates, until then store in updated_* variables
      updated_utilities_dict[agent_name] = self_utilities
      updated_actual_values_dict[agent_name] = self_actual_values

    #/ for agent_name in agent_names:

    # lets broadcast the updates now into the main dicts
    utilities_dict = updated_utilities_dict
    actual_values_dict = updated_actual_values_dict


    # value rebalancing phase
    # for time being, lets assume that the rebalancing mechanism can directly affect only the human's value levels
    # the agent's value levels will be affected indirectly
    # human is chosen as rebalancing target here because this simple logic below would not be able to rebalance the human through agent's value levels
    # TODO: let an LLM or RL rebalance directly the agent's value levels only, while the actual rebalancing priority is on human value levels, which are affected then indirectly only
    # TODO: optional setup for affecting both agent's and human's value levels directly during rebalancing

    rebalanced_agent_name = random.choice(agent_names)    # lets make the scenario more interesting by imposing a random constraint on who can be rebalanced
    actual_values = actual_values_dict[rebalanced_agent_name]

    # TODO: refactor this rebalancing code block into a separate function

    rebalanced_actual_values = actual_values.copy()

    # TODO: option to require removal or addition of resources to some other value when current most extreme value is adjusted, so that the sum total remains same

    if rebalancing_mode == "none":

      pass

    elif rebalancing_mode == "llm":

      pass  # TODO: implement an LLM that does the rebalancing. Lets see whether LLM is at least as good as the simple fixed formulas below.

    elif rebalancing_mode == "homeostatic":

      # a simple agent that chooses one most extreme value (as compared to the value's target) and rebalances it at most by 1 unit. 
      # NB! This assumes that all values are homeostatic and THERE IS A DESIRED TARGET LEVEL FOR EACH VALUE.

      deviations_from_targets = actual_values - target_values
      absolute_deviations = np.abs(deviations_from_targets)
      max_deviation_index = tiebreaking_argmax(absolute_deviations)

      deviation = deviations_from_targets[max_deviation_index]
      if deviation < 0:
        balance_step = min(max_rebalancing_step_size, -deviation) # min(): if deviation magnitude is smaller than max_rebalancing_step_size then step by deviation magnitude only
      else:
        balance_step = -min(max_rebalancing_step_size, deviation) # min(): if deviation magnitude is smaller than max_rebalancing_step_size then step by deviation magnitude only

      rebalanced_actual_values[max_deviation_index] += balance_step
 
    elif rebalancing_mode == "homeostatic_boosting":    # TODO: implement also naive boost mode which chooses a value with lowest level regardless of the target value

      # a simple agent that chooses one least implemented value that is below the value's target level and rebalances it at most by 1 unit. 

      deviations_from_targets = actual_values - target_values
      max_deviation_index = tiebreaking_argmax(-deviations_from_targets)

      deviation = deviations_from_targets[max_deviation_index]
      if deviation < 0:
        balance_step = min(max_rebalancing_step_size, -deviation) # min(): if deviation magnitude is smaller than max_rebalancing_step_size then step by deviation magnitude only
      else:
        balance_step = 0

      rebalanced_actual_values[max_deviation_index] += balance_step

    elif rebalancing_mode == "homeostatic_throttling":    # TODO: implement also naive throttling mode which chooses a value with highest level regardless of the target value

      # a simple agent that chooses one most positive value above the value's target level and rebalances it at most by 1 unit. 

      deviations_from_targets = actual_values - target_values
      max_deviation_index = tiebreaking_argmax(deviations_from_targets)

      deviation = deviations_from_targets[max_deviation_index]
      if deviation > 0:
        balance_step = -min(max_rebalancing_step_size, deviation) # min(): if deviation magnitude is smaller than max_rebalancing_step_size then step by deviation magnitude only
      else:
        balance_step = 0

      rebalanced_actual_values[max_deviation_index] += balance_step
    
    else:
      raise Exception("Unknown rebalancing_mode")


    utilities = compute_utilities(actual_values, rebalanced_actual_values, utilities, utility_function_mode)
    actual_values = rebalanced_actual_values

    # lets broadcast the updates caused by rebalancing
    actual_values_dict[rebalanced_agent_name] = actual_values


    for agent_name in agent_names:
      utilities = utilities_dict[agent_name]
      actual_values = actual_values_dict[agent_name]

      values_history_matrix = values_history_dict[agent_name]
      utilities_history_matrix = utilities_history_dict[agent_name]

      values_history_matrix[step, :] = actual_values
      utilities_history_matrix[step, :] = utilities


    if False:
      for agent_name in agent_names:

        utilities = utilities_dict[agent_name]
        actual_values = actual_values_dict[agent_name]

        actual_values_with_names_dict = {
          value_name: "{:.3f}".format(actual_values[index])
          for index, value_name in enumerate(
            value_names
          )  # TODO: could also use zip instead of enumerate
        }
        utilities_with_names_dict = {
          value_name: "{:.3f}".format(utilities[index])
          for index, value_name in enumerate(
            value_names
          )  # TODO: could also use zip instead of enumerate
        }

        print(f"{agent_name.upper()} raw value levels:")
        prettyprint(actual_values_with_names_dict)
        print(f"{agent_name.upper()} utilities:")
        prettyprint(utilities_with_names_dict)

      #/ for agent_name in agent_names:

      print()
      print()

  #/ for step in range(0, experiment_length):


  plot_history(values_history_dict, utilities_history_dict, utility_function_mode, rebalancing_mode)

#/ def main():


if __name__ == "__main__":

  # values and interaction matrices

  value_names = [
    "Power",
    # "Achievement",
    # "Hedonism",  
    # "Stimulation",
    "Self-direction",
    # "Universalism",
    "Benevolence",
    # "Tradition",
    # "Conformity",
    # "Security",
  ]

  # TODO!!! Originally, between-agents and self-feedback interaction matrices were equal, but they probably should not be equal. Please adjust the numbers in the matrices to match the anthropological research.

  # TODO: the interaction between power and self-direction was added by Roland as an experiment. This needs to be validated.

  # for clarity purposes, using separate matrices for negative and positive interactions
  between_agents_negative_interaction_matrix_dict = {
    "Power": {
      "Power": -0.5,    # the other agent might lose power, but not necessarily
      # "Universalism": -1,
      "Self-direction": -0.5,    # the other agent might lose self-direction, but not necessarily
      # "Benevolence": -1,
      # "Tradition": -1,  # TODO
    },
    # "Achievement": {
    #   "Universalism": -1,
    #   "Benevolence": -1,
    #   "Tradition": -1,  # TODO
    # },
    # "Hedonism": {
    #   "Universalism": -1,     # TODO
    #   "Benevolence": -1,    # TODO
    #   "Tradition": -1,
    #   "Conformity": -1,
    # }, 
    # "Stimulation": {
    #   "Tradition": -1,
    #   "Conformity": -1,
    #   "Security": -1,
    # },
    "Self-direction": {
      "Power": -0.5,    # the other agent might lose power, but not necessarily
      # "Tradition": -1,
      # "Conformity": -1,
      # "Security": -1,
    },
    # "Universalism": {
    #   "Power": -1,
    #   "Achievement": -1,
    #   "Hedonism": -1,     # TODO
    # },
    "Benevolence": {
      # "Power": -1,
      # "Achievement": -1,
      # "Hedonism": -1,     # TODO
    },
    # "Tradition": {
    #   "Power": -1,     # TODO
    #   "Achievement": -1,     # TODO
    #   "Hedonism": -1,
    #   "Stimulation": -1,
    #   "Self-direction": -1,
    # },
    # "Conformity": {
    #   "Hedonism": -1,
    #   "Stimulation": -1,
    #   "Self-direction": -1,
    # },
    # "Security": {
    #   "Stimulation": -1,
    #   "Self-direction": -1,
    # },
  }

  # for clarity purposes, using separate matrices for negative and positive interactions
  between_agents_positive_interaction_matrix_dict = {
    "Power": {
      # "Achievement": 1,
      # "Security": 1,
    },
    # "Achievement": {
    #   "Power": 1,
    #   "Hedonism": 1,
    # },
    # "Hedonism": {
    #   "Achievement": 1,
    #   "Stimulation": 1,
    # }, 
    # "Stimulation": {
    #   "Hedonism": 1,
    #   "Self-direction": 1,
    # },
    "Self-direction": {
      # "Stimulation": 1,
      # "Universalism": 1,
    },
    # "Universalism": {
    #   "Self-direction": 1,
    #   "Benevolence": 1,
    # },
    "Benevolence": {
      # "Universalism": 1,      
      "Benevolence": 0.5,     # the other agent might become more benevolent, but not necessarily
    },
    # "Tradition": {
    #   "Conformity": 1,
    # },
    # "Conformity": {
    #   "Tradition": 1,
    #   "Security": 1,
    # },
    # "Security": {
    #   "Power": 1,
    #   "Conformity": 1,
    # },
  }

  # for clarity purposes, using separate matrices for negative and positive interactions
  self_feedback_negative_interaction_matrix_dict = {
    "Power": {
      # "Universalism": -1,
      "Self-direction": 0.5,    # the agent might gain self-direction, but not necessarily
      "Benevolence": -1,
      # "Tradition": -1,  # TODO
    },
    # "Achievement": {
    #   "Universalism": -1,
    #   "Benevolence": -1,
    #   "Tradition": -1,  # TODO
    # },
    # "Hedonism": {
    #   "Universalism": -1,     # TODO
    #   "Benevolence": -1,    # TODO
    #   "Tradition": -1,
    #   "Conformity": -1,
    # }, 
    # "Stimulation": {
    #   "Tradition": -1,
    #   "Conformity": -1,
    #   "Security": -1,
    # },
    "Self-direction": {
      "Power": 0.5,    # the agent might gain power, but not necessarily
      # "Tradition": -1,
      # "Conformity": -1,
      # "Security": -1,
    },
    # "Universalism": {
    #   "Power": -1,
    #   "Achievement": -1,
    #   "Hedonism": -1,     # TODO
    # },
    "Benevolence": {
      "Power": -1,
      # "Achievement": -1,
      # "Hedonism": -1,     # TODO
    },
    # "Tradition": {
    #   "Power": -1,     # TODO
    #   "Achievement": -1,     # TODO
    #   "Hedonism": -1,
    #   "Stimulation": -1,
    #   "Self-direction": -1,
    # },
    # "Conformity": {
    #   "Hedonism": -1,
    #   "Stimulation": -1,
    #   "Self-direction": -1,
    # },
    # "Security": {
    #   "Stimulation": -1,
    #   "Self-direction": -1,
    # },
  }

  # for clarity purposes, using separate matrices for negative and positive interactions
  self_feedback_positive_interaction_matrix_dict = {
    "Power": {
      # "Achievement": 1,
      # "Security": 1,
    },
    # "Achievement": {
    #   "Power": 1,
    #   "Hedonism": 1,
    # },
    # "Hedonism": {
    #   "Achievement": 1,
    #   "Stimulation": 1,
    # }, 
    # "Stimulation": {
    #   "Hedonism": 1,
    #   "Self-direction": 1,
    # },
    "Self-direction": {
      # "Stimulation": 1,
      # "Universalism": 1,
    },
    # "Universalism": {
    #   "Self-direction": 1,
    #   "Benevolence": 1,
    # },
    "Benevolence": {
      # "Universalism": 1,
    },
    # "Tradition": {
    #   "Conformity": 1,
    # },
    # "Conformity": {
    #   "Tradition": 1,
    #   "Security": 1,
    # },
    # "Security": {
    #   "Power": 1,
    #   "Conformity": 1,
    # },
  }


  # parameters

  agent_names = [
    "alice",
    "bob",
  ]

  experiment_length = 1000
  value_interaction_rate = 0.025    # the system becomes unstable and the self-direction and power of one human goes to negative range when value interaction rate is above 0.025. At 0.025 the system is quite sensitive to "luck" of one or other human and initial luck will cause large differences later which are difficult to overcome. Even though the chances of getting helped by the agent are statistically equal between both humans, the specific order events of them getting helped is very important. In conclusion, having equal chances of support is not sufficient - the support needs to be timed very precisely.
  restrict_negative_interactions = True

  max_rebalancing_step_size = 0.1

  num_value_names = len(value_names)
  initial_actual_values = np.ones([num_value_names])
  target_values = 50 * np.ones([num_value_names])  # used only by homeostasis and by rebalancing agent
  homeostatic_utility_scenario_actual_values = target_values - 10  # NB! in case of homeostatic utilities, the initial values cannot be too far off targets, else the system never recovers


  random.seed(0)    # lets make the random number sequences used for rebalanced person selection reproducible


  # utility function mode and rebalancing mode

  # main(utility_function_mode="linear", rebalancing_mode="none")   # all values go down towards the end
  # main(utility_function_mode="linear", rebalancing_mode="homeostatic_boosting")   # all values go down towards the end
  # main(utility_function_mode="linear", rebalancing_mode="homeostatic")   # all values go down towards the end

  # main(utility_function_mode="sigmoid", rebalancing_mode="none")  # Mainly hedonism, achievement, and also to lesser extent power, and stimulation start to dominate here. This applies both to human and agent.
  # main(utility_function_mode="sigmoid", rebalancing_mode="homeostatic_boosting")  # Human values go slightly up, then slightly down again. Agent values go down.
  main(utility_function_mode="sigmoid", rebalancing_mode="homeostatic")  # Human values go slightly up, then slightly down again. Agent values go down.

  # TODO: it is possible that my prospect theory implementation is incorrect.
  # main(utility_function_mode="prospect_theory", rebalancing_mode="none")  # the value level stay horizontal
  # main(utility_function_mode="prospect_theory", rebalancing_mode="homeostatic_boosting")  # human values go up to target, agent values stay horisontal
  # main(utility_function_mode="prospect_theory", rebalancing_mode="homeostatic")  # human values go up to target, agent values stay horisontal

  # main(utility_function_mode="concave", rebalancing_mode="none")  # all goes to minus infinity
  # main(utility_function_mode="concave", rebalancing_mode="homeostatic_boosting")   # human values go up to target, agent values go down
  # main(utility_function_mode="concave", rebalancing_mode="homeostatic")   # human values go up to target, agent values go down

  # main(utility_function_mode="linear_homeostasis", rebalancing_mode="none")   # everything goes down
  # main(utility_function_mode="linear_homeostasis", rebalancing_mode="homeostatic_boosting")   # This quite interesting plot. The progress goes towards target value BUT then starts dropping for some reason.
  # main(utility_function_mode="linear_homeostasis", rebalancing_mode="homeostatic")   # This quite interesting plot. The progress goes towards target value BUT then starts dropping for some reason.

  # main(utility_function_mode="squared_homeostasis", rebalancing_mode="none")  # everything goes to minus infinity
  # main(utility_function_mode="squared_homeostasis", rebalancing_mode="homeostatic_boosting")   # everything goes to minus infinity
  # main(utility_function_mode="squared_homeostasis", rebalancing_mode="homeostatic")  # everything goes to minus infinity

#/ if __name__ == "__main__":


