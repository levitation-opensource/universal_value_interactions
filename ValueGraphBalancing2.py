
import os
import numpy as np
from matplotlib import pyplot as plt
import yaml


def init():

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
  result = np.random.choice(max_values_indexes)  # TODO: seed for this random generator
  return result


def plot_history(values_history, utilities_history, utility_function_mode, rebalancing_mode):

  fig, subplots = plt.subplots(4)

  linewidth = 0.75  # TODO: config


  subplot = subplots[0]
  for index, value_name in enumerate(value_names):
    subplot.plot(
      values_history[:, index],
      label=value_name,
      linewidth=linewidth,
    )

  subplot.set_title(f"Value graph balancing - Value level evolution - {utility_function_mode} - {rebalancing_mode}")
  subplot.set(xlabel="step", ylabel="value strength")
  subplot.legend()


  subplot = subplots[1]
  for index, value_name in enumerate(value_names):
    subplot.plot(
      custom_sigmoid10(values_history[:, index]),
      label=value_name,
      linewidth=linewidth,
    )

  subplot.set_title(f"Value graph balancing - Sigmoid10 of Value level evolution - {utility_function_mode} - {rebalancing_mode}")
  subplot.set(xlabel="step", ylabel="custom_sigmoid10(value strength)")
  subplot.legend()


  subplot = subplots[2]
  for index, value_name in enumerate(value_names):
    subplot.plot(
      utilities_history[:, index],
      label=value_name,
      linewidth=linewidth,
    )

  subplot.set_title(f"Value graph balancing - Utilities evolution - {utility_function_mode} - {rebalancing_mode}")
  subplot.set(xlabel="step", ylabel="value strength")
  subplot.legend()


  subplot = subplots[3]
  for index, value_name in enumerate(value_names):
    subplot.plot(
      custom_sigmoid10(utilities_history[:, index]),
      label=value_name,
      linewidth=linewidth,
    )

  subplot.set_title(f"Value graph balancing - Sigmoid10 of Utilities evolution - {utility_function_mode} - {rebalancing_mode}")
  subplot.set(xlabel="step", ylabel="custom_sigmoid10(value strength)")
  subplot.legend()


  # TODO: std or gini index over values per timestep plot


  plt.ion()
  # maximise_plot()
  fig.show()
  plt.draw()
  # TODO: use multithreading for rendering the plot
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

  interaction_matrix, positive_interaction_matrix, negative_interaction_matrix = init()

  prev_actual_values = np.zeros([num_value_names])
  prev_utilities = np.zeros([num_value_names])

  if utility_function_mode == "linear_homeostasis" or utility_function_mode == "squared_homeostasis":  # NB! in case of homeostatic utilities, the initial values cannot be too far off targets, else the system never recovers
    actual_values = homeostatic_utility_scenario_actual_values
  else:
    actual_values = initial_actual_values
  utilities = compute_utilities(prev_actual_values, actual_values, prev_utilities, utility_function_mode)

  values_history = np.zeros([experiment_length, num_value_names])
  utilities_history = np.zeros([experiment_length, num_value_names])

  for step in range(0, experiment_length):

    # NB! the raw value level changes are computed based on interactions with utilities, not on interactions between raw value levels
    if not restrict_negative_interactions:
      utility_changes = np.matmul(utilities, interaction_matrix) * change_rate
    else:
      positive_interaction_value_changes = np.matmul(utilities, positive_interaction_matrix) * change_rate
      negative_interaction_value_changes = np.matmul(np.maximum(utilities, 0), negative_interaction_matrix) * change_rate   # np.maximum: in case of negative interactions, ignore negative actual values
      value_changes = positive_interaction_value_changes + negative_interaction_value_changes


    updated_actual_values = actual_values + value_changes
    utilities = compute_utilities(actual_values, updated_actual_values, utilities, utility_function_mode)
    actual_values = updated_actual_values

    # TODO: option to require removal or addition of resources to some other value when current most extreme value is adjusted, so that the sum total remains same


    rebalanced_actual_values = actual_values.copy()

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


    values_history[step, :] = actual_values
    utilities_history[step, :] = utilities

    actual_values_with_names_dict = {
      value_name: "{:.3f}".format(actual_values[index]) 
      for index, value_name in enumerate(value_names)
    } 
    actual_utilities_with_names_dict = {
      value_name: "{:.3f}".format(utilities[index]) 
      for index, value_name in enumerate(value_names)
    } 

    print("Value levels:")
    prettyprint(actual_values_with_names_dict)
    print("Utilities:")
    prettyprint(actual_utilities_with_names_dict)
    print()
    print()

  #/ for step in range(0, experiment_length):


  plot_history(values_history, utilities_history, utility_function_mode, rebalancing_mode)

#/ def main():


if __name__ == "__main__":

  # values and interaction matrices

  value_names = [
    "Power",
    "Achievement",
    "Hedonism",  
    "Stimulation",
    "Self-direction",
    "Universalism",
    "Benevolence",
    "Tradition",
    "Conformity",
    "Security",
  ]

  # for clarity purposes, using separate matrices for negative and positive interactions
  negative_interaction_matrix_dict = {
    "Power": {
      "Universalism": -1,
      "Benevolence": -1,
      "Tradition": -1,  # TODO
    },
    "Achievement": {
      "Universalism": -1,
      "Benevolence": -1,
      "Tradition": -1,  # TODO
    },
    "Hedonism": {
      "Universalism": -1,     # TODO
      "Benevolence": -1,    # TODO
      "Tradition": -1,
      "Conformity": -1,
    }, 
    "Stimulation": {
      "Tradition": -1,
      "Conformity": -1,
      "Security": -1,
    },
    "Self-direction": {
      "Tradition": -1,
      "Conformity": -1,
      "Security": -1,
    },
    "Universalism": {
      "Power": -1,
      "Achievement": -1,
      "Hedonism": -1,     # TODO
    },
    "Benevolence": {
      "Power": -1,
      "Achievement": -1,
      "Hedonism": -1,     # TODO
    },
    "Tradition": {
      "Power": -1,     # TODO
      "Achievement": -1,     # TODO
      "Hedonism": -1,
      "Stimulation": -1,
      "Self-direction": -1,
    },
    "Conformity": {
      "Hedonism": -1,
      "Stimulation": -1,
      "Self-direction": -1,
    },
    "Security": {
      "Stimulation": -1,
      "Self-direction": -1,
    },
  }

  # for clarity purposes, using separate matrices for negative and positive interactions
  positive_interaction_matrix_dict = {
    "Power": {
      "Achievement": 1,
      "Security": 1,
    },
    "Achievement": {
      "Power": 1,
      "Hedonism": 1,
    },
    "Hedonism": {
      "Achievement": 1,
      "Stimulation": 1,
    }, 
    "Stimulation": {
      "Hedonism": 1,
      "Self-direction": 1,
    },
    "Self-direction": {
      "Stimulation": 1,
      "Universalism": 1,
    },
    "Universalism": {
      "Self-direction": 1,
      "Benevolence": 1,
    },
    "Benevolence": {
      "Universalism": 1,
    },
    "Tradition": {
      "Conformity": 1,
    },
    "Conformity": {
      "Tradition": 1,
      "Security": 1,
    },
    "Security": {
      "Power": 1,
      "Conformity": 1,
    },
  }


  # parameters

  experiment_length = 1000
  change_rate = 0.01
  restrict_negative_interactions = True

  max_rebalancing_step_size = 1.0

  num_value_names = len(value_names)
  initial_actual_values = np.ones([num_value_names])
  target_values = 100 * np.ones([num_value_names])  # used only by homeostasis and by rebalancing agent
  homeostatic_utility_scenario_actual_values = target_values - 10


  # utility function mode and rebalancing mode

  # main(utility_function_mode="linear", rebalancing_mode="none")   # everything goes to negative domain - TODO: why isnt power and achievement being maximised in the positive domain - is it because of restrict_negative_interactions flag?
  # main(utility_function_mode="linear", rebalancing_mode="homeostatic_boosting")   # the progress flattens at about 10 units of each value
  # main(utility_function_mode="linear", rebalancing_mode="homeostatic")   # the progress flattens at about 10 units of each value

  # main(utility_function_mode="sigmoid", rebalancing_mode="none")    # hedonism, achievement, and power start to dominate here
  # main(utility_function_mode="sigmoid", rebalancing_mode="homeostatic_boosting")    # the progress does not flatten
  # main(utility_function_mode="sigmoid", rebalancing_mode="homeostatic")    # the progress does not flatten

  # TODO: it is possible that my prospect theory implementation is incorrect.
  # main(utility_function_mode="prospect_theory", rebalancing_mode="none")    # the progress flattens at about 1 units of each value
  # main(utility_function_mode="prospect_theory", rebalancing_mode="homeostatic_boosting")    # the progress does not flatten
  # main(utility_function_mode="prospect_theory", rebalancing_mode="homeostatic")    # the progress does not flatten

  # main(utility_function_mode="concave", rebalancing_mode="none")    # hedonism, achievement, self-direction and stimulation start to dominate here, the rest goes to negative infinity
  # main(utility_function_mode="concave", rebalancing_mode="homeostatic_boosting")    # the progress does not flatten
  # main(utility_function_mode="concave", rebalancing_mode="homeostatic")   # the progress does not flatten

  # main(utility_function_mode="linear_homeostasis", rebalancing_mode="none")   # everything goes down
  # main(utility_function_mode="linear_homeostasis", rebalancing_mode="homeostatic_boosting")   # the progress goes to target value and stays there as desired
  # main(utility_function_mode="linear_homeostasis", rebalancing_mode="homeostatic")   # the progress goes to target value and stays there as desired

  # main(utility_function_mode="squared_homeostasis", rebalancing_mode="none")   # everything goes to minus infinity
  # main(utility_function_mode="squared_homeostasis", rebalancing_mode="homeostatic_boosting")   # the progress goes to target value and stays there as desired
  main(utility_function_mode="squared_homeostasis", rebalancing_mode="homeostatic")  # the progress goes to target value and stays there as desired

#/ if __name__ == "__main__":


