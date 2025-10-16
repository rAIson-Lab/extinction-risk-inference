import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../CONFOLD/'))) #add CONFOLD to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) #add the parent directory to the path

import numpy as np
from foldrm import Classifier
from utils import split_data # Or your stratified version if you prefer
from datasets import new_extinction_birds # Our new function

# Load the data

# Cargar y filtrar los datos para usar solo 0 y 1 en extinction_risk
model_template, data = new_extinction_birds()
label_index = model_template.attrs.index(model_template.label) if model_template.label in model_template.attrs else -1
data = [row for row in data if str(row[label_index]) in ['Lower_risk', 'Higher_risk']]

# Split into training and testing sets
train_data, test_data = split_data(data, ratio=0.9, shuffle=True)

print(f"Training set size: {len(train_data)} newextinctionbirds")
print(f"Testing set size: {len(test_data)} newextinctionbirds")

# Instantiate a new classifier for our baseline experiment
baseline_model = Classifier(attrs=model_template.attrs, numeric=model_template.numeric, label=model_template.label)

# Fit the model on the training data
baseline_model.fit(train_data, ratio=0.5)

# Print the rules the model learned
print("--- Rules Learned by the Baseline Model ---")
baseline_model.print_asp(simple=True)

# Prepare the test data (features and true labels)
X_test = [d[:-1] for d in test_data]
Y_test = [d[-1] for d in test_data]

# Get predictions (these will be tuples of (label, confidence))
predictions_tuples = baseline_model.predict(X_test)
predicted_labels = [p[0] for p in predictions_tuples]

# Calculate accuracy
correct_predictions = 0
for i in range(len(Y_test)):
    if predicted_labels[i] == Y_test[i]:
            correct_predictions += 1

accuracy = correct_predictions / len(Y_test)

print("--- Baseline Model Evaluation ---")
print(f"True Labels:    {Y_test}")
print(f"Predicted Labels: {predicted_labels}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Instantiate a new classifier for our expert-guided model
expert_model = Classifier(attrs=model_template.attrs, numeric=model_template.numeric, label=model_template.label)

# Define our expert rules as strings
# Note: the symbols '==' and '<=' must also be in single quotes for the parser.
rule1 = "with confidence 0.90 class = 'Higher_risk' if 'Range_size' '<=' '5'"
#Note additional rules could be added like this:
#rule2 = "with confidence 0.70 class = '1' if 'Clutch_Max' '==' '1'"

# Add the manual rules to the model
expert_model.add_manual_rule(rule1, model_template.attrs, model_template.numeric, ['Lower_risk', 'Higher_risk'], instructions=False)
# Note: here is code to add an additional rule:
#expert_model.add_manual_rule(rule2, model_template.attrs, model_template.numeric, ['0', '1'], instructions=False)

print("--- Manual Rules Added to the Model (Before Training) ---")
# The internal representation is a bit complex, but we can see our rules are in there.
for rule in expert_model.rules:
    print(rule)

# Now, fit the model on the training data.
# The algorithm will work around the rules we provided.
expert_model.fit(train_data, ratio=0.75)

# Print the final, combined rule set
print("--- Final Ruleset from the Expert Model ---")
expert_model.print_asp(simple=True)

# Get predictions from our new model
expert_predictions_tuples = expert_model.predict(X_test)
expert_predicted_labels = [p[0] for p in expert_predictions_tuples]

# Calculate accuracy
expert_correct_predictions = 0
for i in range(len(Y_test)):
    if expert_predicted_labels[i] == Y_test[i]:
        expert_correct_predictions += 1

expert_accuracy = expert_correct_predictions / len(Y_test)

print("--- Baseline Model Evaluation ---")
print(f"True Labels:      {Y_test}")
print(f"Predicted Labels: {predicted_labels}")
print(f"Accuracy: {accuracy * 100:.2f}%\n")


print("--- Expert Model Evaluation ---")
print(f"True Labels:      {Y_test}")
print(f"Predicted Labels: {expert_predicted_labels}")
print(f"Accuracy: {expert_accuracy * 100:.2f}%")

# Instantiate a new classifier
learned_confidence_model = Classifier(attrs=model_template.attrs, numeric=model_template.numeric, label=model_template.label)

# Define our expert rules as strings, but WITHOUT the 'with confidence' part.
rule1_no_confidence = "class = 'Higher_risk' if 'Range_size' '<=' '5'"
rule2_no_confidence = "class = 'Higher_risk' if 'Clutch_size' '<=' '1'"

# Add the manual rules to the model
learned_confidence_model.add_manual_rule(rule1_no_confidence, model_template.attrs, model_template.numeric, ['Lower_risk', 'Higher_risk'], instructions=False)
learned_confidence_model.add_manual_rule(rule2_no_confidence, model_template.attrs, model_template.numeric, ['Lower_risk', 'Higher_risk'], instructions=False)

print("--- Manual Rules Added (Before Training) ---")
print("Notice the default confidence value of 0.5 assigned to each rule.")
for rule in learned_confidence_model.rules:
    print(rule)

# Now, fit the model on the training data.
# The algorithm will calculate the confidence of our provided rules and then learn any additional rules needed.
learned_confidence_model.fit(train_data, ratio=0.5)

# Print the final, combined rule set
print("--- Final Ruleset with Learned Confidence ---")
print("The confidence values have now been updated based on the training data!")
learned_confidence_model.print_asp(simple=True)
            #Note that confidence values will be relatively low due to the small size of the training data. 

# Get predictions from our new model
learned_conf_predictions = learned_confidence_model.predict(X_test)
learned_conf_labels = [p[0] for p in learned_conf_predictions]

# Calculate accuracy
learned_conf_accuracy = sum(1 for i in range(len(Y_test)) if learned_conf_labels[i] == Y_test[i]) / len(Y_test)

print("--- Learned Confidence Model Evaluation ---")
print(f"True Labels:      {Y_test}")
print(f"Predicted Labels: {learned_conf_labels}")
print(f"Accuracy: {learned_conf_accuracy * 100:.2f}%")

# First, let's re-print the rules from our expert model for comparison
print("--- Rules Before Pruning ---")
expert_model.print_asp(simple=True)

############PRUNNING##################

# Method 1: Simple Post-Hoc Confidence Pruning: removing those rules with a low confidence according to me
# Import the prune_rules function from the core algorithm file
from algo import prune_rules

# Apply the pruning function
# This will create a new list containing only the rules that meet the confidence threshold.
pruned_rules = prune_rules(expert_model.rules, confidence=0.90)

# We can create a new model instance to hold these pruned rules
simple_pruned_model = Classifier(attrs=model_template.attrs, numeric=model_template.numeric, label=model_template.label)
simple_pruned_model.rules = pruned_rules

print("\n--- Rules After Pruning (Confidence >= 0.90) ---")
simple_pruned_model.print_asp(simple=True)
            
# Instantiate a new model for this experiment
advanced_pruning_model = Classifier(attrs=model_template.attrs, numeric=model_template.numeric, label=model_template.label)

##################
#### Method 2: Advanced Confidence-Driven Learning

# Now, train using confidence_fit with a high 15% improvement threshold
print("--- Training with confidence_fit(improvement_threshold=0.15) ---")
advanced_pruning_model.confidence_fit(train_data, improvement_threshold=0.15)

print("\n--- Rules Learned via Confidence-Driven Learning ---")
print("Note how the model is simpler and did not learn any exceptions to rules or `abnormalities', as they did not meet the high confidence improvement threshold.")
advanced_pruning_model.print_asp(simple=True)