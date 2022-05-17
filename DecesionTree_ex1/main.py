from pprint import pprint
import pandas as pd
import numpy as np

dir = "data/decisionTree.xlsx"
data = pd.read_excel(dir)

print("___________DATA SET___________")
print(data)

features = ['district', 'house type', 'income', 'previous customer']
target = ['outcome']


# entropy - calculate entropy selected column
def entropy(target_col):
    element, count = np.unique(target_col, return_counts=True)

    # entropy_root = - (probability of a hit-Yes among all novels * log(probability of a hit-Yes among all novels))
    # + (probability of a hit-No + log(probability of a hit-No among all novels)))

    # Calculate entropy
    entropy = np.sum([(-(count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count))) for i in range(len(element))])

    return entropy


# infoGain - calculate information gain selected column
def infoGain(data, attribute_name, target_name):
    # infomation_Gain = entropy_root – averge(Σweight * child entropy)

    # Calculate total entropy
    entropy_root = entropy(data[target_name])

    # Calculate weighted entropy
    element, count = np.unique(data[attribute_name], return_counts=True)
    entropy_average = np.sum(
        [(count[i] / np.sum(count)) * entropy(data.where(data[attribute_name] == element[i]).dropna()[target_name])
         for i in range(len(element))])

    # Calculate information gain
    infomation_Gain = entropy_root - entropy_average

    # Return information gain
    return infomation_Gain


# ID3 - Make decision tree
def ID3(data, originaldata, features, target_attribute_name, parent_node_class=None):
    # When target attribute has only single value then return the target attribute
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # When there are no data then return target attribute with maximum value in original data
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    # When there are no feature then return parent node's attribute
    elif len(features) == 0:
        return parent_node_class

    else:
        # Define target attribute for the parent node
        parent_node_class = np.unique(data[target_attribute_name]) \
            [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # Select attribute to split data
        item_values = [infoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create tree structure
        tree = {best_feature: {}}

        features = [i for i in features if i != best_feature]

        # Make a branch for each value of the root node attribute
        for value in np.unique(data[best_feature]):
            # create sub tree
            # Split data
            sub_data = data.where(data[best_feature] == value).dropna()

            # Recursively ID3 function call
            subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class)

            # Add sub tree
            tree[best_feature][value] = subtree

        # Return Decision Tree
        return tree


tree = ID3(data, data, features, target)

# Print Decision Tree
print("___________Decision Tree___________")
pprint(tree)

