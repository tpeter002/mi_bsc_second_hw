import numpy as np

######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:

    
    if n_cat1 == 0:
        return 0
    if n_cat2 == 0:
        return 0
    
    n_sum = n_cat1 + n_cat2

    p_cat1 = n_cat1 / n_sum
    p_cat2 = n_cat2 / n_sum
    
    return -p_cat1 * np.log2(p_cat1) - p_cat2 * np.log2(p_cat2)

def get_entropy_for_labels(labels: list) -> float:
    n_cat1 = np.sum(labels == 0)
    n_cat2 = np.sum(labels == 1)
    return get_entropy(n_cat1, n_cat2)

def information_gain(total_entropy: float,label_count: int, labels_left: list, labels_right: list) -> float:
    left_entropy = get_entropy_for_labels(labels_left)
    right_entropy = get_entropy_for_labels(labels_right)

    return total_entropy - (len(labels_left) / label_count) * left_entropy - (len(labels_right) / label_count) * right_entropy


###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(features: list,
                        labels: list) -> (int, int):
    best_separation_feature, best_separation_value = 0, 0
    best_info_gain = -np.inf
    total_entropy = get_entropy_for_labels(labels)
    label_count = len(labels)
 
    for feature in range(len(features[0])):
        
        for value in np.unique(features[:, feature]):
            labels_left = labels[features[:, feature] <= value]
            labels_right = labels[features[:, feature] > value]

            info_gain = information_gain(total_entropy,label_count, labels_left, labels_right)
            
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_separation_feature = feature
                best_separation_value = value

    return best_separation_feature, best_separation_value

class Node:
    def __init__(self, feature_index=None, value=None, true_branch=None, false_branch=None, gain=None, label=None):
        self.feature_index = feature_index
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.gain = gain
        self.label = label

def build_tree(features, labels):
    best_feature_index, best_value = get_best_separation(features, labels)

    if get_entropy_for_labels(labels) == 0.0:
        return Node(label=np.argmax(np.bincount(labels)))

    true_features = features[features[:, best_feature_index] <= best_value]
    true_labels = labels[features[:, best_feature_index] <= best_value]

    false_features = features[features[:, best_feature_index] > best_value]
    false_labels = labels[features[:, best_feature_index] > best_value]

    true_branch = build_tree(true_features, true_labels)
    false_branch = build_tree(false_features, false_labels)

    return Node(feature_index=best_feature_index, value=best_value, true_branch=true_branch, false_branch=false_branch)

################### 3. feladat, döntési fa implementációja ####################
def main():
    train_data = np.genfromtxt('train.csv', delimiter=',', dtype=int)
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    X_test = np.genfromtxt('test.csv', delimiter=',')
    dec_tree = build_tree(X_train, y_train)

    y_pred = [predict(dec_tree, sample) for sample in X_test]

    np.savetxt('results.csv', y_pred, fmt='%d', delimiter=',')

    return 0


def predict(node, instance):
    if node.label is not None:
        return node.label

    if instance[node.feature_index] <= node.value:
        return predict(node.true_branch, instance)
    else:
        return predict(node.false_branch, instance)
    
if __name__ == "__main__":
    main()
