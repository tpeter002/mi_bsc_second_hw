import numpy as np

######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:

    n_sum = n_cat1 + n_cat2
    if n_sum == 0:
        return 0

    p_cat1 = n_cat1 / n_sum
    p_cat2 = n_cat2 / n_sum
    entropy = 0
    if p_cat1 != 0:
        entropy -= p_cat1 * np.log2(p_cat1)
    if p_cat2 != 0:
        entropy -= p_cat2 * np.log2(p_cat2)

    return entropy

def get_entropy_for_labels(labels: list) -> float:
    n_cat1 = np.sum(labels == 0)
    n_cat2 = np.sum(labels == 1)
    return get_entropy(n_cat1, n_cat2)

def information_gain(labels: list, labels_left: list, labels_right: list) -> float:
    total_entropy = get_entropy_for_labels(labels)
    left_entropy = get_entropy_for_labels(labels_left)
    right_entropy = get_entropy_for_labels(labels_right)

    return total_entropy - (len(labels_left) / len(labels)) * left_entropy - (len(labels_right) / len(labels)) * right_entropy


###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(features: list,
                        labels: list) -> (int, int):
    best_separation_feature, best_separation_value = 0, 0
    best_info_gain = -np.inf

    for feature in range(len(features[0])):
        for value in np.unique(features[:, feature]):
            labels_left = labels[features[:, feature] <= value]
            labels_right = labels[features[:, feature] > value]

            info_gain = information_gain(labels, labels_left, labels_right)
            
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_separation_feature = feature
                best_separation_value = value

    return best_separation_feature, best_separation_value

################### 3. feladat, döntési fa implementációja ####################
def main():
    #TODO: implementálja a döntési fa tanulását!
    return 0

if __name__ == "__main__":
    main()
