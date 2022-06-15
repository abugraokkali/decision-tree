class DecisionTreeClassifier:
    
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.min_samples_split = 4

    def fit(self, X_train_list, y_train_list):
        train = []
        for i in range(len(X_train_list)):
            row = X_train_list[i]
            row.append(y_train_list[i])
            train.append(row)  
        root = self.evaluate_split(train)
        self.split(root, 1)
        self.root = root

    def predict(self, X_test_list):
        predictions = []
        for row in X_test_list:
            predictions.append(self.find(self.root, row))
        return predictions

    def impurity(self, groups, classes):
        impurity = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            impurity += (1.0 - score)
        return impurity

    def find_split(self, index, value, dataset):
        left = [] 
        right = []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def evaluate_split(self, data):
        class_values = list(set(row[-1] for row in data))
        b_index, b_value, b_score, b_groups = float('inf'), float('inf'), float('inf'), None
        for index in range(len(data[0])-1):
            for row in data:
                groups = self.find_split(index, row[index], data)
                impurity = self.impurity(groups, class_values)
                if impurity < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], impurity, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def makeLeaf(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.makeLeaf(left + right)
            return

        if self.max_depth and depth >= self.max_depth:
            node['left'], node['right'] = self.makeLeaf(left), self.makeLeaf(right)
            return

        if len(left) < self.min_samples_split:
            node['left'] = self.makeLeaf(left)
        else:
            node['left'] = self.evaluate_split(left)
            self.split(node['left'], depth+1)

        if len(right) < self.min_samples_split:
            node['right'] = self.makeLeaf(right)
        else:
            node['right'] = self.evaluate_split(right)
            self.split(node['right'], depth+1)

    def find(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.find(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.find(node['right'], row)
            else:
                return node['right']