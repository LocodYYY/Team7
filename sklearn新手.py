from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载iris数据集
iris = datasets.load_iris()
iris_X = iris.data  # 特征数据
iris_Y = iris.target  # 标签数据

# 打印数据的形状
print("特征数据的形状（iris_X.shape）：", iris_X.shape)  # 输出特征数据的形状：(150, 4)，表示有150个样本，每个样本有4个特征
print("标签数据的形状（iris_Y.shape）：", iris_Y.shape)  # 输出标签数据的形状：(150,)，表示有150个标签

# 打印特征数据和标签数据
print("特征数据（iris_X）：")  # 打印特征数据：每个样本的4个特征值
print(iris_X)
print("标签数据（iris_Y）：")  # 打印标签数据：每个样本对应的类别标签（0、1、2）
print(iris_Y)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=0.3, random_state=42)

# 打印训练集和测试集的特征数据和标签数据
print("训练集特征数据（X_train）：")  # 打印训练集特征数据：大约105个样本（70%）
print(X_train)
print("测试集特征数据（X_test）：")  # 打印测试集特征数据：大约45个样本（30%）
print(X_test)
print("训练集标签数据（Y_train）：")  # 打印训练集标签数据：训练集对应的标签
print(Y_train)
print("测试集标签数据（Y_test）：")  # 打印测试集标签数据：测试集对应的标签
print(Y_test)

# 定义KNN模型，默认使用K=5（邻居数）
knn = KNeighborsClassifier()
# 训练模型：使用训练集特征和标签训练KNN模型
knn.fit(X_train, Y_train)

# 使用训练好的模型对测试集进行预测
print("模型对测试集的预测结果（knn.predict(X_test)）：")  # 打印预测结果：模型预测的测试集标签
print(knn.predict(X_test))
print("测试集的真实标签（Y_test）：")  # 打印真实结果：测试集的实际标签
print(Y_test)