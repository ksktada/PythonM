##################3
#### 画像データの表示

import matplotlib.pyplot as plt
from sklearn import datasets

# digitsデータをロード
digits = datasets.load_digits()

# 画像を 2行 5列に表示
for label, img in zip(digits.target[:10], digits.images[:10]):
	plt.subplot(2, 5, label + 1)
	plt.axis('off')
	plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
	plt.title('Digit: {0}'.format(label))

plt.show()


################
#### 分類

### データ準備

# 3と8のデータ位置を求める → 3か8のところがtrueになるlistができる
# ex) [3, 2, 4, 3, 8, 6] → [True, False, False, True, True, False]
flag_3_8 = (digits.target == 3) + (digits.target == 8)

# 3と8のデータを取得 → Boolの配列でTrueになってるとこだけ取ってこれる
images = digits.images[flag_3_8] #画像
labels = digits.target[flag_3_8] #解

# 3と8の画像データを1次元化（1次元でも2次元でも変わらないので計算しやすいように変換してる（らしい））
# invariant的な？
images = images.reshape(images.shape[0], -1)

### 分類器を生成
from sklearn import svm

n_samples = len(flag_3_8[flag_3_8])
train_size = int(n_samples * 3 / 5)
classifier = svm.SVC(C=1.0, gamma=0.001)
classifier.fit(images[:train_size], labels[:train_size])

### 性能評価
from sklearn import metrics

expected = labels[train_size:]
predicted = classifier.predict(images[train_size:])

print('Accuracy:\n', metrics.accuracy_score(expected, predicted))
print('\nConfusion matrix:\n', metrics.confusion_matrix(expected, predicted))
print('\nPrecision:\n', metrics.precision_score(expected, predicted, pos_label=3))
print('\nRecall:\n', metrics.recall_score(expected, predicted, pos_label=3))
print('\nf-measure:\n', metrics.f1_score(expected, predicted, pos_label=3))