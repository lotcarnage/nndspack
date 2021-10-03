# はじめに
本パッケージはニューラルネットワークの学習データセットをパッキングした形でファイルに保存し、
学習時にストリーミング読み込みするためのモジュールパッケージです。

ニューラルネットワークの学習データセットは膨大なサイズになりやすく、
メインメモリやVRAMに乗り切らないことが多いです。
そのような膨大なサイズの学習データセットを用いて学習を行うために、
本パッケージでは学習データセットを独自のバイナリフォーマットで保存し、それをストリーミング読み込みするための機能を提供します。

# 設計思想
学習データセットをファイルストリーミング化するにあたり、
学習データ一つ一つをファイルに分割して保存してしまうと、
OSのファイルシステム上でデータを扱う際のシステム負荷が非常に大きくなってしまい効率的に作業できません。

また、CSVのような人間が読めるテキスト形式では、データ量が無駄に大きくなってしまいます。

本パッケージでは、これらの問題を回避するためにデータセットを一つのバイナリファイルにパッキングして保存します。
パッキングしたデータはプリミティブにはインデックス指定で読み込む事ができます。

また、本パッケージはデータセットの型としてnumpy.ndarrayを想定しています。
データの保存および読み込みにおいてnumpy.ndarray型を使うため、本パッケージはnumpyに依存しています。

# インストール方法

```
pip install nndspack
```

# 基本的な使い方

モジュールは主に二つのクラスを提供します。

- Packer
- Loader

## Packer
Packerはパックファイルを作るためのクラスです。
新しく作成するパックファイル名と、入力データ、教師データそれぞれのサンプルを指定してインスタンスを生成します。
インスタンスのpack()メソッドで入力データと教師データのペアをパックする事ができます。
インスタンス破棄と同時にパックファイルをクローズします。

```
import nndspack
import keras
from keras.datasets import mnist

# MNISTデータを読込む
(x_train, y_train), (x_test, y_test) = mnist.load_data()

packer = nndspack.Packer('test.dat', x_train[0], y_train[0])
packer.pack(x_train[0], y_train[0])
packer.pack(x_train[1], y_train[1])
packer.pack(x_train[2], y_train[2])
del packer
```

## Loader
Loaderはパックファイルをオープンし、データのペアをインデックス指定で読み込むクラスです。
パックファイル名を指定してインスタンスを生成します。
データセットのフォーマットはパックファイル内に記録されているため、Loader側で指定する必要はありません。

```
import nndspack
import matplotlib.pyplot as plt

loader = nndspack.Loader('test.dat')
x_datum, y_datum = packer.load(0)
del loader

print(y_datum)
plt.imshow(x_datum)
plt.show()
plt.close()
```

## 注意点
PackerとLoaderは同一ファイルに対して同時に読み書きする事を想定していません。
Packerでパックファイルを作成しきったあとで、そのパックファイルをLoaderで使い回す事を想定しています。

