""" Neural Network Dataset Pack

ニューラルネットワークの学習等で扱うデータセットをファイルに保存し、
ランダムアクセスで読み込んでくるための機能を提供するモジュールです。
"""
import os
import struct
import numpy


_TYPE_TO_CODE_DICT = {
    numpy.int8: 0, numpy.int16: 1, numpy.int32: 2, numpy.int64: 3,
    numpy.uint8: 4, numpy.uint16: 5, numpy.uint32: 6, numpy.uint64: 7,
    numpy.float16: 8, numpy.float32: 9, numpy.float64: 10,
    numpy.bool_: 11
}

_CODE_TO_TYPE_DICT = {
    0: numpy.int8, 1: numpy.int16, 2: numpy.int32, 3: numpy.int64,
    4: numpy.uint8, 5: numpy.uint16, 6: numpy.uint32, 7: numpy.uint64,
    8: numpy.float16, 9: numpy.float32, 10: numpy.float64,
    11: numpy.bool_
}

_TYPE_TO_NBYTES_DICT = {
    numpy.int8: 1, numpy.int16: 2, numpy.int32: 4, numpy.int64: 8,
    numpy.uint8: 1, numpy.uint16: 2, numpy.uint32: 4, numpy.uint64: 8,
    numpy.float16: 2, numpy.float32: 4, numpy.float64: 8,
    numpy.bool_: 1
}


_TYPE_TO_STRUCT_FORMAT = {
    numpy.int8: 'b', numpy.int16: 'h', numpy.int32: 'i', numpy.int64: 'q',
    numpy.uint8: 'B', numpy.uint16: 'H', numpy.uint32: 'I', numpy.uint64: 'Q',
    numpy.float16: 'e', numpy.float32: 'f', numpy.float64: 'd',
    numpy.bool_: '?'
}

_ACCEPTABLE_DATA_TYPES = [
    numpy.ndarray,
    numpy.int8, numpy.uint8,
    numpy.int16, numpy.uint16,
    numpy.int32, numpy.uint32,
    numpy.int64, numpy.uint64,
    numpy.float16, numpy.float32, numpy.float64,
    numpy.bool_
]


def _make_data_header(datum_sample):
    if type(datum_sample) not in _ACCEPTABLE_DATA_TYPES:
        return None
    element_type_code = _TYPE_TO_CODE_DICT[datum_sample.dtype.type]
    if type(datum_sample) == numpy.ndarray:
        dim = len(datum_sample.shape)
        shape = datum_sample.shape
    else:
        dim = 1
        shape = (1,)
    data_format = f'HH{dim}I'
    return struct.pack(data_format, element_type_code, dim, *shape)


def _read_data_header(fp):
    element_type_code, dim = struct.unpack('HH', fp.read(4))
    element_type = _CODE_TO_TYPE_DICT[element_type_code]
    shape_format = f'{dim}I'
    shape = tuple(struct.unpack(shape_format, fp.read(dim * 4)))
    return {'element_type': element_type, 'shape': shape}


def _make_header(total_count, sample_data_record):
    num_columns = len(sample_data_record)
    data_headers = bytes().join([_make_data_header(datum_sample) for datum_sample in sample_data_record])
    header_format = f'II{len(data_headers)}B'
    return struct.pack(header_format, total_count, num_columns, *data_headers)


def _read_header(fp):
    total_count, num_columns = struct.unpack('II', fp.read(8))
    data_headers = [_read_data_header(fp) for _ in range(num_columns)]
    header_size = fp.tell()
    return header_size, total_count, data_headers


def _make_column_format(column):
    num_elements = numpy.prod(column.shape) if column.shape != () else 1
    struct_format = _TYPE_TO_STRUCT_FORMAT[column.dtype.type]
    return num_elements, f'{num_elements}{struct_format}'


def _make_column_info(data_header):
    shape = data_header['shape']
    element_type = data_header['element_type']
    num_elements = int(numpy.prod(shape))
    struct_format = _TYPE_TO_STRUCT_FORMAT[element_type]
    byte_length = num_elements * _TYPE_TO_NBYTES_DICT[element_type]
    is_scalar = (len(shape) == 1 and shape[0] == 1)
    return {
        'element_type': element_type,
        'shape': shape,
        'num_elements': num_elements,
        'struct_format': f'{num_elements}{struct_format}',
        'byte_length': byte_length,
        'is_scalar': is_scalar
    }


class Packer:
    """ Packerクラス

    ファイルに対してデータレコードを順次書き込む機能を提供します。
    """
    def __init__(self, packfile_path: str, *sample_data_record):
        """コンストラクタ

        パックファイル名とパック対象のデータ構造をサンプルで指定してパックファイルを作成します。
        既存ファイルが存在する場合は内容を消去して新規に作成します。
        そのため、既存ファイルに対する追記はできません。

        :arg str packfile_path: 作成するパックファイル名
        :arg * sample_data_record: パック対象のデータフォーマットを指定するためのデータサンプル(可変長引数)
        """

        if type(sample_data_record) != tuple:
            raise Exception('type error')
        for datum_sample in sample_data_record:
            if type(datum_sample) not in _ACCEPTABLE_DATA_TYPES:
                raise Exception('type error')
        self.__fp = open(packfile_path, 'wb')
        self.__total_count = 0
        self.__column_formats = [_make_column_format(column) for column in sample_data_record]
        self.__fp.write(_make_header(self.__total_count, sample_data_record))
        self.__num_colmuns = len(sample_data_record)

    def __del__(self):
        self.__fp.seek(0, os.SEEK_SET)
        self.__fp.write(struct.pack('I', self.__total_count))
        self.__fp.close()

    def pack(self, *data_record):
        """ 指定したデータレコードをパックします

        任意長引数で指定したデータレコードをパックファイルに対して追記で保存します。

        :arg * data_record: パック対象のデータレコード(コンストラクタで指定したサンプルと同じ構造でなければならない)
        """
        if self.__num_colmuns != len(data_record):
            raise Exception('カラム数が合っていません。')
        for column, expected_column_format in zip(data_record, self.__column_formats):
            is_array = type(column) == numpy.ndarray
            writable_values = column.reshape(expected_column_format[0]).tolist() if is_array else (column, )
            self.__fp.write(struct.pack(expected_column_format[1], *writable_values))
        self.__total_count += 1
        return None


class Loader:
    """ Packerクラス

    パックファイルに対して指定したインデックス値のデータレコードを読み込む機能を提供します。
    """
    def __init__(self, packfile_path: str):
        """ コンストラクタ

        パックファイル名を指定して既存のパックファイルを開きます。
        データ構造はパックファイル内に記録されているデータ構造になります。
        開いたパックファイルに対してはランダムアクセスでデータレコードを読み込むことができます。

        :arg str packfile_path: 開くパックファイル名
        """

        self.__fp = open(packfile_path, 'rb')
        header_size, total_count, data_headers = _read_header(self.__fp)
        self.__header_size = header_size
        self.__total_count = total_count
        self.__column_info = [_make_column_info(data_header) for data_header in data_headers]
        self.__block_size = sum([colmun_info['byte_length'] for colmun_info in self.__column_info])
        self.__block_format = ''.join([colmun_info['struct_format'] for colmun_info in self.__column_info])

    def __del__(self):
        self.__fp.close()

    def count(self):
        """パックファイルが含むレコード数"""
        return self.__total_count

    def __len__(self):
        return self.count()

    def load(self, index: int):
        """パックファイルから指定したインデックスのデータレコードを読み込む

        指定したインデックスのデータレコードを読み込みます。

        arg: int index: 読み込み対象のデータレコードのインデックス値
        """
        if self.__total_count <= index:
            raise Exception("存在しないデータのインデックスが指定されました。")
        self.__fp.seek(self.__header_size + self.__block_size * index, os.SEEK_SET)
        values = struct.unpack(self.__block_format, self.__fp.read(self.__block_size))
        data_record = []
        head_index = 0
        for column_info in self.__column_info:
            tail_index = column_info['num_elements']
            datum = values[head_index:tail_index]
            if column_info['is_scalar']:
                datum = column_info['element_type'](datum[0])
            else:
                datum = numpy.array(datum).astype(column_info['element_type']).reshape(*(column_info['shape']))
            data_record.append(datum)
        return tuple(data_record)


class BatchLoader:
    """ BatchLoaderクラス

    指定したバッチサイズ単位でデータレコードをロードするイテレータを提供します。
    """
    def __init__(self, packfile_path: str, batch_size: int, down_samples: int = None):
        """ コンストラクタ

        パックファイル名を指定して既存のパックファイルを開きます。
        データ構造はパックファイル内に記録されているデータ構造になります。
        インスタンスはbatch_sizeで指定したレコード数単位でデータを読み込むイテレータを提供します。

        :arg str packfile_path: 開くパックファイル名
        :arg int batch_size: バッチサイズ
        :arg down_samples: データレコードを間引く単位
        """
        self.__loader = Loader(packfile_path)
        self.__batch_size = batch_size
        self.__count = self.__loader.count()
        self.__times = 0
        self.__down_samples = down_samples

    def __del__(self):
        del self.__loader

    def __iter__(self):
        self.__times = 0
        return self

    def __len__(self):
        count = self.__count // self.__down_samples if self.__down_samples is not None else self.__count
        return (count + self.__batch_size - 1) // self.__batch_size

    def __next__(self):
        step_size = self.__down_samples if self.__down_samples is not None else 1

        head_index = self.__times * self.__batch_size * step_size
        tail_index = (self.__times + 1) * self.__batch_size * step_size
        if self.__count <= head_index:
            raise StopIteration()
        tail_index = min(tail_index, self.__count)
        records = [self.__loader.load(index) for index in range(head_index, tail_index, step_size)]
        n = len(records)
        num_column = len(records[0])

        batches = []
        for column_i in range(num_column):
            shape = records[0][column_i].shape
            if len(shape) == 1 and shape[0] == 1:
                batch = numpy.array([record[column_i] for record in records])
            else:
                batch = numpy.zeros(shape=(n, *shape))
                for i in range(n):
                    batch[i] = records[i][column_i]
            batches.append(batch)
        self.__times += 1
        return tuple(batches)
