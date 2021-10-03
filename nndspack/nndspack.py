import struct
import os
import numpy

__all__ = [
    'Packer',
    'Loader'
]

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


def _make_header(total_count, x_datum_sample: numpy.ndarray, y_datum_sample: numpy.ndarray):
    x_v_type = _TYPE_TO_CODE_DICT[x_datum_sample.dtype.type]
    y_v_type = _TYPE_TO_CODE_DICT[y_datum_sample.dtype.type]
    is_x_array = type(x_datum_sample) == numpy.ndarray
    is_y_array = type(y_datum_sample) == numpy.ndarray
    if is_x_array:
        x_dim = len(x_datum_sample.shape)
        x_shape = x_datum_sample.shape
    else:
        x_dim = 1
        x_shape = (1,)
    if is_y_array:
        y_dim = len(y_datum_sample.shape)
        y_shape = y_datum_sample.shape
    else:
        y_dim = 1
        y_shape = (1,)
    header_format = f'IBBBB{x_dim if x_dim != 0 else 1}I{y_dim if y_dim != 0 else 1}I'
    return struct.pack(header_format, total_count, x_v_type, y_v_type, x_dim, y_dim, *x_shape, *y_shape)


def _read_header(fp):
    total_count = struct.unpack('I', fp.read(4))[0]
    x_v_type = _CODE_TO_TYPE_DICT[struct.unpack('B', fp.read(1))[0]]
    y_v_type = _CODE_TO_TYPE_DICT[struct.unpack('B', fp.read(1))[0]]
    x_dim, y_dim = struct.unpack('BB', fp.read(2))
    format = f'{x_dim}I{y_dim}I'
    values = struct.unpack(format, fp.read(4 * (x_dim + y_dim)))
    x_shape = tuple(values[:x_dim])
    y_shape = tuple(values[x_dim:])
    header_size = fp.tell()
    return header_size, total_count, x_v_type, y_v_type, x_shape, y_shape


class Packer:
    def __init__(self, filename, x_datum_sample: numpy.ndarray, y_datum_sample: numpy.ndarray):
        self.__fp = open(filename, 'wb')
        self.__total_count = 0
        self.__x_datum_n = numpy.prod(x_datum_sample.shape) if x_datum_sample.shape != () else 1
        self.__y_datum_n = numpy.prod(y_datum_sample.shape) if y_datum_sample.shape != () else 1
        x_datum_format = f'{self.__x_datum_n}{_TYPE_TO_STRUCT_FORMAT[x_datum_sample.dtype.type]}'
        y_datum_format = f'{self.__y_datum_n}{_TYPE_TO_STRUCT_FORMAT[y_datum_sample.dtype.type]}'
        self.__block_format = f'{x_datum_format}{y_datum_format}'
        self.__fp.write(_make_header(self.__total_count, x_datum_sample, y_datum_sample))

    def __del__(self):
        self.__fp.seek(0, os.SEEK_SET)
        self.__fp.write(struct.pack('I', self.__total_count))
        self.__fp.close()

    def pack(self, x_datum: numpy.ndarray, y_datum: numpy.ndarray):
        is_x_array = type(x_datum) == numpy.ndarray
        is_y_array = type(y_datum) == numpy.ndarray
        x_datum_writable = x_datum.reshape(self.__x_datum_n).tolist() if is_x_array else (x_datum, )
        y_datum_writable = y_datum.reshape(self.__y_datum_n).tolist() if is_y_array else (y_datum, )
        self.__fp.write(struct.pack(self.__block_format, *x_datum_writable, *y_datum_writable))
        self.__total_count += 1
        return None


class Loader:
    def __init__(self, filename):
        self.__fp = open(filename, 'rb')
        header_size, total_count, x_v_type, y_v_type, x_shape, y_shape = _read_header(self.__fp)
        self.__header_size = header_size
        self.__total_count = total_count
        self.__x_v_type = x_v_type
        self.__y_v_type = y_v_type
        self.__x_shape = x_shape
        self.__y_shape = y_shape
        self.__x_n = numpy.prod(x_shape)
        self.__y_n = numpy.prod(y_shape)
        self.__block_size = self.__x_n * _TYPE_TO_NBYTES_DICT[x_v_type] + self.__y_n * _TYPE_TO_NBYTES_DICT[y_v_type]
        self.__block_format = f'{self.__x_n}{_TYPE_TO_STRUCT_FORMAT[x_v_type]}{self.__y_n}{_TYPE_TO_STRUCT_FORMAT[y_v_type]}'

    def __del__(self):
        self.__fp.close()

    def count(self):
        return self.__total_count

    def load(self, index):
        if self.__total_count <= index:
            raise Exception("存在しないデータのインデックスが指定されました。")
        self.__fp.seek(self.__header_size + self.__block_size * index, os.SEEK_SET)
        values = struct.unpack(self.__block_format, self.__fp.read(self.__block_size))
        x_datum = values[:self.__x_n]
        y_datum = values[self.__x_n:]
        if len(self.__x_shape) == 1 and self.__x_shape[0] == 1:
            x_datum = self.__x_v_type(x_datum[0])
        else:
            x_datum = numpy.array(x_datum).astype(self.__x_v_type).reshape(*self.__x_shape)

        if len(self.__y_shape) == 1 and self.__y_shape[0] == 1:
            y_datum = self.__y_v_type(y_datum[0])
        else:
            y_datum = numpy.array(y_datum).astype(self.__y_v_type).reshape(*self.__y_shape)
        return x_datum, y_datum
