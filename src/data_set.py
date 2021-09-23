import csv
import functools

import numpy


def _read_csv(file_name):
    reader = csv.reader(open(file_name, "r"), delimiter=",")
    data = list(reader)
    return numpy.squeeze(numpy.array(data).astype("float"))


read_x = functools.partial(_read_csv, file_name="X.csv")
read_y = functools.partial(_read_csv, file_name="Y.csv")
read_x_test = functools.partial(_read_csv, file_name="X_test.csv")
